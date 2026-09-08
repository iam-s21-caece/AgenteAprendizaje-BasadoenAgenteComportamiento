"""Experimento de validación del dominio `agent_behavior`: ¿el meta-agente aprende?

Qué demuestra
-------------
Corre el meta-agente sobre trazas sintéticas con ground truth conocido y mide, a lo largo del
tiempo, si acierta al anticipar qué análisis del agente vigilado van a terminar sin producir
nada. Si el loop está cerrado, la PRECISIÓN debe subir a medida que el agente acumula
experiencia (deja de marcar análisis que después resultan productivos) sin desplomar el recall.

Por qué se mide por EPISODIO y no por observación
--------------------------------------------------
Un análisis del agente vigilado produce decenas de observaciones, y la verdad es del análisis
entero. Medir por observación mezclaría dos cosas distintas: la calidad del juicio y cuántas
veces se observó el mismo episodio. La pregunta operativa real es por episodio: *de los
análisis que marqué, ¿cuántos efectivamente no produjeron nada?* y *de los que no produjeron
nada, ¿cuántos marqué?*. Eso es precisión y recall a nivel de análisis.

Uso:
    python experiments/behavior_learning_curve.py [n_analisis]
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import LearningAgentLoop, SignatureDeduper
from domains.agent_behavior.audit import BoundedAuditTrail
from domains.agent_behavior.behavior_model import BehaviorNormalityModel
from domains.agent_behavior.critic import BehaviorCritic
from domains.agent_behavior.effector import MetaAlertEffector
from domains.agent_behavior.learning_element import BehaviorLearningElement
from domains.agent_behavior.ledger import AnalysisLedger
from domains.agent_behavior.performance_element import BehaviorPerformanceElement
from domains.agent_behavior.problem_generator import BehaviorProblemGenerator
from domains.agent_behavior.trace_collector import TraceTailCollector
from domains.agent_behavior.trace_gen import generate
from domains.agent_behavior.truth_source import DerivedDeferredTruth

WINDOW = 20          # análisis por ventana de medición
AGENT_ID = "keycloak-agent"


class _FiniteCollector(TraceTailCollector):
    """Colector que termina cuando se agotan las trazas, en vez de esperar en vivo.

    El colector de producción es un daemon: si no hay eventos nuevos, duerme y sigue. Para un
    experimento reproducible hace falta lo contrario — que el stream termine — así que se
    sobreescribe el único punto donde difiere el comportamiento. El resto (parseo, tail
    incremental, orden de eventos) es exactamente el mismo código que corre en la VPS: si se
    reimplementara acá, el experimento validaría otra cosa que la que se despliega.
    """

    def stream(self):
        for path in self._discover_files():
            yield from self._read_new(path)


def run(n_analyses: int) -> Dict:
    """Genera datos, corre el loop completo y devuelve la evidencia recolectada."""
    workdir = Path(tempfile.mkdtemp(prefix="meta-agent-exp-"))
    try:
        traces_dir, unproductive = generate(workdir / "traces", analyses=n_analyses)

        model = BehaviorNormalityModel(state_path=None)   # sin persistencia: corrida limpia
        ledger = AnalysisLedger()
        audit = BoundedAuditTrail(max_records=200_000)

        collector = _FiniteCollector(
            traces_dir=traces_dir, ledger=ledger, agent_id=AGENT_ID, from_start=True
        )
        learning = BehaviorLearningElement(model, ledger, save_every=10**9)

        loop = LearningAgentLoop(
            collector=collector,
            performance=BehaviorPerformanceElement(model, ledger),
            learning=learning,
            critic=BehaviorCritic(),
            problem_generator=BehaviorProblemGenerator(model, audit, llm=None, every=500),
            effector=MetaAlertEffector(
                alerts_path=workdir / "meta-alerts.jsonl",
                model=model,
                ledger=ledger,
                echo=False,
            ),
            truth_source=DerivedDeferredTruth(ledger),
            audit=audit,
            deduper=SignatureDeduper(cooldown=50),
        )
        loop.run()

        return _episodes(audit, ledger, unproductive, model)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _episodes(audit, ledger, unproductive, model) -> Dict:
    """Reduce la traza por ciclos a una tabla por episodio (análisis).

    Un episodio se considera MARCADO si el meta-agente alertó sobre cualquiera de sus
    dimensiones mientras el análisis estaba en curso.
    """
    order: List[str] = []
    flagged: Dict[str, bool] = {}
    warm: Dict[str, bool] = {}

    for record in audit.records:
        analysis_id = ledger.analysis_for(record.observation.timestamp)
        if analysis_id is None:
            continue
        if analysis_id not in flagged:
            order.append(analysis_id)
            flagged[analysis_id] = False
            warm[analysis_id] = False
        if record.prediction.alerted:
            flagged[analysis_id] = True
        # Un episodio solo es evaluable si el agente ya tenía normalidad aprendida: durante el
        # warmup no puede alertar, y contarlo como error sería medir el arranque en frío en vez
        # del aprendizaje.
        if model.is_warm(record.observation.entity_id):
            warm[analysis_id] = True

    rows = []
    for analysis_id in order:
        if not warm[analysis_id]:
            continue
        rows.append(
            {
                "analysis_id": analysis_id,
                "flagged": flagged[analysis_id],
                "unproductive": analysis_id in unproductive,
            }
        )
    return {"episodes": rows, "model": model}


def windows(rows: List[Dict]) -> List[Dict]:
    """Agrupa episodios en ventanas y computa precisión/recall por ventana."""
    out: List[Dict] = []
    for start in range(0, len(rows), WINDOW):
        chunk = rows[start : start + WINDOW]
        if len(chunk) < WINDOW // 2:
            break  # ventana final incompleta: no se reporta para no engañar con ruido
        tp = sum(1 for r in chunk if r["flagged"] and r["unproductive"])
        fp = sum(1 for r in chunk if r["flagged"] and not r["unproductive"])
        fn = sum(1 for r in chunk if not r["flagged"] and r["unproductive"])
        tn = sum(1 for r in chunk if not r["flagged"] and not r["unproductive"])
        out.append(
            {
                "episode_start": start,
                "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
                "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
            }
        )
    return out


def _bar(value: float, width: int = 30) -> str:
    if value != value:  # nan
        return " " * width
    filled = int(round(value * width))
    return "#" * filled + "." * (width - filled)


def _pct(value: float) -> str:
    return "  n/a" if value != value else f"{value * 100:5.1f}%"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    result = run(n)
    rows = result["episodes"]
    model = result["model"]
    metrics = windows(rows)

    print(f"\nEpisodios evaluables (post warmup): {len(rows)} de {n} análisis simulados")
    print(f"Improductivos reales: {sum(1 for r in rows if r['unproductive'])}\n")
    print("Curva de aprendizaje por ventana de episodios")
    print("  precision = de los análisis que marqué, cuántos no produjeron nada")
    print("  recall    = de los que no produjeron nada, cuántos marqué\n")
    for m in metrics:
        print(
            f"  episodios {m['episode_start']:>4}+ | "
            f"precision={_pct(m['precision'])} [{_bar(m['precision'])}] "
            f"recall={_pct(m['recall'])}  "
            f"(TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']})"
        )

    print("\nConocimiento final por dimensión de comportamiento:")
    for entity in sorted(model.entities()):
        stats = model.stats(entity)
        print(
            f"  {stats['entity']:<42} umbral={stats['threshold_sigmas']:.2f} sigmas  "
            f"media={stats['mean']:.1f}  desvio={stats['std']:.1f}  "
            f"(FP={stats['false_positives']} FN={stats['false_negatives']})"
        )

    out_csv = _ROOT / "data" / "behavior_learning_curve.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode_start", "precision", "recall", "TP", "FP", "FN", "TN"])
        for m in metrics:
            writer.writerow(
                [
                    m["episode_start"],
                    f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}",
                    m["TP"],
                    m["FP"],
                    m["FN"],
                    m["TN"],
                ]
            )
    print(f"\nMetricas guardadas en: {out_csv}")

    if len(metrics) >= 2:
        first, last = metrics[0]["precision"], metrics[-1]["precision"]
        if first == first and last == last:
            verdict = "SUBIO" if last > first else "no subio"
            print(
                f"Precision: primera ventana={first*100:.1f}%  ultima={last*100:.1f}%  "
                f"-> {verdict}."
            )


if __name__ == "__main__":
    main()
