"""Experimento de validación: ¿el loop está CERRADO? ¿el agente aprende?

Qué demuestra
-------------
Corre el dominio de referencia A TRAVÉS del núcleo y mide la detección a lo largo del tiempo,
comparando contra el ground truth. Si el loop está cerrado (la verdad diferida realimenta el
modelo), la tasa de FALSOS POSITIVOS debe BAJAR a medida que el agente acumula experiencia,
mientras el recall (detección de anomalías reales) se mantiene alto. Esa curva descendente de
falsos positivos es la evidencia visible de que el agente aprende de sus errores.

Cómo mide
---------
Cada corrección que emite el critic trae su desenlace (TP/FP/TN/FN) y el CICLO en que se hizo la
predicción. Se agrupan las correcciones en ventanas por ciclo de predicción y se computa, por
ventana: tasa de falsos positivos y recall. Se imprime una curva ASCII (sin dependencias) y se
guarda un CSV con las métricas.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Permitir ejecutar el script directamente: agrega la raíz del repo al path de imports.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import AuditTrail, LearningAgentLoop, Outcome, SignatureDeduper
from domains.reference import (
    CsvCollector,
    MovingBaselineModel,
    ReferenceCritic,
    ReferenceEffector,
    ReferenceLearningElement,
    ReferencePerformanceElement,
    ReferenceProblemGenerator,
    SimulatedDeferredTruth,
    generate,
)

WINDOW = 200          # tamaño de ventana (en ciclos) para promediar métricas
TRUTH_DELAY = 10      # ciclos que tarda la verdad diferida en llegar


def build_loop() -> Tuple[LearningAgentLoop, ReferenceEffector]:
    """Cablea el dominio de referencia dentro del núcleo (inyección de dependencias).

    Este cableado es TODO lo que hace falta para correr un dominio: el modelo de normalidad es
    un único objeto compartido entre performance y learning; los cuatro componentes R&N, los
    tres puertos y las capas transversales se pasan al loop del núcleo. El núcleo no cambia."""
    sample_csv, truth_csv, anomalies = generate()  # datos + ground truth reproducibles

    model = MovingBaselineModel()  # <- estado de conocimiento compartido
    loop = LearningAgentLoop(
        collector=CsvCollector(sample_csv),
        performance=ReferencePerformanceElement(model),
        learning=ReferenceLearningElement(model),
        critic=ReferenceCritic(),
        problem_generator=ReferenceProblemGenerator(model),
        effector=(effector := ReferenceEffector(verbose=False)),
        truth_source=SimulatedDeferredTruth(anomalies, delay=TRUTH_DELAY),
        deduper=SignatureDeduper(cooldown=5),
    )
    return loop, effector


def collect_metrics(audit: AuditTrail) -> List[Dict[str, float]]:
    """Agrupa las correcciones por ventana de ciclo de predicción y computa métricas."""
    # Acumuladores por índice de ventana.
    buckets: Dict[int, Dict[str, int]] = {}
    for rec in audit.records:
        for corr in rec.corrections:
            w = corr.prediction.cycle // WINDOW
            b = buckets.setdefault(w, {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
            b[_short(corr.outcome)] += 1

    metrics: List[Dict[str, float]] = []
    for w in sorted(buckets):
        b = buckets[w]
        normals = b["FP"] + b["TN"]
        anomalies = b["TP"] + b["FN"]
        fp_rate = b["FP"] / normals if normals else 0.0
        recall = b["TP"] / anomalies if anomalies else float("nan")
        metrics.append(
            {
                "window": w,
                "cycle_start": w * WINDOW,
                "fp_rate": fp_rate,
                "recall": recall,
                "FP": b["FP"],
                "TN": b["TN"],
                "TP": b["TP"],
                "FN": b["FN"],
            }
        )
    return metrics


def _short(outcome: Outcome) -> str:
    return {
        Outcome.TRUE_POSITIVE: "TP",
        Outcome.FALSE_POSITIVE: "FP",
        Outcome.TRUE_NEGATIVE: "TN",
        Outcome.FALSE_NEGATIVE: "FN",
    }[outcome]


def _bar(value: float, width: int = 40, vmax: float = 0.25) -> str:
    filled = int(round(min(value, vmax) / vmax * width))
    return "#" * filled + "." * (width - filled)


def render_curve(metrics: List[Dict[str, float]]) -> str:
    lines = [
        "",
        "Curva de aprendizaje  (tasa de falsos positivos por ventana de ciclos)",
        "  barra mas corta = menos falsos positivos = agente mas afinado",
        "",
    ]
    for m in metrics:
        rec = m["recall"]
        rec_str = "  n/a" if rec != rec else f"{rec*100:5.1f}%"  # nan check
        lines.append(
            f"  ciclos {m['cycle_start']:>4}+ | FP={m['fp_rate']*100:5.1f}% "
            f"[{_bar(m['fp_rate'])}]  recall={rec_str}  "
            f"(FP={m['FP']} TN={m['TN']} TP={m['TP']} FN={m['FN']})"
        )
    return "\n".join(lines)


def write_csv(metrics: List[Dict[str, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["window", "cycle_start", "fp_rate", "recall", "FP", "TN", "TP", "FN"])
        for m in metrics:
            w.writerow(
                [m["window"], m["cycle_start"], f"{m['fp_rate']:.4f}",
                 f"{m['recall']:.4f}", m["FP"], m["TN"], m["TP"], m["FN"]]
            )


def main() -> None:
    loop, effector = build_loop()
    audit = loop.run()  # corre el ciclo R&N completo sobre todo el stream

    metrics = collect_metrics(audit)

    print("Muestra de la traza auditable (ultimos 12 ciclos):")
    print(audit.render(last=12))

    print(render_curve(metrics))

    out_csv = _ROOT / "data" / "learning_curve.csv"
    write_csv(metrics, out_csv)
    print(f"\nMetricas guardadas en: {out_csv}")
    print(f"Alertas emitidas en total (post dedupe/cooldown): {len(effector.emitted)}")

    # Veredicto legible: comparar primera vs ultima ventana con datos.
    if len(metrics) >= 2:
        first, last = metrics[0]["fp_rate"], metrics[-1]["fp_rate"]
        verdict = "BAJO" if last < first else "NO bajo"
        print(
            f"Falsos positivos: primera ventana={first*100:.1f}%  ultima={last*100:.1f}%  "
            f"-> {verdict}. Loop {'cerrado (aprende)' if last < first else 'revisar'}."
        )


if __name__ == "__main__":
    main()
