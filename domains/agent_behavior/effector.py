"""Effector del dominio: materializa la meta-alerta.

Qué escribe y DÓNDE (la restricción que ordena todo el despliegue)
-------------------------------------------------------------------
El meta-agente NO escribe una sola línea en el territorio del agente observado: ese directorio
se monta de solo lectura. Sus meta-alertas van a su propio volumen. Es lo que permite sostener
la promesa de que sumar esta capa no altera en nada el comportamiento del agente vigilado — la
observación no debe perturbar lo observado.

Formato: JSONL, deliberadamente el mismo patrón que usa el agente observado para sus alertas.
Un archivo append-only por línea es directamente consumible con `tail -f`, y el día que el
esquema crezca a cuatro agentes y el volumen justifique una base de datos, cambiar el destino
es escribir OTRO `Effector` — el núcleo y el resto del dominio no se enteran.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core import Effector, Prediction

from .ledger import AnalysisLedger


class MetaAlertEffector(Effector):
    """Persiste la meta-alerta en JSONL y la imprime por stdout.

    Args:
        alerts_path: archivo destino (en el volumen propio del meta-agente).
        model: modelo de normalidad, para adjuntar el estado que produjo la decisión.
        ledger: libro de análisis, para atribuir la meta-alerta al análisis observado.
        agent_id: identificador del meta-agente (quién alerta).
        echo: si `True`, además imprime por stdout (visible en `docker logs`).
    """

    def __init__(
        self,
        alerts_path: Path | str,
        model,
        ledger: AnalysisLedger,
        agent_id: str = "meta-agent",
        echo: bool = True,
    ) -> None:
        self._path = Path(alerts_path)
        self._model = model
        self._ledger = ledger
        self._agent_id = agent_id
        self._echo = echo

    def act(self, prediction: Prediction) -> None:
        """Emite la meta-alerta correspondiente a una predicción alertada."""
        analysis_id = self._ledger.analysis_for(prediction.timestamp)
        observed_agent, _, dimension = prediction.entity_id.partition(":")

        record = {
            "meta_alert_id": f"meta-{prediction.cycle}-{prediction.timestamp}",
            "emitted_at": datetime.now(timezone.utc).isoformat(),
            "emitted_by": self._agent_id,
            "observed_agent": observed_agent,
            "dimension": dimension,
            "analysis_id": analysis_id,
            "cycle": prediction.cycle,
            "observation_timestamp_us": prediction.timestamp,
            "deviation_sigmas": round(prediction.deviation_sigmas, 3),
            # El estado del conocimiento EN EL MOMENTO de decidir. Sin esto la meta-alerta no
            # sería auditable: no se podría reconstruir por qué este desvío superó el umbral
            # hoy cuando ayer no lo habría superado.
            "model_state": self._model.stats(prediction.entity_id),
            "summary": self._summarize(observed_agent, dimension, prediction),
        }

        line = json.dumps(record, ensure_ascii=False, default=str)
        self._write(line)
        if self._echo:
            print(line, flush=True)

    # --- Internos --------------------------------------------------------------------------

    @staticmethod
    def _summarize(agent: str, dimension: str, prediction: Prediction) -> str:
        """Narrativa determinista de la meta-alerta.

        Es texto plantillado a propósito, no generado por el LLM: la meta-alerta debe poder
        emitirse aunque no haya API key, y debe decir exactamente lo que la evidencia dice.
        El aporte del LLM en este dominio está en la exploración (ver `problem_generator`),
        donde equivocarse es barato porque solo propone dónde mirar.
        """
        direction = "por encima" if prediction.deviation_sigmas > 0 else "por debajo"
        return (
            f"El agente '{agent}' presenta un comportamiento anómalo en '{dimension}': "
            f"{abs(prediction.deviation_sigmas):.2f} sigmas {direction} de su normalidad "
            f"aprendida durante un análisis en curso."
        )

    def _write(self, line: str) -> None:
        """Anexa la meta-alerta al JSONL. Nunca propaga: alertar no puede tumbar el loop."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            print(
                json.dumps(
                    {
                        "level": "error",
                        "msg": "no se pudo persistir la meta-alerta",
                        "error": str(exc),
                    }
                ),
                flush=True,
            )
