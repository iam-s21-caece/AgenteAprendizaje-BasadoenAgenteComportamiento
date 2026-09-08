"""Fuente de verdad DIFERIDA Y DERIVADA: el desenlace del análisis observado.

Por qué derivada (y no un revisor humano)
------------------------------------------
En este dominio no hace falta que alguien etiquete a mano: el entorno ya produce la verdad por
su cuenta. El agente observado, cada vez que analiza, termina emitiendo una alerta con
contenido o no emitiendo nada. Ese desenlace es un hecho externo al meta-agente — no lo decide
él, ni su LLM — y por eso sirve como estándar de desempeño en el sentido de R&N.

    was_anomaly = NOT productivo
    (productivo = el análisis terminó en una alerta con CVEs o severidad > info)

Traducido: el meta-agente aprende a anticipar, mirando cómo se comporta un análisis EN CURSO,
si ese análisis va a terminar en algo o va a girar en falso.

Por qué el diferimiento es real y no cosmético
-----------------------------------------------
Es la objeción obvia contra una verdad derivada de los mismos datos, así que conviene dejarla
cerrada: la verdad de un análisis vive en su evento `analysis_end`, que el colector todavía NO
leyó cuando emite las observaciones de ese análisis (lee el archivo append-only en orden, sin
lookahead). Al momento de predecir, la información que define la verdad no existe en el stream.
El retardo no lo impone esta clase por decreto — lo impone la estructura del dato.

De ahí que `arrivals()` sea tan simple: no agenda nada por número de ciclos. Solo pregunta al
libro "¿ya cerró este análisis?" y libera lo que corresponda. El calendario lo pone el entorno.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from core import DeferredTruth, DeferredTruthSource, Prediction

from .ledger import AnalysisLedger


class DerivedDeferredTruth(DeferredTruthSource):
    """Libera la verdad de cada predicción cuando cierra el análisis que la originó.

    Args:
        ledger: libro compartido donde el colector registra apertura, avance y cierre.
        max_pending: tope de predicciones retenidas a la espera de veredicto. Es una red de
            seguridad para un daemon largo: si por algún motivo un análisis nunca cerrara ni
            fuera barrido, las predicciones más viejas se descartan sin juzgar en vez de hacer
            crecer la memoria sin límite. Descartar sin juzgar es preferible a inventar una
            verdad que nadie confirmó.
    """

    def __init__(self, ledger: AnalysisLedger, max_pending: int = 20_000) -> None:
        self._ledger = ledger
        self._max_pending = max_pending
        # (analysis_id, prediction) en orden de llegada.
        self._pending: List[Tuple[str, Prediction]] = []

    def register(self, prediction: Prediction) -> None:
        """El loop entrega cada predicción; se retiene hasta que su análisis tenga desenlace."""
        analysis_id = self._ledger.analysis_for(prediction.timestamp)
        if analysis_id is None:
            # Sin análisis asociado no hay verdad posible: no se retiene (y no se inventa).
            return
        self._pending.append((analysis_id, prediction))
        if len(self._pending) > self._max_pending:
            del self._pending[: len(self._pending) - self._max_pending]

    def arrivals(self, cycle: int) -> Sequence[DeferredTruth]:
        """Devuelve las verdades cuyos análisis ya cerraron (puede ser vacío)."""
        if not self._pending:
            return ()

        ready: List[DeferredTruth] = []
        still_waiting: List[Tuple[str, Prediction]] = []
        resolved: set = set()

        for analysis_id, prediction in self._pending:
            verdict = self._ledger.verdict(analysis_id)
            if verdict is None:
                # Análisis todavía en curso: nadie puede corregir al agente aún. Esto es
                # exactamente lo que mantiene abierta la ventana de incertidumbre.
                still_waiting.append((analysis_id, prediction))
                continue
            ready.append(
                DeferredTruth(
                    entity_id=prediction.entity_id,
                    timestamp=prediction.timestamp,
                    was_anomaly=not verdict.productive,
                )
            )
            resolved.add(analysis_id)

        self._pending = still_waiting

        # Higiene: un veredicto ya consumido por todas sus predicciones no se necesita más.
        waiting_ids = {aid for aid, _ in still_waiting}
        for analysis_id in resolved - waiting_ids:
            self._ledger.forget(analysis_id)

        return ready

    def pending_count(self) -> int:
        """Predicciones a la espera de veredicto (visibilidad operativa)."""
        return len(self._pending)
