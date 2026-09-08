"""Fuente de verdad diferida SIMULADA para el dominio de referencia.

Por qué simulada
----------------
En un dominio real la verdad llega de afuera y tarde (un revisor confirma el fraude semanas
después, una falla se diagnostica al día siguiente, etc.). Para EJERCITAR el critic sin depender
de ese proceso externo, aquí la verdad se conoce de antemano (ground truth) pero se ENTREGA con
un retardo de N ciclos, imitando la demora. Así el loop se comporta como en producción: predice
a ciegas y solo más tarde sabe si acertó.

Contrato
--------
Implementa `DeferredTruthSource`: `register(prediction)` agenda la verdad de esa predicción para
`prediction.cycle + delay`; `arrivals(cycle)` devuelve las que ya vencieron. El retardo vive
aquí, no en el loop, porque 'cuánto tarda en saberse la verdad' es propiedad del entorno.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Sequence, Set, Tuple

from core import DeferredTruth, DeferredTruthSource, Prediction


class SimulatedDeferredTruth(DeferredTruthSource):
    def __init__(self, anomalies: Set[Tuple[str, int]], delay: int = 10) -> None:
        # `anomalies`: conjunto de (entidad, timestamp) que SON anomalías reales (ground truth).
        # Ausencia en el conjunto = normal. Es el estándar de desempeño externo del experimento.
        self._anomalies = anomalies
        # `delay`: cuántos ciclos tarda la verdad en llegar tras la predicción. >0 obliga al
        # agente a predecir sin saber el resultado -> es lo que hace 'diferida' a la verdad.
        self._delay = delay
        # Cola de verdades agendadas: (ciclo_de_maduración, DeferredTruth), en orden de llegada.
        self._scheduled: Deque[Tuple[int, DeferredTruth]] = deque()

    def register(self, prediction: Prediction) -> None:
        was_anomaly = (prediction.entity_id, prediction.timestamp) in self._anomalies
        ready_at = prediction.cycle + self._delay
        self._scheduled.append(
            (
                ready_at,
                DeferredTruth(
                    entity_id=prediction.entity_id,
                    timestamp=prediction.timestamp,
                    was_anomaly=was_anomaly,
                ),
            )
        )

    def arrivals(self, cycle: int) -> Sequence[DeferredTruth]:
        # Como las predicciones se registran en orden creciente de ciclo, los vencimientos
        # también salen en orden: basta con desencolar desde el frente mientras ya maduraron.
        ready: List[DeferredTruth] = []
        while self._scheduled and self._scheduled[0][0] <= cycle:
            _, truth = self._scheduled.popleft()
            ready.append(truth)
        return ready
