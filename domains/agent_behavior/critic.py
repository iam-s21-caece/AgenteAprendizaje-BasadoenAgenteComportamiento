"""Critic del dominio: clasifica predicción contra verdad diferida.

Es idéntico al de referencia y eso es una señal de que la abstracción del núcleo aguanta: al
cambiar de dominio, el componente que compara "lo que hice" con "lo que era" no necesitó
cambiar nada. Existe como archivo propio del dominio (en vez de reusar el de referencia) para
no crear una dependencia entre dominios, que dejaría a `agent_behavior` atado a un molde de
ejemplo.

Lo importante es lo que este componente NO hace: no consulta al LLM y no inventa la verdad.
El estándar de desempeño es externo — sale de `truth.was_anomaly`, que en este dominio proviene
del desenlace real del análisis observado. Si el juicio lo emitiera el propio meta-agente (o su
LLM), el sistema estaría calificándose a sí mismo y el aprendizaje no significaría nada.
"""

from __future__ import annotations

from core import Correction, Critic, DeferredTruth, Outcome, Prediction


class BehaviorCritic(Critic):
    def judge(self, prediction: Prediction, truth: DeferredTruth) -> Correction:
        outcome = Outcome.of(
            alerted=prediction.alerted, was_anomaly=truth.was_anomaly
        )
        return Correction(prediction=prediction, truth=truth, outcome=outcome)
