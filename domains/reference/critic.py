"""Implementación de referencia del Critic.

Su única tarea: dada una predicción pasada y la verdad diferida que por fin llegó, clasificar
el desenlace y empaquetarlo como `Correction`. No sabe de dónde salió la verdad (se la pasa el
loop desde la fuente de verdad diferida) ni decide cómo ajustar el modelo (eso es del learning
element). El estándar de desempeño es EXTERNO: proviene de `truth.was_anomaly`, no del agente."""

from __future__ import annotations

from core import Correction, Critic, DeferredTruth, Outcome, Prediction


class ReferenceCritic(Critic):
    def judge(self, prediction: Prediction, truth: DeferredTruth) -> Correction:
        # Comparar lo que el agente HIZO (alertar o no) contra lo que la verdad diferida dice
        # que ERA (anomalía o no). `Outcome.of` es la tabla de confusión del núcleo: se reusa
        # para no reinventar la clasificación TP/FP/TN/FN.
        outcome = Outcome.of(alerted=prediction.alerted, was_anomaly=truth.was_anomaly)
        return Correction(prediction=prediction, truth=truth, outcome=outcome)
