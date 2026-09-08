"""Implementación de referencia del PerformanceElement.

Percibe una observación, consulta el modelo de normalidad compartido y decide si alertar.
No aprende: solo aplica el conocimiento actual (fiel a la frontera de R&N). El modelo se
inyecta en el constructor y es el MISMO objeto que modifica el LearningElement; así 'decidir
mejor' es consecuencia automática de 'haber aprendido'."""

from __future__ import annotations

from core import Decision, Observation, PerformanceElement

from .normality_model import MovingBaselineModel

# Clave de la dimensión que este dominio de referencia observa. El contrato admite varias
# dimensiones; el ejemplo usa una sola a propósito, para mantenerse mínimo.
VALUE_KEY = "value"


class ReferencePerformanceElement(PerformanceElement):
    def __init__(self, model: MovingBaselineModel) -> None:
        # El modelo llega por inyección: el performance element NO lo crea ni lo posee en
        # exclusiva. Es conocimiento compartido con el learning element.
        self._model = model

    def evaluate(self, obs: Observation, cycle: int) -> Decision:
        value = obs.values[VALUE_KEY]
        dev = self._model.deviation_sigmas(obs.entity_id, value)

        # Alertar si el desvío (en magnitud) supera el umbral ADAPTATIVO de la entidad.
        # El umbral no es fijo: la verdad diferida lo movió en ciclos anteriores. Por eso el
        # mismo desvío puede alertar hoy y no mañana -> el aprendizaje se ve en la decisión.
        threshold = self._model.threshold(obs.entity_id)
        alerted = self._model.is_warm(obs.entity_id) and abs(dev) >= threshold

        # Firma = entidad: para el dedupe/cooldown, 'la misma alerta' es 'la misma entidad
        # alertando'. El dominio define esto porque sabe qué alertas son 'la misma'.
        signature = f"entity:{obs.entity_id}"

        return Decision(deviation_sigmas=dev, alerted=alerted, signature=signature)
