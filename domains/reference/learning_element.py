"""Implementación de referencia del LearningElement.

Es el único componente que ESCRIBE en el modelo de normalidad. Traduce las dos entradas de
R&N a dos operaciones concretas sobre el baseline móvil:
  - `observe`     -> aprendizaje continuo (sigue la normalidad que evoluciona).
  - `incorporate` -> aprendizaje correctivo (la verdad diferida ajusta el umbral).
Mantener las dos separadas hace legible qué causó cada cambio del modelo."""

from __future__ import annotations

from core import Correction, LearningElement, Observation, Outcome

from .normality_model import MovingBaselineModel
from .performance_element import VALUE_KEY


class ReferenceLearningElement(LearningElement):
    def __init__(self, model: MovingBaselineModel) -> None:
        # Mismo objeto de modelo que consulta el performance element: aprender aquí cambia
        # cómo decide allá. Ese acoplamiento por el modelo compartido ES el aprendizaje de R&N.
        self._model = model

    def observe(self, obs: Observation) -> None:
        # Experiencia directa: el baseline se refina con cada observación (el modelo decide
        # internamente si la observación es 'limpia' antes de plegarla, para no contaminarse).
        self._model.learn_observation(obs.entity_id, obs.values[VALUE_KEY])

    def incorporate(self, correction: Correction) -> None:
        # Retroalimentación del critic: solo los ERRORES mueven el umbral. Los aciertos
        # (TP/TN) confirman el modelo actual y no requieren ajuste -> el agente no se
        # desestabiliza cuando ya está acertando.
        entity_id = correction.prediction.entity_id
        if correction.outcome is Outcome.FALSE_POSITIVE:
            # Alertó de más: era normal. El umbral estaba demasiado sensible -> agrandarlo.
            self._model.adjust_threshold(entity_id, too_sensitive=True)
        elif correction.outcome is Outcome.FALSE_NEGATIVE:
            # No alertó y era anomalía: demasiado tolerante -> achicar el umbral.
            self._model.adjust_threshold(entity_id, too_sensitive=False)
