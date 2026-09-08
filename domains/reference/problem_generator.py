"""Implementación de referencia del ProblemGenerator.

Estrategia de exploración deliberadamente simple: cada cierto intervalo, sugerir vigilar la
entidad MENOS observada. Justificación contra R&N: la explotación (mirar donde el modelo ya
espera desvíos) deja entidades poco vistas con baselines pobres; esas son justamente donde el
agente es más ciego. Proponerlas es generar experiencia nueva que el aprendizaje por
explotación no produciría. Se mantiene mínimo a propósito, pero respeta el contrato para poder
crecer sin tocar el núcleo."""

from __future__ import annotations

from typing import Sequence

from core import ExplorationSuggestion, ProblemGenerator

from .normality_model import MovingBaselineModel


class ReferenceProblemGenerator(ProblemGenerator):
    def __init__(self, model: MovingBaselineModel, every: int = 50) -> None:
        self._model = model
        # `every`: cada cuántos ciclos emitir una sugerencia. Explorar en CADA ciclo sería
        # ruido; explorar nunca sería estancarse. El intervalo hace la exploración dosificada.
        self._every = every

    def suggest(self, cycle: int) -> Sequence[ExplorationSuggestion]:
        # Solo proponer en los ciclos múltiplos del intervalo, y una vez que hay entidades.
        if cycle == 0 or cycle % self._every != 0:
            return ()
        entities = self._model.entities()
        if not entities:
            return ()
        # Entidad con menor conteo de observaciones = la más subvigilada.
        least = min(entities, key=self._model.seen_count)
        return (
            ExplorationSuggestion(
                entity_id=least,
                dimension=None,
                reason=f"vigilar '{least}': es la entidad menos observada "
                f"({self._model.seen_count(least)} obs) y su normalidad está peor aprendida",
            ),
        )
