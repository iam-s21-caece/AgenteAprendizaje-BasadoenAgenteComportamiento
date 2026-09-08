"""ProblemGenerator: propone exploración deliberada para que el agente no se estanque."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .contracts import ExplorationSuggestion


class ProblemGenerator(ABC):
    """R&N 'problem generator': sugiere acciones exploratorias, no solo explotación.

    Por qué la exploración es aprendizaje y no ruido
    ------------------------------------------------
    El performance element, guiado por el modelo actual, tiende a mirar siempre donde ya sabe
    que puede pasar algo (explotación). Pero la normalidad evoluciona y hay entidades o
    dimensiones que hoy nadie vigila: si el agente nunca las observa, nunca podrá aprender su
    normalidad ni detectar sus desvíos. R&N introduce el problem generator justamente para
    forzar experiencia NUEVA que el aprendizaje por explotación jamás generaría. Sus
    sugerencias son hipótesis de 'acá podría haber algo que no estás midiendo', deliberadas y
    auditables (cada una lleva su `reason`), no aleatoriedad porque sí.

    En el esqueleto se mantiene simple a propósito (p. ej. 'vigilá la entidad menos observada'),
    pero el contrato deja lugar a estrategias de exploración más ricas sin tocar el núcleo.
    """

    @abstractmethod
    def suggest(self, cycle: int) -> Sequence[ExplorationSuggestion]:
        """Devuelve entidades/dimensiones a vigilar que normalmente no se vigilan.
        Puede devolver vacío: no todo ciclo requiere explorar."""
        raise NotImplementedError
