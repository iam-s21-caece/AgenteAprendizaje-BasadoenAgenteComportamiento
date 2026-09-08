"""PerformanceElement: el componente de R&N que PERCIBE y DECIDE."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .contracts import Decision, Observation


class PerformanceElement(ABC):
    """R&N 'performance element': aplica el conocimiento ACTUAL para elegir una acción.

    Rol y frontera
    --------------
    - Percibe una `Observation` y decide: ¿cuánto se desvía de lo normal? ¿amerita alertar?
    - NO aprende. Modificar el conocimiento es trabajo exclusivo del LearningElement (R&N
      separa deliberadamente 'actuar bien con lo que sé' de 'mejorar lo que sé'). Si el
      performance element aprendiera, no habría forma de auditar qué cambió el modelo.
    - Consulta el modelo de normalidad que le inyecta el dominio; el núcleo trata ese modelo
      como opaco (no sabe qué es 'normal', solo recibe un desvío en sigmas).

    Por qué mide el desvío en sigmas
    --------------------------------
    Es la cuantificación agnóstica de la pieza (2) del contrato de aplicabilidad: 'una noción
    de normalidad'. Sigmas normaliza dominios con escalas distintas a una unidad común.
    """

    @abstractmethod
    def evaluate(self, obs: Observation, cycle: int) -> Decision:
        """Puntúa el desvío y decide si alertar.

        Recibe `cycle` solo para que el dominio pueda construir la firma de la alerta si la
        firma depende del tiempo; la contabilidad del ciclo la lleva el loop, no el dominio.
        Devuelve una `Decision` (contenido puro); el loop la sella como `Prediction`."""
        raise NotImplementedError
