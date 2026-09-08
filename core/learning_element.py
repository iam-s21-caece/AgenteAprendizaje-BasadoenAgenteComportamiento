"""LearningElement: el ÚNICO componente que modifica el conocimiento del agente."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .contracts import Correction, Observation


class LearningElement(ABC):
    """R&N 'learning element': mejora el modelo de normalidad a partir de la experiencia.

    Por qué es el único que escribe el modelo
    -----------------------------------------
    R&N centraliza el aprendizaje en un componente para que el resto del agente pueda actuar
    de forma estable y para que TODA modificación del conocimiento sea rastreable a una causa
    (una observación o una corrección). Esto es lo contrario de una caja negra.

    Dos entradas, fieles a R&N
    --------------------------
    1. `observe`: experiencia directa. Con cada observación limpia el modelo de normalidad se
       refina (el baseline sigue la evolución natural de lo 'normal' en el tiempo).
    2. `incorporate`: retroalimentación del critic. Cuando la verdad diferida revela un error,
       el modelo se ajusta. ESTE método es el que cierra el loop de R&N: la verdad tardía
       cambia cómo el agente predecirá en el futuro. Sin él, el agente nunca aprendería de
       sus aciertos y errores, solo derivaría con los datos.
    """

    @abstractmethod
    def observe(self, obs: Observation) -> None:
        """Actualiza el modelo de normalidad con una observación (aprendizaje continuo).
        El dominio decide si la observación es 'limpia' y cuánto pesa en el baseline."""
        raise NotImplementedError

    @abstractmethod
    def incorporate(self, correction: Correction) -> None:
        """Ajusta el modelo ante un error o acierto revelado por la verdad diferida.
        Es el retorno del loop: convierte 'me equivoqué' en 'la próxima predigo distinto'."""
        raise NotImplementedError
