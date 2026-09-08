"""Critic: compara el desempeño contra un estándar EXTERNO y emite la señal de corrección."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .contracts import Correction, DeferredTruth, Prediction


class Critic(ABC):
    """R&N 'critic': juzga qué tan bien actuó el agente según un estándar de desempeño.

    Por qué el estándar de desempeño es EXTERNO
    -------------------------------------------
    En R&N el critic no puede derivar el estándar del agente mismo: si el agente definiera
    su propio éxito, aprendería a satisfacerse en lugar de a mejorar. Aquí ese estándar externo
    es la VERDAD DIFERIDA (pieza 3 del contrato de aplicabilidad): una señal que llega desde
    fuera del agente y confirma, tarde, si el desvío era real. El critic no la produce ni sabe
    de dónde sale (un revisor humano, un proceso posterior, una medición tardía); solo la recibe.

    Qué rol cumple la verdad diferida
    ---------------------------------
    Es lo que hace 'diferido' a este problema y lo que da sentido al learning element: la
    calidad de una predicción no se conoce en el momento de predecir, sino después. El critic
    es el puente temporal: asocia una predicción PASADA con su verdad TARDÍA y traduce el par
    a una `Correction` que el learning element puede usar. No decide cómo ajustar el modelo
    (eso es del learning element); solo dictamina 'esto estuvo bien / mal, y de qué tipo'.
    """

    @abstractmethod
    def judge(self, prediction: Prediction, truth: DeferredTruth) -> Correction:
        """Asocia una predicción pasada con su verdad diferida y emite la corrección.
        La implementación clasifica el desenlace (TP/FP/TN/FN) comparando `alerted` con
        `was_anomaly`; el núcleo ofrece `Outcome.of(...)` para no reinventar esa tabla."""
        raise NotImplementedError
