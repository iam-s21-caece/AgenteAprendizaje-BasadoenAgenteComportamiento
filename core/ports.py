"""Puertos de entrada/salida: los 'sensores' y 'actuadores' de R&N como contratos abstractos.

Por qué viven en el núcleo pero los implementa el dominio
---------------------------------------------------------
El loop de R&N necesita PERCIBIR (sensores) y ACTUAR (actuadores), y necesita una fuente para
la verdad diferida. Pero CÓMO se percibe (leer un CSV, una API, una DB) y CÓMO se actúa
(loguear, mandar un mail) es específico del dominio. El núcleo define la FORMA del puerto
(qué entra, qué sale) y el dominio la rellena. Así el loop se escribe una sola vez y funciona
con cualquier dominio que respete estos contratos.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Sequence

from .contracts import DeferredTruth, Observation, Prediction


class Collector(ABC):
    """Sensores de R&N. Traduce la realidad cruda del dominio a `Observation` agnósticas.

    Es el ÚNICO lugar autorizado a conocer formatos (CSV, JSON, DB...). El núcleo consume el
    stream sin saber su origen. Justificación: aísla toda la dependencia de formato en un
    borde, cumpliendo la regla núcleo/dominio."""

    @abstractmethod
    def stream(self) -> Iterator[Observation]:
        """Emite observaciones en orden temporal, una por 'tick' del mundo."""
        raise NotImplementedError


class Effector(ABC):
    """Actuadores de R&N. Ejecuta la acción externa que decidió el performance element.

    En esta familia la acción típica es 'emitir/registrar una alerta'. Se abstrae porque el
    efecto real (log, notificación, ticket) es del dominio; el núcleo solo sabe 'actuá con
    esta predicción'."""

    @abstractmethod
    def act(self, prediction: Prediction) -> None:
        """Materializa la alerta correspondiente a una predicción."""
        raise NotImplementedError


class DeferredTruthSource(ABC):
    """El mecanismo de VERDAD DIFERIDA (pieza 3 del contrato de aplicabilidad).

    Se separa del Critic a propósito: el critic sabe CÓMO juzgar, pero no debe asumir de dónde
    sale la verdad ni cuándo llega. Esta fuente modela ambas cosas: el loop le ENTREGA cada
    predicción (`register`) y ella decide CUÁNDO madura su verdad (el retardo N vive acá, no en
    el loop, porque 'cuánto tarda en saberse la verdad' es una propiedad del entorno/dominio).
    Cada ciclo el loop le pregunta qué verdades ya llegaron (`arrivals`)."""

    @abstractmethod
    def register(self, prediction: Prediction) -> None:
        """El entorno solo puede producir verdad sobre predicciones que ocurrieron: el loop
        registra cada predicción y la fuente agenda su verdad para más adelante."""
        raise NotImplementedError

    @abstractmethod
    def arrivals(self, cycle: int) -> Sequence[DeferredTruth]:
        """Devuelve las verdades cuya demora ya venció en este ciclo (puede ser vacío)."""
        raise NotImplementedError
