"""Estructuras tipadas que intercambian el núcleo y los dominios.

Por qué este archivo existe
---------------------------
El núcleo (R&N: performance element, learning element, critic, problem generator)
necesita un vocabulario COMÚN para hablar con cualquier dominio sin conocer el dominio.
Estas estructuras SON ese vocabulario. Son 100% agnósticas: no saben qué es una entidad
concreta, ni de qué formato salieron (CSV/JSON/DB), ni qué significa "normal" en el dominio.

Contrato de aplicabilidad (la familia de problemas que este agente resuelve)
----------------------------------------------------------------------------
Detección sobre normalidad con verdad diferida. Un dominio es compatible si puede proveer:
    (1) observaciones por entidad en el tiempo   -> `Observation`
    (2) una noción de normalidad                 -> se cuantifica como desvío en sigmas
    (3) un mecanismo de verdad diferida          -> `DeferredTruth`
Cada estructura de abajo mapea a una de esas tres piezas o al canal critic->learning de R&N.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class Observation:
    """Qué recibe el agente, sin saber de dónde salió.

    Justificación (contrato de aplicabilidad, pieza 1): 'observaciones por entidad en el
    tiempo'. El núcleo solo exige entidad + orden temporal + valores numéricos. El dominio
    es el ÚNICO que traduce su formato crudo a esta estructura; el núcleo nunca ve un CSV.
    `values` es un Mapping para admitir varias dimensiones (el dominio de referencia usa una
    sola clave, pero el contrato no lo obliga)."""

    entity_id: str
    timestamp: int                    # orden temporal; el dominio decide la unidad
    values: Mapping[str, float]       # dimensiones numéricas observadas


@dataclass(frozen=True)
class Decision:
    """Salida del PerformanceElement: el juicio del agente con su conocimiento ACTUAL.

    Justificación (R&N performance element): percibe y decide, pero no aprende ni conoce el
    reloj del loop. Devuelve solo el contenido de la decisión; el loop la sella con el ciclo
    para producir la `Prediction` auditable. Separar Decision de Prediction evita que el
    dominio tenga que conocer la numeración de ciclos del núcleo."""

    deviation_sigmas: float           # cuánto se aparta de lo normal (contrato pieza 2)
    alerted: bool                     # ¿el desvío amerita actuar?
    signature: str                    # identidad de la alerta para dedupe/cooldown


@dataclass(frozen=True)
class Prediction:
    """Registro auditable de una decisión pasada, retenido hasta que llegue su verdad.

    Justificación: sin esto el loop NO puede cerrarse. El critic necesita 'la predicción
    pasada' para asociarla con la verdad tardía. Guarda el ciclo (cuándo se decidió) porque
    la verdad diferida madura relativa a ese momento."""

    cycle: int
    entity_id: str
    timestamp: int
    deviation_sigmas: float
    alerted: bool
    signature: str

    @classmethod
    def stamp(cls, decision: Decision, obs: Observation, cycle: int) -> "Prediction":
        """El loop 'sella' una Decision con contexto (entidad, tiempo, ciclo).
        Concentra en un solo lugar cómo se convierte una decisión en registro auditable."""
        return cls(
            cycle=cycle,
            entity_id=obs.entity_id,
            timestamp=obs.timestamp,
            deviation_sigmas=decision.deviation_sigmas,
            alerted=decision.alerted,
            signature=decision.signature,
        )


@dataclass(frozen=True)
class DeferredTruth:
    """La verdad que llega DESPUÉS de la predicción (contrato de aplicabilidad, pieza 3).

    Justificación: es el estándar de desempeño EXTERNO. El agente no puede declararse exitoso
    por su cuenta; alguien fuera del agente confirma si el desvío era real. Apunta a la
    observación original (entity_id + timestamp) que ahora, tardíamente, se puede juzgar."""

    entity_id: str
    timestamp: int
    was_anomaly: bool


class Outcome(Enum):
    """Las cuatro combinaciones predicción-vs-verdad. Base de la señal de corrección.

    Justificación: el critic no inventa una métrica arbitraria; usa la matriz de confusión
    clásica porque es la traducción mínima y transparente de 'acerté / me equivoqué' que el
    learning element puede consumir para ajustar en la dirección correcta."""

    TRUE_POSITIVE = auto()      # alertó y era anomalía        -> acierto
    FALSE_POSITIVE = auto()     # alertó y era normal          -> demasiado sensible
    TRUE_NEGATIVE = auto()      # no alertó y era normal        -> acierto
    FALSE_NEGATIVE = auto()     # no alertó y era anomalía      -> demasiado tolerante

    @staticmethod
    def of(alerted: bool, was_anomaly: bool) -> "Outcome":
        if alerted and was_anomaly:
            return Outcome.TRUE_POSITIVE
        if alerted and not was_anomaly:
            return Outcome.FALSE_POSITIVE
        if not alerted and was_anomaly:
            return Outcome.FALSE_NEGATIVE
        return Outcome.TRUE_NEGATIVE


@dataclass(frozen=True)
class Correction:
    """La señal que el Critic emite y el LearningElement consume (canal critic->learning de R&N).

    Justificación: transporta el ERROR (qué predicción, contra qué verdad, con qué desenlace),
    NO la solución. El critic detecta la discrepancia; el learning element decide cómo ajustar
    el modelo. Mantener esa división es lo que evita una caja negra: cada corrección es legible."""

    prediction: Prediction
    truth: DeferredTruth
    outcome: Outcome


@dataclass(frozen=True)
class ExplorationSuggestion:
    """Una propuesta del ProblemGenerator: dónde mirar fuera de la vigilancia habitual.

    Justificación (R&N problem generator): la exploración deliberada es parte del aprendizaje,
    no ruido. Un agente que solo revisa lo que ya sabe revisar nunca descubre desvíos en
    entidades o dimensiones que hoy ignora. `reason` deja la sugerencia auditable."""

    entity_id: Optional[str]          # entidad a vigilar (None = sugerencia por dimensión)
    dimension: Optional[str]          # dimensión a vigilar (None = sugerencia por entidad)
    reason: str
