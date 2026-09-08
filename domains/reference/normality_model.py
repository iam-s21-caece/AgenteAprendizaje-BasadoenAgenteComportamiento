"""Modelo de normalidad del dominio de referencia: baseline estadístico móvil por entidad.

Por qué este archivo NO está en el núcleo
-----------------------------------------
'Qué es normal' es conocimiento de dominio. El núcleo trata el modelo como opaco (solo recibe
un desvío en sigmas). Este modelo es la implementación concreta de la pieza (2) del contrato de
aplicabilidad. Es deliberadamente simple y transparente: media y desvío móviles por entidad,
más un umbral ADAPTATIVO por entidad que la verdad diferida ajusta. Nada de caja negra: se
puede leer, ciclo a ciclo, cómo cambia.

Quién lo usa
------------
Un ÚNICO objeto de este tipo se inyecta tanto en el PerformanceElement (lo consulta para
puntuar y decidir) como en el LearningElement (lo modifica). Ese estado compartido es el
'conocimiento del agente' de R&N; por eso el núcleo no lo define: lo cablea el dominio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class _EntityBaseline:
    """Estadística móvil de una entidad + su umbral adaptativo.

    Usa el método de Welford (conteo, media, M2) para media y varianza online: permite seguir
    una normalidad que evoluciona sin guardar todo el histórico, y es transparente (cada
    actualización es una fórmula legible, no un modelo entrenado aparte)."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0                       # suma de cuadrados de diferencias (Welford)
    # Umbral de alerta ACTUAL (en sigmas); la verdad diferida lo mueve. Arranca DELIBERADAMENTE
    # sensible (2.0s): el agente empieza alertando de mas y debe aprender a callarse. Un umbral
    # inicial alto escondería el aprendizaje (ya casi no habría falsos positivos que corregir).
    threshold_sigmas: float = 2.0

    def std(self) -> float:
        # Desvío estándar muestral; indefinido con menos de 2 muestras.
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))

    def update(self, value: float) -> None:
        """Incorpora una observación limpia al baseline (Welford online)."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)


class MovingBaselineModel:
    """Modelo de normalidad por entidad con umbral adaptativo por verdad diferida.

    Parámetros (todos justificados, ninguno 'porque sí'):
    - `warmup`: cuántas observaciones se necesitan antes de emitir juicios. Sin warmup, con 0-1
      muestras el desvío no está definido y el agente alertaría sobre ruido. Refleja que la
      normalidad hay que APRENDERLA antes de poder detectar desvíos.
    - `clean_guard_sigmas`: por encima de este desvío, la observación se considera sospechosa y
      NO se pliega al baseline. Evita que una anomalía real contamine la media/desvío ('cada
      observación LIMPIA actualiza el baseline', del enunciado).
    - `threshold_step` / `min_threshold` / `max_threshold`: cuánto y hasta dónde mueve el umbral
      cada corrección de la verdad diferida. Es el mecanismo VISIBLE de aprendizaje correctivo.
    """

    def __init__(
        self,
        warmup: int = 20,
        clean_guard_sigmas: float = 4.0,
        threshold_step: float = 0.15,
        min_threshold: float = 1.5,
        max_threshold: float = 6.0,
    ) -> None:
        self.warmup = warmup
        self.clean_guard_sigmas = clean_guard_sigmas
        self.threshold_step = threshold_step
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self._entities: Dict[str, _EntityBaseline] = {}
        # Conteo de observaciones por entidad: lo consume el problem generator para saber
        # cuál se vigila menos. Vive acá porque es parte del 'estado de conocimiento'.
        self._seen: Dict[str, int] = {}

    def _baseline(self, entity_id: str) -> _EntityBaseline:
        if entity_id not in self._entities:
            self._entities[entity_id] = _EntityBaseline()
        return self._entities[entity_id]

    # --- Lectura: la usa el PerformanceElement -------------------------------------------

    def deviation_sigmas(self, entity_id: str, value: float) -> float:
        """Desvío del valor respecto de la normalidad de la entidad, en sigmas (con signo).
        Devuelve 0.0 durante el warmup o si el desvío aún no es medible: sin normalidad
        aprendida no hay desvío que reportar."""
        b = self._baseline(entity_id)
        if b.count < self.warmup:
            return 0.0
        s = b.std()
        if s == 0.0:
            return 0.0
        return (value - b.mean) / s

    def threshold(self, entity_id: str) -> float:
        """Umbral de alerta actual de la entidad (en sigmas). Lo mueve la verdad diferida."""
        return self._baseline(entity_id).threshold_sigmas

    def is_warm(self, entity_id: str) -> bool:
        return self._baseline(entity_id).count >= self.warmup

    def seen_count(self, entity_id: str) -> int:
        return self._seen.get(entity_id, 0)

    def entities(self):
        return tuple(self._seen.keys())

    # --- Escritura: la usa el LearningElement --------------------------------------------

    def learn_observation(self, entity_id: str, value: float) -> None:
        """Aprendizaje continuo: pliega la observación al baseline SOLO si parece limpia.
        Durante el warmup se pliega todo (hay que arrancar el baseline de algún lado)."""
        b = self._baseline(entity_id)
        self._seen[entity_id] = self._seen.get(entity_id, 0) + 1
        if b.count < self.warmup:
            b.update(value)
            return
        dev = abs(self.deviation_sigmas(entity_id, value))
        if dev <= self.clean_guard_sigmas:
            b.update(value)   # limpia: refina la normalidad
        # si dev > guard: no se pliega, para no contaminar el baseline con una posible anomalía

    def adjust_threshold(self, entity_id: str, *, too_sensitive: bool) -> None:
        """Aprendizaje correctivo: la verdad diferida movió el umbral de esta entidad.
        - too_sensitive=True  (falso positivo): el umbral era muy chico -> agrandarlo.
        - too_sensitive=False (falso negativo): el umbral era muy grande -> achicarlo.
        Este es el punto donde se VE al agente aprender de sus errores."""
        b = self._baseline(entity_id)
        if too_sensitive:
            b.threshold_sigmas = min(self.max_threshold, b.threshold_sigmas + self.threshold_step)
        else:
            b.threshold_sigmas = max(self.min_threshold, b.threshold_sigmas - self.threshold_step)
