"""Modelo de normalidad del comportamiento de los agentes observados.

Qué cambia respecto del dominio de referencia (y qué NO)
---------------------------------------------------------
La mecánica de aprendizaje es la misma y a propósito: baseline móvil por entidad (Welford) más
un umbral ADAPTATIVO que la verdad diferida corrige. Se conserva porque es transparente — se
puede leer, ciclo a ciclo, por qué el agente decidió distinto — y esa transparencia es un
requisito del proyecto, no una preferencia estética.

Lo que cambia es QUÉ es una entidad: acá una entidad es `<agente>:<dimensión-de-comportamiento>`.
El agente aprende que 4 segundos entre pasos es normal para este agente pero 40 no, y lo aprende
por separado para cada dimensión y para cada agente del esquema.

Lo que este archivo AGREGA: persistencia
-----------------------------------------
El dominio de referencia corre un experimento por lotes y muere; podía permitirse tener el
conocimiento solo en memoria. Este dominio corre como daemon en una VPS durante semanas y se
reinicia con cada `docker compose up`. Sin persistencia, cada deploy borraría todo lo aprendido
y la curva de aprendizaje volvería a cero — el agente nunca acumularía experiencia.

La escritura es ATÓMICA (archivo temporal + `os.replace`): si el proceso muere a mitad de un
guardado, en disco queda el snapshot anterior íntegro, nunca un JSON truncado que impida
arrancar.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# Versión del formato del snapshot. Si el modelo cambia de forma, un snapshot viejo se descarta
# en vez de cargarse mal: es preferible re-aprender que aprender sobre un estado corrupto.
_SNAPSHOT_VERSION = 1


@dataclass
class _EntityBaseline:
    """Estadística móvil de una entidad + su umbral adaptativo (Welford online)."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    # Arranca deliberadamente sensible: el agente empieza alertando de más y debe APRENDER a
    # callarse. Un umbral inicial alto escondería el aprendizaje.
    threshold_sigmas: float = 2.0

    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)


class BehaviorNormalityModel:
    """Normalidad por `<agente>:<dimensión>` con umbral adaptativo y snapshot en disco.

    Args:
        warmup: observaciones necesarias antes de emitir juicios sobre una entidad. Sin
            normalidad aprendida no hay desvío que reportar.
        clean_guard_sigmas: por encima de este desvío la observación se considera sospechosa y
            NO se pliega al baseline, para que un episodio anómalo no contamine la normalidad.
        threshold_step / min_threshold / max_threshold: cuánto y hasta dónde mueve el umbral
            cada corrección. Es el mecanismo VISIBLE del aprendizaje correctivo.
        state_path: dónde persistir el conocimiento. `None` = solo en memoria (tests).
    """

    def __init__(
        self,
        warmup: int = 20,
        clean_guard_sigmas: float = 4.0,
        threshold_step: float = 0.15,
        min_threshold: float = 1.5,
        max_threshold: float = 6.0,
        state_path: Optional[Path | str] = None,
    ) -> None:
        self.warmup = warmup
        self.clean_guard_sigmas = clean_guard_sigmas
        self.threshold_step = threshold_step
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self._state_path = Path(state_path) if state_path else None
        self._entities: Dict[str, _EntityBaseline] = {}
        self._seen: Dict[str, int] = {}
        # Contadores de corrección por entidad: no alimentan la decisión, pero hacen legible
        # (en la meta-alerta y en el log) CUÁNTO error tuvo el agente en cada dimensión.
        self._false_positives: Dict[str, int] = {}
        self._false_negatives: Dict[str, int] = {}

    def _baseline(self, entity_id: str) -> _EntityBaseline:
        if entity_id not in self._entities:
            self._entities[entity_id] = _EntityBaseline()
        return self._entities[entity_id]

    # --- Lectura: la usa el PerformanceElement -------------------------------------------

    def deviation_sigmas(self, entity_id: str, value: float) -> float:
        """Desvío respecto de la normalidad de la entidad, en sigmas (con signo)."""
        b = self._baseline(entity_id)
        if b.count < self.warmup:
            return 0.0
        s = b.std()
        if s == 0.0:
            return 0.0
        return (value - b.mean) / s

    def threshold(self, entity_id: str) -> float:
        return self._baseline(entity_id).threshold_sigmas

    def is_warm(self, entity_id: str) -> bool:
        return self._baseline(entity_id).count >= self.warmup

    def seen_count(self, entity_id: str) -> int:
        return self._seen.get(entity_id, 0)

    def entities(self):
        return tuple(self._seen.keys())

    def stats(self, entity_id: str) -> dict:
        """Estado legible de una entidad (para la meta-alerta y la exploración con LLM)."""
        b = self._baseline(entity_id)
        return {
            "entity": entity_id,
            "count": b.count,
            "mean": round(b.mean, 4),
            "std": round(b.std(), 4),
            "threshold_sigmas": round(b.threshold_sigmas, 3),
            "false_positives": self._false_positives.get(entity_id, 0),
            "false_negatives": self._false_negatives.get(entity_id, 0),
        }

    # --- Escritura: la usa el LearningElement --------------------------------------------

    def learn_observation(self, entity_id: str, value: float) -> None:
        """Aprendizaje continuo: pliega la observación al baseline si parece limpia."""
        b = self._baseline(entity_id)
        self._seen[entity_id] = self._seen.get(entity_id, 0) + 1
        if b.count < self.warmup:
            b.update(value)
            return
        dev = abs(self.deviation_sigmas(entity_id, value))
        if dev <= self.clean_guard_sigmas:
            b.update(value)

    def adjust_threshold(self, entity_id: str, *, too_sensitive: bool) -> None:
        """Aprendizaje correctivo: la verdad diferida mueve el umbral de esta entidad."""
        b = self._baseline(entity_id)
        if too_sensitive:
            self._false_positives[entity_id] = self._false_positives.get(entity_id, 0) + 1
            b.threshold_sigmas = min(
                self.max_threshold, b.threshold_sigmas + self.threshold_step
            )
        else:
            self._false_negatives[entity_id] = self._false_negatives.get(entity_id, 0) + 1
            b.threshold_sigmas = max(
                self.min_threshold, b.threshold_sigmas - self.threshold_step
            )

    # --- Persistencia --------------------------------------------------------------------

    def save(self) -> bool:
        """Escribe el conocimiento aprendido a disco de forma atómica.

        Returns:
            `True` si se guardó; `False` si no hay ruta configurada o falló la escritura.
            Nunca lanza: perder un snapshot no puede tumbar al agente.
        """
        if self._state_path is None:
            return False
        snapshot = {
            "version": _SNAPSHOT_VERSION,
            "warmup": self.warmup,
            "entities": {
                eid: {
                    "count": b.count,
                    "mean": b.mean,
                    "m2": b.m2,
                    "threshold_sigmas": b.threshold_sigmas,
                    "seen": self._seen.get(eid, 0),
                    "false_positives": self._false_positives.get(eid, 0),
                    "false_negatives": self._false_negatives.get(eid, 0),
                }
                for eid, b in self._entities.items()
            },
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp, self._state_path)  # atómico: nunca queda un JSON a medio escribir
            return True
        except OSError:
            return False

    def load(self) -> bool:
        """Restaura el conocimiento de un snapshot previo, si existe y es compatible.

        Returns:
            `True` si se restauró algo. Un snapshot ausente, ilegible o de otra versión se
            ignora en silencio: el agente arranca en frío, que es un estado válido.
        """
        if self._state_path is None or not self._state_path.exists():
            return False
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if data.get("version") != _SNAPSHOT_VERSION:
            return False

        for eid, raw in (data.get("entities") or {}).items():
            baseline = _EntityBaseline(
                count=int(raw.get("count", 0)),
                mean=float(raw.get("mean", 0.0)),
                m2=float(raw.get("m2", 0.0)),
                threshold_sigmas=float(raw.get("threshold_sigmas", 2.0)),
            )
            self._entities[eid] = baseline
            self._seen[eid] = int(raw.get("seen", baseline.count))
            self._false_positives[eid] = int(raw.get("false_positives", 0))
            self._false_negatives[eid] = int(raw.get("false_negatives", 0))
        return bool(self._entities)
