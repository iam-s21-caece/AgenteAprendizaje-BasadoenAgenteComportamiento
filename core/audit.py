"""Capa transversal: traza auditable de cada ciclo del loop.

Por qué existe (no es un 'log' cualquiera)
-------------------------------------------
El requisito central de este agente es que el aprendizaje sea VISIBLE, no una caja negra. La
traza es la evidencia de que el loop de R&N está cerrado: por cada ciclo deja registrado qué
se percibió, qué se decidió, si se actuó, qué verdad llegó y qué corrección produjo. Vive en el
núcleo porque audita el LOOP (algo del núcleo), no un dominio particular. El experimento de
validación lee esta traza para construir la curva de aprendizaje.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from .contracts import (
    Correction,
    ExplorationSuggestion,
    Observation,
    Prediction,
)


@dataclass(frozen=True)
class CycleRecord:
    """Fotografía legible de un ciclo del loop. Un registro = un giro de la rueda R&N."""

    cycle: int
    observation: Observation
    prediction: Prediction
    emitted: bool                               # ¿la alerta pasó el filtro dedupe/cooldown?
    corrections: Sequence[Correction]           # verdades que maduraron en este ciclo
    suggestions: Sequence[ExplorationSuggestion]

    def render(self) -> str:
        """Una línea humana por ciclo: lo mínimo para 'ver' el loop trabajando."""
        obs_vals = ", ".join(f"{k}={v:.3f}" for k, v in self.observation.values.items())
        head = (
            f"[c{self.cycle:04d}] {self.observation.entity_id} t={self.observation.timestamp} "
            f"({obs_vals}) dev={self.prediction.deviation_sigmas:+.2f}s "
            f"alert={'Y' if self.prediction.alerted else 'N'} "
            f"emit={'Y' if self.emitted else 'N'}"
        )
        if self.corrections:
            fb = " | ".join(
                f"verdad<-c{c.prediction.cycle}:{c.outcome.name}" for c in self.corrections
            )
            head += f"  >> {fb}"
        if self.suggestions:
            head += "  ?? " + "; ".join(s.reason for s in self.suggestions)
        return head


class AuditTrail:
    """Acumula los `CycleRecord` y los ofrece para inspección o para el experimento.

    Es deliberadamente pasiva y en memoria: el núcleo no impone dónde se persiste la traza
    (archivo, consola, base) — esa decisión, de nuevo, es de quien orqueste el dominio."""

    def __init__(self) -> None:
        self._records: List[CycleRecord] = []

    def record(self, rec: CycleRecord) -> None:
        self._records.append(rec)

    @property
    def records(self) -> Sequence[CycleRecord]:
        return tuple(self._records)

    def render(self, last: int | None = None) -> str:
        """Devuelve la traza como texto; `last` limita a los N ciclos finales."""
        rows = self._records if last is None else self._records[-last:]
        return "\n".join(r.render() for r in rows)
