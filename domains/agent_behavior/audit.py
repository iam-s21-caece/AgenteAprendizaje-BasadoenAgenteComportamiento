"""Traza auditable acotada, para operación continua.

Por qué hace falta una variante y por qué NO se tocó el núcleo
---------------------------------------------------------------
`core.AuditTrail` acumula un `CycleRecord` por ciclo en una lista y nunca la poda. Es la
decisión correcta para lo que fue escrito: un experimento por lotes que corre unos miles de
ciclos, termina, y cuya traza completa es justamente el resultado (la curva de aprendizaje se
construye leyéndola entera).

Este dominio la usa distinto: un daemon que corre semanas y procesa un evento tras otro. Con la
traza sin podar, la memoria del contenedor crecería hasta que el proceso muera — no por un
error, sino por diseño de otro caso de uso.

En vez de cambiar el núcleo para acomodar este dominio (que es exactamente lo que el proyecto
prohíbe), se extiende por subclase: misma interfaz, misma semántica, ventana acotada. El loop
sigue recibiendo un `AuditTrail` y no se entera; `render()` y `records` siguen funcionando,
solo que sobre los últimos N ciclos.
"""

from __future__ import annotations

from core import AuditTrail, CycleRecord


class BoundedAuditTrail(AuditTrail):
    """`AuditTrail` con ventana deslizante de los últimos `max_records` ciclos.

    Args:
        max_records: cuántos ciclos se retienen. Tiene que alcanzar para el contexto que el
            problem generator le pasa al LLM y para inspeccionar el pasado reciente por
            `docker logs`; más allá de eso, la evidencia duradera son las meta-alertas
            persistidas, no la traza en memoria.
    """

    def __init__(self, max_records: int = 5_000) -> None:
        super().__init__()
        self._max_records = max(1, max_records)
        self._total = 0

    def record(self, rec: CycleRecord) -> None:
        super().record(rec)
        self._total += 1
        # Poda por lotes en vez de en cada inserción: recortar una lista de miles de elementos
        # una vez cada 10% es notablemente más barato que hacer un `pop(0)` por ciclo.
        excess = len(self._records) - self._max_records
        if excess > self._max_records // 10:
            del self._records[:excess]

    @property
    def total_cycles(self) -> int:
        """Ciclos vistos desde el arranque (la ventana olvida; el contador no)."""
        return self._total
