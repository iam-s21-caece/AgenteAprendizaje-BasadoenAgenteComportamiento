"""Colector de referencia: lee un CSV simple (entidad, timestamp, valor).

ESTE ES EL ÚNICO LUGAR DEL PROYECTO QUE CONOCE UN FORMATO.
El resto del sistema trabaja con `Observation`, agnóstico al origen. Por eso el borde de
formato está aislado aquí: cumplir la regla núcleo/dominio depende de que ninguna otra pieza
lea archivos ni parsee.

EXTENSIÓN FUTURA (documentada, NO implementada a propósito)
-----------------------------------------------------------
Otros orígenes —JSON, PDF, imágenes, una base externa, una API— serían COLECTORES ADICIONALES
que implementan el mismo contrato `Collector.stream() -> Iterator[Observation]`. Es decir: se
agrega un archivo nuevo tipo `json_collector.py` que produce las mismas `Observation`, y NADA
más cambia (ni el núcleo ni los otros componentes del dominio). Ese es el beneficio de haber
aislado el formato en el colector. No se implementan ahora por decisión explícita de alcance.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from core import Collector, Observation

from .performance_element import VALUE_KEY


class CsvCollector(Collector):
    """Emite una `Observation` por fila del CSV, en el orden en que aparecen.

    Se asume que el CSV ya viene ordenado temporalmente (así lo genera `data_gen.py`).
    Columnas esperadas: entity, timestamp, value."""

    def __init__(self, csv_path: str | Path) -> None:
        self._path = Path(csv_path)

    def stream(self) -> Iterator[Observation]:
        with self._path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield Observation(
                    entity_id=row["entity"],
                    timestamp=int(row["timestamp"]),
                    # El contrato admite varias dimensiones; el ejemplo mapea la única columna
                    # numérica a la clave estándar del dominio.
                    values={VALUE_KEY: float(row["value"])},
                )
