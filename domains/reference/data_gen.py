"""Generador de datos sintéticos con ground truth conocido.

Por qué existe
--------------
El experimento de validación necesita una VERDAD conocida para poder medir si el agente mejora.
Datos reales no traen ground truth limpio; por eso se sintetizan: cada entidad tiene una
normalidad definida (media/desvío) y se inyectan desvíos en momentos que NOSOTROS controlamos.
Esos momentos son exactamente la verdad diferida que el critic recibirá tarde.

Produce dos archivos, respetando la separación de contratos:
  - reference_sample.csv : (entity, timestamp, value)  -> lo lee el CsvCollector (formato).
  - reference_truth.csv  : (entity, timestamp)          -> anomalías; lo lee la fuente de verdad.
El CSV de observación NO incluye la etiqueta: el agente no debe 'ver' la verdad al percibir; la
verdad llega por el canal diferido, no por el sensor. Mantener los archivos separados hace
cumplir esa frontera de forma tangible.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Ubicación por defecto de los datos (carpeta data/ en la raíz del repo).
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SAMPLE_CSV = _DATA_DIR / "reference_sample.csv"
TRUTH_CSV = _DATA_DIR / "reference_truth.csv"

# Normalidad por entidad: (media, desvío). Escalas distintas a propósito, para mostrar que el
# desvío en sigmas normaliza entidades heterogéneas.
_ENTITIES: Dict[str, Tuple[float, float]] = {
    "sensor_A": (100.0, 5.0),
    "sensor_B": (20.0, 2.0),
    "sensor_C": (0.0, 1.0),
    "sensor_D": (50.0, 8.0),
    "sensor_E": (200.0, 10.0),
}


def generate(
    n_steps: int = 400,
    anomaly_rate: float = 0.03,
    anomaly_sigmas: float = 6.0,
    warmup_gap: int = 25,
    seed: int = 42,
) -> Tuple[Path, Path, Set[Tuple[str, int]]]:
    """Genera los CSV y devuelve (ruta_muestra, ruta_verdad, conjunto_de_anomalías).

    Parámetros (justificados):
    - `warmup_gap`: no se inyectan anomalías antes de este timestamp, para que el baseline tenga
      tiempo de aprender la normalidad primero (comparable al warmup del modelo). Inyectar antes
      mediría al agente cuando todavía no puede saber nada -> no sería una prueba justa.
    - `anomaly_sigmas`: magnitud del desvío inyectado. Grande (6s) para que sea detectable en
      principio; el mérito del agente es NO alertar de más sobre el ruido normal, no adivinar
      desvíos imperceptibles.
    """
    rng = random.Random(seed)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[str, int, float]] = []
    anomalies: Set[Tuple[str, int]] = set()

    # Interleave por timestamp: en cada 'tick' emite todas las entidades. Así el loop avanza en
    # ciclos que mezclan entidades, como un flujo real de observaciones concurrentes.
    for t in range(n_steps):
        for entity, (mean, std) in _ENTITIES.items():
            if t >= warmup_gap and rng.random() < anomaly_rate:
                # Desvío inyectado: signo aleatorio, magnitud fija en sigmas. Es ground truth.
                sign = 1.0 if rng.random() < 0.5 else -1.0
                value = mean + sign * anomaly_sigmas * std
                anomalies.add((entity, t))
            else:
                # Observación normal ~ N(mean, std).
                value = rng.gauss(mean, std)
            rows.append((entity, t, value))

    with SAMPLE_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["entity", "timestamp", "value"])
        for entity, t, value in rows:
            w.writerow([entity, t, f"{value:.6f}"])

    with TRUTH_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["entity", "timestamp"])
        for entity, t in sorted(anomalies):
            w.writerow([entity, t])

    return SAMPLE_CSV, TRUTH_CSV, anomalies


def load_anomalies(truth_csv: Path = TRUTH_CSV) -> Set[Tuple[str, int]]:
    """Carga el conjunto de anomalías (entidad, timestamp) desde el CSV de verdad."""
    anomalies: Set[Tuple[str, int]] = set()
    with Path(truth_csv).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            anomalies.add((row["entity"], int(row["timestamp"])))
    return anomalies


if __name__ == "__main__":
    sample, truth, anomalies = generate()
    print(f"Generado: {sample}  ({sum(1 for _ in sample.open()) - 1} filas)")
    print(f"Generado: {truth}   ({len(anomalies)} anomalías inyectadas)")
