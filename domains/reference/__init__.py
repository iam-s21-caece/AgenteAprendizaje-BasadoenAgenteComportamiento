"""Dominio de REFERENCIA: implementación mínima y comentada de los contratos del núcleo.

Doble propósito
---------------
1. Demostrar que el núcleo (agnóstico) puede resolver un caso concreto de la familia 'detección
   sobre normalidad con verdad diferida'.
2. Ser el MOLDE que se copia para crear dominios reales. Para pivotar a un dominio nuevo:
   copiar esta carpeta, rellenar las cuatro piezas R&N + los puertos con la lógica del nuevo
   dominio, y NO tocar `core/`. Ver README.

Todo lo específico de dominio y de FORMATO vive aquí (nunca en `core/`).
"""

from .collector import CsvCollector
from .critic import ReferenceCritic
from .data_gen import generate, load_anomalies
from .effector import ReferenceEffector
from .learning_element import ReferenceLearningElement
from .normality_model import MovingBaselineModel
from .performance_element import ReferencePerformanceElement
from .problem_generator import ReferenceProblemGenerator
from .truth_source import SimulatedDeferredTruth

__all__ = [
    "CsvCollector",
    "ReferenceCritic",
    "ReferenceEffector",
    "ReferenceLearningElement",
    "MovingBaselineModel",
    "ReferencePerformanceElement",
    "ReferenceProblemGenerator",
    "SimulatedDeferredTruth",
    "generate",
    "load_anomalies",
]
