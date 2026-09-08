"""Núcleo agnóstico al dominio del agente de aprendizaje (taxonomía R&N).

Regla NO NEGOCIABLE: nada en este paquete conoce un dominio o un formato concreto. Aquí solo
viven los CONTRATOS (clases base abstractas + estructuras tipadas), las capas transversales del
loop y el orquestador. Los dominios (ver `domains/`) implementan estos contratos; el núcleo
nunca sabe cuál corre.

Familia de problemas soportada (contrato de aplicabilidad): detección sobre normalidad con
verdad diferida. Ver `core.contracts` para el detalle de las tres piezas exigidas al dominio.
"""

from .audit import AuditTrail, CycleRecord
from .contracts import (
    Correction,
    Decision,
    DeferredTruth,
    ExplorationSuggestion,
    Observation,
    Outcome,
    Prediction,
)
from .critic import Critic
from .dedup import SignatureDeduper
from .learning_element import LearningElement
from .loop import LearningAgentLoop
from .performance_element import PerformanceElement
from .ports import Collector, DeferredTruthSource, Effector
from .problem_generator import ProblemGenerator

__all__ = [
    # Estructuras
    "Observation",
    "Decision",
    "Prediction",
    "DeferredTruth",
    "Outcome",
    "Correction",
    "ExplorationSuggestion",
    # Componentes R&N (contratos)
    "PerformanceElement",
    "LearningElement",
    "Critic",
    "ProblemGenerator",
    # Puertos de dominio
    "Collector",
    "Effector",
    "DeferredTruthSource",
    # Transversales + orquestador
    "AuditTrail",
    "CycleRecord",
    "SignatureDeduper",
    "LearningAgentLoop",
]
