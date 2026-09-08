"""LearningAgentLoop: el ciclo de R&N que ATA las cuatro piezas y CIERRA el bucle.

Por qué este archivo es el corazón del núcleo
---------------------------------------------
R&N describe un learning agent como cuatro componentes conectados por un flujo. Este loop ES
ese flujo, escrito una sola vez y agnóstico al dominio. Su responsabilidad única es la
ORQUESTACIÓN: percibir -> decidir -> actuar -> (más tarde) recibir verdad -> criticar -> aprender.
No contiene lógica de dominio ni de formato; solo coordina los contratos.

El cierre del loop (requisito, no opcional)
-------------------------------------------
La verdad diferida realimenta el aprendizaje: una predicción hecha en el ciclo C se retiene,
su verdad madura ciclos después, el critic la juzga y el learning element ajusta el modelo. A
partir de ahí el performance element predice distinto. Ese retorno es lo que convierte a esto
en un AGENTE DE APRENDIZAJE y no en un mero detector estático.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .audit import AuditTrail, CycleRecord
from .contracts import Prediction
from .critic import Critic
from .dedup import SignatureDeduper
from .learning_element import LearningElement
from .performance_element import PerformanceElement
from .ports import Collector, DeferredTruthSource, Effector
from .problem_generator import ProblemGenerator


class LearningAgentLoop:
    """Orquestador del agente de aprendizaje de R&N.

    Recibe por inyección de dependencias las cuatro piezas R&N, los tres puertos de dominio y
    las dos capas transversales. No construye nada específico: todo lo específico llega ya
    resuelto por el dominio. Esta inversión de dependencias es lo que hace que el núcleo nunca
    'sepa' qué dominio corre."""

    def __init__(
        self,
        collector: Collector,
        performance: PerformanceElement,
        learning: LearningElement,
        critic: Critic,
        problem_generator: ProblemGenerator,
        effector: Effector,
        truth_source: DeferredTruthSource,
        audit: Optional[AuditTrail] = None,
        deduper: Optional[SignatureDeduper] = None,
    ) -> None:
        self._collector = collector
        self._performance = performance
        self._learning = learning
        self._critic = critic
        self._problem_generator = problem_generator
        self._effector = effector
        self._truth_source = truth_source
        self._audit = audit or AuditTrail()
        self._deduper = deduper or SignatureDeduper()

        # Predicciones a la espera de su verdad diferida, indexadas por (entidad, timestamp).
        # Es la 'memoria' que permite que el critic empareje pasado con verdad tardía.
        self._pending: Dict[Tuple[str, int], Prediction] = {}

    def run(self, max_cycles: Optional[int] = None) -> AuditTrail:
        """Corre el loop sobre el stream del colector y devuelve la traza auditable.

        Un ciclo = una observación. El orden de los pasos respeta el flujo de R&N; los
        comentarios numerados marcan cada componente para que el loop se lea como el diagrama."""
        for cycle, obs in enumerate(self._collector.stream()):
            if max_cycles is not None and cycle >= max_cycles:
                break

            # (1) PERCIBIR + (2) DECIDIR  -- PerformanceElement con el modelo actual.
            decision = self._performance.evaluate(obs, cycle)
            prediction = Prediction.stamp(decision, obs, cycle)

            # (3) APRENDER (continuo) -- el baseline sigue la normalidad que evoluciona.
            #     Va antes de actuar para que el modelo incorpore la experiencia directa;
            #     el ajuste por verdad (correctivo) llega más abajo, cuando esa verdad exista.
            self._learning.observe(obs)

            # (4) ACTUAR -- solo si se alertó Y la capa de dedupe/cooldown lo permite.
            #     La predicción se registra igual aunque no se emita: el aprendizaje no debe
            #     perder información por una supresión que es puramente de higiene de salida.
            emitted = False
            if prediction.alerted and self._deduper.should_emit(prediction.signature, cycle):
                self._effector.act(prediction)
                emitted = True

            # Entregar la predicción al entorno para que agende su verdad diferida, y retenerla.
            self._truth_source.register(prediction)
            self._pending[(obs.entity_id, obs.timestamp)] = prediction

            # (5) RECIBIR VERDAD + (6) CRITICAR + (7) APRENDER (correctivo) -- cierre del loop.
            corrections = []
            for truth in self._truth_source.arrivals(cycle):
                past = self._pending.pop((truth.entity_id, truth.timestamp), None)
                if past is None:
                    continue  # verdad sin predicción retenida: nada que juzgar.
                correction = self._critic.judge(past, truth)
                self._learning.incorporate(correction)   # <-- aquí el modelo cambia por la verdad.
                corrections.append(correction)

            # (8) EXPLORAR -- el problem generator propone dónde mirar fuera de lo habitual.
            suggestions = self._problem_generator.suggest(cycle)

            # (9) TRAZAR -- dejar el ciclo auditable (evidencia de que el bucle giró completo).
            self._audit.record(
                CycleRecord(
                    cycle=cycle,
                    observation=obs,
                    prediction=prediction,
                    emitted=emitted,
                    corrections=tuple(corrections),
                    suggestions=tuple(suggestions),
                )
            )

        return self._audit
