"""LearningElement del dominio: el único componente que ESCRIBE en el conocimiento.

Traduce las dos entradas de R&N a dos operaciones sobre el modelo de comportamiento:
  * `observe`     -> aprendizaje continuo (sigue la normalidad del agente observado, que
                     evoluciona: cambia el hardware, cambia el LLM, cambia la carga).
  * `incorporate` -> aprendizaje correctivo (el desenlace del análisis ajusta el umbral).

Además es el responsable de PERSISTIR. Es el lugar correcto por una razón de diseño, no de
comodidad: el que modifica el conocimiento es el que sabe cuándo cambió, así que es el que debe
decidir cuándo vale la pena bajarlo a disco. Ninguna otra pieza tiene por qué enterarse de que
el conocimiento se guarda.

Asignación de crédito: UN episodio, UNA corrección por dimensión
-----------------------------------------------------------------
La verdad de este dominio es por ANÁLISIS, pero las predicciones son por OBSERVACIÓN, y un solo
análisis del agente vigilado produce decenas de observaciones. Sin cuidado, esa asimetría
rompe el aprendizaje: un análisis improductivo de 80 observaciones sobre el que el meta-agente
ya alertó (y cuya repetición el dedupe suprime, correctamente) devolvería 1 acierto y 79 falsos
negativos. El agente quedaría castigado 79 veces por haber detectado bien una sola vez, un
único episodio dominaría el ajuste y el umbral se hundiría hasta el piso sin volver a subir.

Por eso cada par (análisis, dimensión) mueve el umbral UNA sola vez. No es un parche de
conveniencia: es la traducción correcta de "el agente se equivocó en este episodio" cuando el
episodio se observó muchas veces. El error sigue contándose entero en las métricas del critic;
lo que se acota es cuántas veces ese mismo error mueve el conocimiento.
"""

from __future__ import annotations

from collections import OrderedDict

from core import Correction, LearningElement, Observation, Outcome

from .performance_element import VALUE_KEY

# Tope de episodios recordados para la asignación de crédito. Es una memoria de corto plazo:
# solo hace falta mientras llegan las verdades de un análisis, que es cuestión de segundos.
_MAX_CREDITED = 20_000


class BehaviorLearningElement(LearningElement):
    """Aprende sobre el comportamiento del agente observado y persiste periódicamente.

    Args:
        model: modelo de normalidad COMPARTIDO con el performance element.
        ledger: libro de análisis, para saber a qué episodio pertenece cada corrección.
        save_every: cada cuántas escrituras se baja un snapshot a disco. No se guarda en cada
            observación porque el volumen de eventos es alto y el disco de una VPS no es
            gratis; no se guarda solo al terminar porque un daemon no termina.
    """

    def __init__(self, model, ledger, save_every: int = 200) -> None:
        self._model = model
        self._ledger = ledger
        self._save_every = max(1, save_every)
        self._writes = 0
        # Pares (análisis, dimensión) que ya movieron el umbral. `OrderedDict` como conjunto
        # con orden de inserción, para poder descartar los más viejos sin recorrer todo.
        self._credited: "OrderedDict[tuple, bool]" = OrderedDict()

    def observe(self, obs: Observation) -> None:
        # Experiencia directa: el baseline se refina con cada observación limpia (el modelo
        # decide internamente si lo es, para no contaminarse con un episodio anómalo).
        self._model.learn_observation(obs.entity_id, obs.values[VALUE_KEY])
        self._tick()

    def incorporate(self, correction: Correction) -> None:
        # Retroalimentación del critic: SOLO los errores mueven el umbral. Los aciertos
        # confirman el modelo actual y no requieren ajuste, así el agente no se desestabiliza
        # cuando ya está acertando.
        entity_id = correction.prediction.entity_id
        if correction.outcome in (Outcome.TRUE_POSITIVE, Outcome.TRUE_NEGATIVE):
            return

        # Un error ya contabilizado para este episodio y esta dimensión no vuelve a mover el
        # umbral (ver la nota sobre asignación de crédito al inicio del módulo).
        if not self._claim_credit(correction):
            return

        if correction.outcome is Outcome.FALSE_POSITIVE:
            # Alertó y el análisis terminó siendo productivo: el desvío de comportamiento era
            # trabajo legítimo. El umbral estaba demasiado sensible -> agrandarlo.
            self._model.adjust_threshold(entity_id, too_sensitive=True)
            self._tick()
        elif correction.outcome is Outcome.FALSE_NEGATIVE:
            # No alertó y el análisis terminó sin producir nada. Ahora bien: si OTRA dimensión
            # sí alertó sobre ese mismo análisis, el episodio no se perdió — el agente cumplió
            # su objetivo. Exigirle además que TODAS las dimensiones se desviaran sería pedirle
            # que detecte una señal que en esta dimensión no existe, y el castigo empujaría su
            # umbral a la baja hasta volverla ruidosa. Solo se corrige el miss REAL: el
            # episodio improductivo que pasó entero desapercibido.
            analysis_id = self._ledger.analysis_for(correction.prediction.timestamp)
            if analysis_id is not None and self._ledger.was_flagged(analysis_id):
                return
            self._model.adjust_threshold(entity_id, too_sensitive=False)
            self._tick()

    def flush(self) -> None:
        """Fuerza un snapshot (apagado ordenado, o inactividad prolongada)."""
        self._writes = 0
        self._model.save()

    def _claim_credit(self, correction: Correction) -> bool:
        """Indica si esta corrección debe mover el umbral, o si su episodio ya se contabilizó.

        Returns:
            `True` la primera vez que un par (análisis, dimensión) reporta un error; `False`
            en las repeticiones del mismo error dentro del mismo episodio.
        """
        analysis_id = self._ledger.analysis_for(correction.prediction.timestamp)
        if analysis_id is None:
            # Sin episodio identificable no se puede agrupar: se ajusta, que es el
            # comportamiento conservador (no perder una señal de error legítima).
            return True
        key = (analysis_id, correction.prediction.entity_id)
        if key in self._credited:
            return False
        self._credited[key] = True
        while len(self._credited) > _MAX_CREDITED:
            self._credited.popitem(last=False)
        return True

    def _tick(self) -> None:
        self._writes += 1
        if self._writes >= self._save_every:
            self._writes = 0
            self._model.save()
