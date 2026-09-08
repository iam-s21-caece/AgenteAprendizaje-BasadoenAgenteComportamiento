"""PerformanceElement del dominio: decide si el comportamiento observado se desvía.

Percibe una observación de comportamiento, consulta el modelo compartido y decide. NO aprende
y NO llama al LLM — ambas cosas son deliberadas:

  * No aprende, porque la frontera de R&N asigna esa tarea al learning element. El modelo llega
    inyectado y es el MISMO objeto que aquél modifica; "decidir mejor" es consecuencia
    automática de "haber aprendido".

  * No llama al LLM, porque la decisión debe seguir siendo un umbral legible. Si el juicio lo
    emitiera un modelo de lenguaje, el aprendizaje dejaría de ser auditable (no se podría
    atribuir un cambio de decisión al ajuste del umbral) y la curva de aprendizaje dejaría de
    ser medible. El LLM aporta en la exploración, no en el veredicto.
"""

from __future__ import annotations

from core import Decision, Observation, PerformanceElement

# Clave de la dimensión numérica en `Observation.values`. El contrato del núcleo admite varias
# dimensiones por observación; este dominio usa una sola y codifica la dimensión en el
# `entity_id` (`<agente>:<dimensión>`), para que el baseline y el umbral sean por dimensión.
VALUE_KEY = "value"


class BehaviorPerformanceElement(PerformanceElement):
    def __init__(self, model, ledger) -> None:
        # Modelo y libro llegan por inyección: son estado COMPARTIDO con el resto del dominio,
        # no propiedad de este componente.
        self._model = model
        self._ledger = ledger

    def evaluate(self, obs: Observation, cycle: int) -> Decision:
        value = obs.values[VALUE_KEY]
        dev = self._model.deviation_sigmas(obs.entity_id, value)

        # El umbral es ADAPTATIVO y por entidad: la verdad diferida lo movió en ciclos
        # anteriores. Por eso el mismo desvío puede alertar hoy y callarse mañana — ahí es
        # donde el aprendizaje se vuelve visible en la decisión.
        threshold = self._model.threshold(obs.entity_id)
        alerted = self._model.is_warm(obs.entity_id) and abs(dev) >= threshold

        # Firma = análisis + entidad. Dos alertas son "la misma" si son sobre la misma
        # dimensión DENTRO del mismo análisis del agente observado: mientras un análisis se
        # descarrila, la latencia se dispara en varios pasos seguidos y no aporta nada
        # repetir la meta-alerta. Un análisis nuevo sí vuelve a poder alertar.
        analysis_id = self._ledger.analysis_for(obs.timestamp) or "unknown"
        signature = f"{analysis_id}|{obs.entity_id}"

        # Se deja constancia de que este episodio quedó marcado. Es lo que después permite al
        # aprendizaje distinguir "no detecté el episodio" de "lo detecté por otra dimensión".
        if alerted:
            self._ledger.note_flagged(analysis_id)

        return Decision(deviation_sigmas=dev, alerted=alerted, signature=signature)
