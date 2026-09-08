"""ProblemGenerator del dominio: dónde mirar fuera de la vigilancia habitual.

Por qué el LLM va ACÁ y no en otro componente
----------------------------------------------
De los cuatro componentes de R&N, el problem generator es el único cuyo trabajo es *proponer
hipótesis*: sugerir experiencia nueva que la explotación por sí sola no produciría. Es
exactamente la tarea en la que un modelo de lenguaje aporta —encontrar un patrón entre señales
que no están correlacionadas por construcción— y es el único lugar donde equivocarse es barato:
una sugerencia mala solo hace mirar en la dirección incorrecta, no emite una alerta ni mueve un
umbral.

Los otros tres lugares fueron descartados por razones concretas:
  * en el performance element convertiría la DECISIÓN en caja negra y haría inmedible la curva
    de aprendizaje (no se podría atribuir un cambio de decisión al ajuste del umbral);
  * en el critic destruiría la externalidad de la verdad: el agente se estaría calificando solo;
  * en el learning element el conocimiento dejaría de ser un umbral legible.

Degradación: si no hay API key, si la API falla o si tarda de más, se cae a la heurística
determinista (vigilar la entidad menos observada). El agente NUNCA depende del LLM para seguir
aprendiendo — con `--no-llm` el experimento corre reproducible y sin costo de tokens.
"""

from __future__ import annotations

import json
from typing import List, Optional, Sequence

from core import AuditTrail, ExplorationSuggestion, ProblemGenerator

_SYSTEM_PROMPT = (
    "Sos el componente de exploración de un agente de aprendizaje (taxonomía Russell & "
    "Norvig) que supervisa el COMPORTAMIENTO de otros agentes de software autónomos. "
    "No analizás seguridad ni vulnerabilidades: analizás cómo se comportan los agentes "
    "mientras razonan (latencia entre pasos, longitud de su razonamiento, tamaño de los "
    "resultados de sus herramientas, eventos por iteración).\n\n"
    "Tu única tarea es proponer DÓNDE MIRAR a continuación: qué dimensión de comportamiento "
    "conviene vigilar y por qué, a partir de la evidencia reciente. No decidís alertas y no "
    "juzgás si el agente acertó — eso lo hacen otros componentes con umbrales y verdad "
    "externa.\n\n"
    "Respondé SOLO con un array JSON de 1 a 3 objetos, sin texto alrededor y sin markdown:\n"
    '[{"entity": "<entity_id exacto de la lista, o null>", "reason": "<una frase concreta>"}]'
)


class BehaviorProblemGenerator(ProblemGenerator):
    """Sugiere dimensiones a vigilar; con LLM si está disponible, heurística si no.

    Args:
        model: modelo de normalidad compartido (fuente de la evidencia y de la heurística).
        audit: traza del loop, para darle al LLM los ciclos recientes como contexto.
        llm: cliente opcional. `None` o sin credencial = modo puramente determinista.
        every: cada cuántos ciclos explorar. Explorar en cada ciclo sería ruido (y, con LLM,
            gasto continuo de tokens); no explorar nunca sería estancarse.
        context_cycles: cuántos ciclos recientes se le resumen al LLM.
    """

    def __init__(
        self,
        model,
        audit: AuditTrail,
        llm=None,
        every: int = 200,
        context_cycles: int = 40,
    ) -> None:
        self._model = model
        self._audit = audit
        self._llm = llm
        self._every = max(1, every)
        self._context_cycles = context_cycles

    def suggest(self, cycle: int) -> Sequence[ExplorationSuggestion]:
        if cycle == 0 or cycle % self._every != 0:
            return ()
        entities = self._model.entities()
        if not entities:
            return ()

        if self._llm is not None and getattr(self._llm, "enabled", False):
            suggestions = self._suggest_with_llm(entities)
            if suggestions:
                return suggestions
        return self._suggest_heuristic(entities)

    # --- Heurística determinista (el piso, siempre disponible) ----------------------------

    def _suggest_heuristic(self, entities) -> Sequence[ExplorationSuggestion]:
        """Vigilar la dimensión menos observada: es donde el modelo está más ciego."""
        least = min(entities, key=self._model.seen_count)
        return (
            ExplorationSuggestion(
                entity_id=least,
                dimension=None,
                reason=(
                    f"vigilar '{least}': es la dimensión de comportamiento menos observada "
                    f"({self._model.seen_count(least)} obs) y su normalidad está peor aprendida"
                ),
            ),
        )

    # --- Exploración asistida por LLM ------------------------------------------------------

    def _suggest_with_llm(self, entities) -> Optional[Sequence[ExplorationSuggestion]]:
        """Pide hipótesis al LLM sobre la evidencia reciente. `None` si no se pudo."""
        try:
            prompt = self._build_prompt(entities)
        except Exception:  # noqa: BLE001 - construir contexto jamás debe romper el loop
            return None

        raw = self._llm.complete(_SYSTEM_PROMPT, prompt)
        if not raw:
            return None

        parsed = self._parse(raw, set(entities))
        return parsed or None

    def _build_prompt(self, entities) -> str:
        """Arma el contexto: estado del conocimiento + ciclos recientes del loop."""
        knowledge = [self._model.stats(entity) for entity in entities]

        recent = []
        for record in self._audit.records[-self._context_cycles :]:
            row = {
                "cycle": record.cycle,
                "entity": record.observation.entity_id,
                "value": round(next(iter(record.observation.values.values())), 3),
                "deviation_sigmas": round(record.prediction.deviation_sigmas, 2),
                "alerted": record.prediction.alerted,
            }
            if record.corrections:
                row["truth_arrived"] = [c.outcome.name for c in record.corrections]
            recent.append(row)

        return (
            "Estado actual del conocimiento del agente (una entrada por dimensión de "
            "comportamiento observada; 'threshold_sigmas' es el umbral que la verdad diferida "
            "fue ajustando, y false_positives/false_negatives son los errores acumulados):\n"
            f"{json.dumps(knowledge, ensure_ascii=False, indent=2)}\n\n"
            "Ciclos recientes del loop (cada uno es una observación del comportamiento del "
            "agente vigilado; 'truth_arrived' aparece cuando maduró la verdad de una "
            "predicción anterior):\n"
            f"{json.dumps(recent, ensure_ascii=False, indent=2)}\n\n"
            "Entidades válidas para sugerir (usá el identificador exacto):\n"
            f"{json.dumps(list(entities), ensure_ascii=False)}\n\n"
            "¿Qué dimensión conviene vigilar ahora y por qué? Buscá patrones entre "
            "dimensiones, no repitas lo obvio."
        )

    def _parse(self, raw: str, valid: set) -> List[ExplorationSuggestion]:
        """Convierte la respuesta del LLM en sugerencias, descartando lo que no sirva.

        Es deliberadamente desconfiado: se recorta el markdown, se acepta solo JSON, y toda
        entidad que el modelo haya inventado se descarta. Una sugerencia sobre una entidad
        inexistente sería ruido no auditable en la traza.
        """
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            data = json.loads(text)
        except ValueError:
            return []
        if not isinstance(data, list):
            return []

        suggestions: List[ExplorationSuggestion] = []
        for item in data[:3]:
            if not isinstance(item, dict):
                continue
            reason = item.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                continue
            entity = item.get("entity")
            if entity is not None and entity not in valid:
                continue  # el modelo inventó una entidad: se descarta
            suggestions.append(
                ExplorationSuggestion(
                    entity_id=entity,
                    dimension=None,
                    # Se marca el origen: en la traza tiene que poder distinguirse una
                    # hipótesis del LLM de una regla determinista del agente.
                    reason=f"[llm] {reason.strip()}",
                )
            )
        return suggestions
