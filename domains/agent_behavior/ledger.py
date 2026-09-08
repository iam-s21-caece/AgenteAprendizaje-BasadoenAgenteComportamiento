"""Libro de análisis: el estado de los análisis del agente observado.

Por qué existe (y por qué es un objeto compartido)
--------------------------------------------------
La verdad diferida de este dominio es DERIVADA: no la dicta un humano, la dicta el desenlace
del propio análisis observado. Pero el desenlace se conoce en un evento (`analysis_end`) que
llega MUCHO después de las observaciones que ese análisis fue generando. Alguien tiene que
recordar, entre medio, qué análisis está abierto y qué observaciones le pertenecen.

Este libro es ese alguien. Igual que el modelo de normalidad, es un ÚNICO objeto compartido:
  * el colector lo ESCRIBE (abre análisis, anota observaciones, cierra con veredicto),
  * el performance element y la fuente de verdad lo LEEN.

Vive en el dominio (no en el núcleo) porque "qué es un análisis" y "cuándo fue productivo" son
conocimiento del dominio observado, no del loop de R&N.

Sobre el veredicto (la regla de verdad, en una línea)
------------------------------------------------------
Un análisis fue PRODUCTIVO si terminó emitiendo una alerta con contenido (CVEs detectados o
severidad por encima de `info`). Si terminó sin nada, con error, o quedó abandonado, el
esfuerzo que el meta-agente observó fue esfuerzo desperdiciado.

    was_anomaly (para el critic) = NOT productivo

Es decir: lo que el meta-agente aprende a distinguir es **un agente trabajando duro con razón**
de **un agente girando en falso**. Esa lectura es independiente de Keycloak y por eso sirve
igual para los agentes que se sumen después.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional

# Cuántas asociaciones (timestamp de observación -> análisis) se retienen. Es una caché de
# correlación, no un registro histórico: una vez que la verdad de un análisis se entregó, sus
# entradas ya no hacen falta. El tope evita que un daemon de semanas crezca sin límite.
_MAX_OBSERVATION_LINKS = 100_000

# Cuántos análisis se recuerdan como "ya marcados". Sobrevive al veredicto porque el
# aprendizaje lo consulta después de que la verdad llegó (ver `was_flagged`).
_MAX_FLAGGED = 50_000


@dataclass
class AnalysisVerdict:
    """Desenlace de un análisis ya cerrado. Es la materia prima de la verdad diferida."""

    productive: bool          # ¿el análisis produjo una alerta con contenido?
    iterations: int           # cuántas iteraciones ReAct consumió
    emitted_cves: int         # cuántas CVEs identificó
    severity: str             # severidad de la alerta emitida (o "" si no hubo)
    errored: bool             # ¿hubo al menos un evento de error?
    abandoned: bool           # ¿se cerró por timeout en vez de por `analysis_end`?
    closed_at: float          # epoch en que se registró el cierre


@dataclass
class _OpenAnalysis:
    """Un análisis en curso: lo que sabemos de él ANTES de conocer su desenlace."""

    analysis_id: str
    opened_at: float
    iterations: int = 0
    emitted_cves: int = 0
    severity: str = ""
    errored: bool = False
    saw_alert: bool = False


class AnalysisLedger:
    """Registro de análisis abiertos, sus observaciones y el veredicto de los cerrados.

    Args:
        abandon_after_seconds: cuánto puede estar abierto un análisis antes de darlo por
            abandonado. Un análisis que nunca cierra significa que el agente observado se
            colgó o murió a mitad del razonamiento — eso ES una anomalía de comportamiento,
            así que se cierra con veredicto no-productivo en vez de quedar colgado para
            siempre reteniendo predicciones sin juzgar.
    """

    def __init__(self, abandon_after_seconds: float = 1800.0) -> None:
        self._abandon_after = abandon_after_seconds
        self._open: Dict[str, _OpenAnalysis] = {}
        self._verdicts: Dict[str, AnalysisVerdict] = {}
        # timestamp de observación -> analysis_id. Permite que el performance element y la
        # fuente de verdad recuperen a qué análisis pertenece una predicción, sin que el
        # núcleo tenga que transportar ese dato (`Observation` es deliberadamente mínima).
        self._links: "OrderedDict[int, str]" = OrderedDict()
        # Análisis sobre los que el meta-agente alertó por ALGUNA dimensión. Se registra
        # mientras el análisis transcurre (antes de conocer su desenlace) y sobrevive al
        # veredicto, porque el aprendizaje lo consulta después. Ver `was_flagged`.
        self._flagged: "OrderedDict[str, bool]" = OrderedDict()

    # --- Escritura: la usa el colector ----------------------------------------------------

    def open(self, analysis_id: str) -> None:
        """Registra el inicio de un análisis del agente observado."""
        if analysis_id not in self._open:
            self._open[analysis_id] = _OpenAnalysis(
                analysis_id=analysis_id, opened_at=time.time()
            )
        self._sweep_abandoned()

    def ensure_open(self, analysis_id: str) -> None:
        """Abre el análisis si no se vio su `analysis_start`.

        Hace falta porque el colector puede empezar a leer una traza por la mitad (arranque en
        vivo sobre un análisis ya iniciado, o rotación de archivos). Perder el evento de
        apertura no debe impedir juzgar el resto del análisis.
        """
        self.open(analysis_id)

    def note_iteration(self, analysis_id: str, iteration: int) -> None:
        """Actualiza el contador de iteraciones observado hasta ahora."""
        state = self._open.get(analysis_id)
        if state is not None:
            state.iterations = max(state.iterations, iteration)

    def note_error(self, analysis_id: str) -> None:
        """Marca que el análisis registró un error."""
        state = self._open.get(analysis_id)
        if state is not None:
            state.errored = True

    def note_alert(self, analysis_id: str, cves: int, severity: str) -> None:
        """Registra la alerta que el análisis emitió (su producto)."""
        state = self._open.get(analysis_id)
        if state is not None:
            state.saw_alert = True
            state.emitted_cves = cves
            state.severity = severity

    def note_observation(self, timestamp: int, analysis_id: str) -> None:
        """Asocia una observación (por su timestamp) con el análisis que la originó."""
        self._links[timestamp] = analysis_id
        while len(self._links) > _MAX_OBSERVATION_LINKS:
            self._links.popitem(last=False)

    def note_flagged(self, analysis_id: str) -> None:
        """Registra que el meta-agente alertó sobre este análisis (por cualquier dimensión).

        Lo escribe el performance element en el momento de decidir, es decir ANTES de que
        exista el desenlace. Esa precedencia importa: es información sobre lo que el agente
        hizo, no sobre lo que resultó ser cierto.
        """
        self._flagged[analysis_id] = True
        while len(self._flagged) > _MAX_FLAGGED:
            self._flagged.popitem(last=False)

    def close(self, analysis_id: str, iterations: Optional[int] = None) -> AnalysisVerdict:
        """Cierra un análisis y fija su veredicto. Acá nace la verdad diferida.

        Args:
            analysis_id: análisis a cerrar.
            iterations: iteraciones reportadas por el evento de cierre (si vino).

        Returns:
            El veredicto registrado.
        """
        state = self._open.pop(analysis_id, None)
        if state is None:
            state = _OpenAnalysis(analysis_id=analysis_id, opened_at=time.time())
        if iterations is not None:
            state.iterations = max(state.iterations, iterations)

        # La regla de productividad, explícita y en un solo lugar: hubo alerta Y esa alerta
        # tenía contenido. Un análisis que corrió, gastó iteraciones y no concluyó nada es
        # exactamente el caso que el meta-agente debe aprender a anticipar.
        productive = state.saw_alert and (
            state.emitted_cves > 0 or (state.severity and state.severity != "info")
        )

        verdict = AnalysisVerdict(
            productive=bool(productive),
            iterations=state.iterations,
            emitted_cves=state.emitted_cves,
            severity=state.severity,
            errored=state.errored,
            abandoned=False,
            closed_at=time.time(),
        )
        self._verdicts[analysis_id] = verdict
        return verdict

    # --- Lectura: la usan el performance element y la fuente de verdad --------------------

    def analysis_for(self, timestamp: int) -> Optional[str]:
        """Devuelve el análisis al que pertenece una observación, si se conoce."""
        return self._links.get(timestamp)

    def verdict(self, analysis_id: str) -> Optional[AnalysisVerdict]:
        """Veredicto de un análisis cerrado, o `None` si sigue abierto.

        `None` es la respuesta correcta e importante: significa "todavía no se sabe". Es lo que
        mantiene DIFERIDA a la verdad — mientras el análisis está en curso, el meta-agente ya
        predijo pero nadie puede corregirlo aún.
        """
        return self._verdicts.get(analysis_id)

    def was_flagged(self, analysis_id: str) -> bool:
        """¿El meta-agente alertó sobre este análisis por alguna dimensión?

        Lo consulta el aprendizaje para no penalizar como "miss" a las dimensiones que
        guardaron silencio en un episodio que YA fue detectado por otra.
        """
        return analysis_id in self._flagged

    def is_open(self, analysis_id: str) -> bool:
        return analysis_id in self._open

    def forget(self, analysis_id: str) -> None:
        """Descarta el veredicto ya consumido (higiene de memoria del daemon)."""
        self._verdicts.pop(analysis_id, None)

    def open_count(self) -> int:
        return len(self._open)

    # --- Internos --------------------------------------------------------------------------

    def _sweep_abandoned(self) -> None:
        """Cierra como abandonados los análisis que llevan demasiado tiempo abiertos."""
        if not self._open:
            return
        now = time.time()
        stale = [
            aid
            for aid, st in self._open.items()
            if (now - st.opened_at) > self._abandon_after
        ]
        for aid in stale:
            state = self._open.pop(aid)
            self._verdicts[aid] = AnalysisVerdict(
                productive=False,          # un análisis que nunca cerró no produjo nada
                iterations=state.iterations,
                emitted_cves=state.emitted_cves,
                severity=state.severity,
                errored=state.errored,
                abandoned=True,
                closed_at=now,
            )
