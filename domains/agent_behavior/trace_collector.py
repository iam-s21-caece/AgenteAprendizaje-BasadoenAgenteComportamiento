"""Colector: convierte las trazas del agente observado en `Observation` agnósticas.

ESTE ES EL ÚNICO ARCHIVO DEL DOMINIO QUE CONOCE UN FORMATO.
Lee los JSONL del ciclo agéntico que el agente observado escribe en su volumen, y emite
observaciones. Ninguna otra pieza (ni el núcleo ni el resto del dominio) sabe qué es una traza.

Cómo lee: tail incremental, no lectura de archivos completos
------------------------------------------------------------
Esto NO es un detalle de implementación, es lo que hace legítima la verdad diferida. Si el
colector leyera un análisis ya terminado de una sola vez, conocería el desenlace en el mismo
instante que la observación, y "diferir" la verdad sería un retardo cosmético: el agente no
estaría aprendiendo, estaría haciendo un lookup con delay.

Leyendo evento por evento en el orden en que fueron escritos, cuando el meta-agente juzga el
comportamiento de un análisis EN CURSO, el desenlace todavía no existe en el stream. La verdad
llega después porque estructuralmente no puede llegar antes.

(Al arrancar sobre trazas históricas — `from_start=True` — se replica el mismo orden: los
eventos de un análisis se emiten antes que su `analysis_end`. El diferimiento en ciclos se
conserva; lo que se gana es arrancar con el baseline ya caliente en vez de esperar días.)

Qué dimensiones observa (y por qué ESAS)
-----------------------------------------
Todas tienen que ser ESTACIONARIAS para que "desvío en sigmas" signifique algo. Una métrica que
crece por construcción dentro del análisis (el número de iteración, el total acumulado de
tools) se desviaría trivialmente al final de cada análisis y el agente aprendería a alertar
sobre el paso del tiempo. Por eso se miden magnitudes por-paso, no acumuladas:

  * `step_latency_s`       segundos entre dos eventos consecutivos del mismo análisis.
                           Un agente que se traba o que pelea con el LLM lo muestra acá.
  * `thought_len`          caracteres del chain-of-thought en cada `llm_thought`.
                           Razonamientos anormalmente largos o cortos = el agente perdido.
  * `tool_result_size`     tamaño del resultado de cada tool.
                           Detecta tools devolviendo vacío o volcados desmedidos.
  * `events_per_iteration` cuántos eventos consumió la iteración que acaba de cerrar.
                           Es la firma clásica del agente ReAct girando en falso.

Las magnitudes por análisis (total de iteraciones, duración total) NO se usan como
observaciones: llegan con `analysis_end`, que es el mismo momento en que llega la verdad, así
que tendrían diferimiento cero. Se conservan como contexto de la meta-alerta, no como señal.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from core import Collector, Observation

from .ledger import AnalysisLedger
from .performance_element import VALUE_KEY

# Dimensiones de comportamiento observadas. El `entity_id` final es "<agente>:<dimensión>",
# de modo que sumar agentes al esquema solo agrega entidades al mismo stream.
DIM_STEP_LATENCY = "step_latency_s"
DIM_THOUGHT_LEN = "thought_len"
DIM_TOOL_RESULT_SIZE = "tool_result_size"
DIM_EVENTS_PER_ITERATION = "events_per_iteration"

DIMENSIONS = (
    DIM_STEP_LATENCY,
    DIM_THOUGHT_LEN,
    DIM_TOOL_RESULT_SIZE,
    DIM_EVENTS_PER_ITERATION,
)


def _parse_timestamp_us(raw: str) -> Optional[int]:
    """Convierte el timestamp ISO de la traza a microsegundos epoch.

    Se usa microsegundos (no segundos) porque el timestamp es además la CLAVE con la que el
    loop empareja una predicción con su verdad: dos eventos de la misma dimensión a la misma
    marca de tiempo se pisarían. La traza los emite con precisión de microsegundos.
    """
    if not raw:
        return None
    try:
        text = raw.replace("Z", "+00:00")
        return int(datetime.fromisoformat(text).timestamp() * 1_000_000)
    except (ValueError, TypeError):
        return None


class _AnalysisCursor:
    """Estado incremental de un análisis en curso: lo mínimo para calcular las dimensiones.

    Solo mira hacia atrás (el evento anterior, la iteración anterior). Nunca hace lookahead:
    si necesitara leer un evento futuro para emitir una observación, podría estar mirando el
    desenlace antes de predecir y rompería el diferimiento.
    """

    def __init__(self, analysis_id: str, first_event_us: int) -> None:
        self.analysis_id = analysis_id
        self.last_event_us = first_event_us
        self.current_iteration: int = -1
        self.events_in_iteration: int = 0


class TraceTailCollector(Collector):
    """Sigue los JSONL de traza del agente observado y emite `Observation` en vivo.

    Args:
        traces_dir: raíz de trazas del agente observado (estructura `<YYYY-MM-DD>/<id>.jsonl`).
        ledger: libro compartido donde se anota la apertura, el avance y el cierre de análisis.
        agent_id: nombre del agente observado; prefija cada `entity_id`.
        poll_interval_seconds: cuánto esperar cuando no hay eventos nuevos.
        from_start: si `True`, procesa también lo ya escrito (arranca con baseline caliente);
            si `False`, se posiciona al final y solo observa actividad nueva. Solo aplica al
            PRIMER arranque: si hay offsets persistidos, se retoma donde se dejó.
        offsets_path: dónde persistir la posición de lectura. Sin esto, cada reinicio del
            contenedor volvería a procesar todo el histórico: el agente re-aprendería de los
            mismos datos (inflando los conteos de su baseline) y re-emitiría meta-alertas de
            análisis viejos como si fueran nuevos. El progreso de lectura es estado operativo
            y tiene que sobrevivir al deploy igual que el conocimiento.
        idle_callback: se invoca en cada vuelta sin datos. Lo usa el entry point para persistir
            el modelo aprendido mientras el agente observado está inactivo — sin esto, un
            colector bloqueado en poll dejaría al proceso sin ningún punto de control.
    """

    def __init__(
        self,
        traces_dir: Path | str,
        ledger: AnalysisLedger,
        agent_id: str = "keycloak-agent",
        poll_interval_seconds: float = 2.0,
        from_start: bool = True,
        offsets_path: Optional[Path | str] = None,
        idle_callback=None,
    ) -> None:
        self._dir = Path(traces_dir)
        self._ledger = ledger
        self._agent = agent_id
        self._poll = poll_interval_seconds
        self._from_start = from_start
        self._offsets_path = Path(offsets_path) if offsets_path else None
        self._idle_callback = idle_callback
        # Offset en bytes ya consumido de cada archivo. Es lo que hace incremental al tail:
        # se relee solo lo nuevo, y una línea a medio escribir no se consume hasta completarse.
        self._offsets: Dict[Path, int] = {}
        self._cursors: Dict[str, _AnalysisCursor] = {}
        self._bootstrapped = False
        self._dirty = False
        # Contadores de percepción. Existen porque "no hay actividad" es un diagnóstico
        # ambiguo y caro: no distingue "el volumen está mal montado" de "el analista no
        # generó trazas" de "las generó pero no las puedo parsear". Sin estos números, esa
        # diferencia solo se resuelve entrando al contenedor a mano.
        self._stats = {
            "dir_existe": False,
            "archivos_vistos": 0,
            "bytes_leidos": 0,
            "eventos_leidos": 0,
            "observaciones": 0,
            "lineas_ilegibles": 0,
            "eventos_sin_campos": 0,
            "eventos_sin_timestamp": 0,
        }

    # --- Autodiagnóstico de la percepción --------------------------------------------------

    def diagnostics(self) -> dict:
        """Estado de la percepción, con un veredicto legible de por qué no hay observaciones.

        Por qué el agente se diagnostica a sí mismo
        -------------------------------------------
        "Sin actividad del agente observado" es verdadero en tres situaciones muy distintas
        —volumen mal montado, analista que no generó trazas, o trazas ilegibles— y las tres
        se ven idénticas desde afuera. Que el agente diga CUÁL de las tres está ocurriendo
        convierte un problema de una tarde en una línea de log. La cadena de percepción es
        estrictamente secuencial (directorio -> archivos -> bytes -> eventos -> observaciones),
        así que el primer eslabón en cero es la causa, y los siguientes son consecuencia.
        """
        stats = dict(self._stats)
        stats["ruta_observada"] = str(self._dir)
        stats["archivos_con_offset"] = len(self._offsets)

        if not stats["dir_existe"]:
            verdict = (
                f"El directorio {self._dir} NO existe dentro del contenedor. Es un problema de "
                "montaje: revisar OBSERVED_DATA_DIR (Docker crea un directorio vacío en vez de "
                "fallar cuando la ruta del host no existe)."
            )
        elif stats["archivos_vistos"] == 0:
            verdict = (
                f"El directorio {self._dir} existe pero no contiene ningún <fecha>/<id>.jsonl. "
                "El agente observado todavía no escribió trazas: solo las escribe cuando un "
                "check determinista dispara Y corre el ReAct (con --no-llm produce alertas "
                "pero NO trazas)."
            )
        elif stats["eventos_leidos"] == 0:
            verdict = (
                "Hay archivos de traza pero no se leyó ningún evento nuevo: ya estaban "
                "consumidos según tail_offsets.json. Es lo esperable si el analista está "
                "quieto; si se esperaba reprocesar, hay que borrar ese archivo."
            )
        elif stats["observaciones"] == 0:
            verdict = (
                "Se leyeron eventos pero ninguno produjo una observación. El formato de la "
                "traza no es el esperado (ver eventos_sin_campos / eventos_sin_timestamp)."
            )
        else:
            verdict = "Percepción normal: hay observaciones."
        stats["veredicto"] = verdict
        return stats

    def probe(self, sample_events: int = 3) -> dict:
        """Inspección puntual del origen de trazas, SIN consumir la posición de lectura.

        Es la versión bajo demanda de `diagnostics()`: en vez de reportar lo que se percibió
        hasta ahora, va a mirar el directorio y valida unos pocos eventos reales contra los
        mismos criterios que usa el tail. No toca `self._offsets` a propósito — diagnosticar
        no puede consumir datos que después el loop dejaría de procesar.
        """
        report: dict = {
            "ruta_observada": str(self._dir),
            "existe": self._dir.is_dir(),
            "archivos": 0,
            "archivos_recientes": [],
            "eventos_validados": 0,
            "eventos_rechazados": 0,
            "detalle": [],
        }
        if not report["existe"]:
            report["veredicto"] = (
                f"{self._dir} no existe dentro del contenedor -> problema de MONTAJE. "
                "Revisar OBSERVED_DATA_DIR: si la ruta del host no existe, Docker crea un "
                "directorio vacío en vez de fallar."
            )
            return report

        files = sorted(self._dir.glob("*/*.jsonl"))
        report["archivos"] = len(files)
        report["archivos_recientes"] = [
            str(f.relative_to(self._dir).as_posix()) for f in files[-5:]
        ]
        if not files:
            report["veredicto"] = (
                f"{self._dir} existe pero está vacío de trazas. El analista no escribió "
                "ninguna: solo las genera cuando dispara un check Y corre el ReAct. Con "
                "--no-llm emite alertas pero NO trazas."
            )
            return report

        # Validar los primeros eventos del archivo más reciente con los mismos criterios
        # que aplica el tail, para distinguir "no hay datos" de "no puedo leerlos".
        newest = files[-1]
        try:
            lines = [l for l in newest.read_text(encoding="utf-8").splitlines() if l.strip()]
        except OSError as exc:
            report["veredicto"] = f"No se pudo leer {newest}: {exc}"
            return report

        for line in lines[:sample_events]:
            try:
                event = json.loads(line)
            except ValueError:
                report["eventos_rechazados"] += 1
                report["detalle"].append("línea JSON ilegible")
                continue
            faltantes = [k for k in ("analysis_id", "event_type", "timestamp") if not event.get(k)]
            if faltantes:
                report["eventos_rechazados"] += 1
                report["detalle"].append(f"faltan campos: {', '.join(faltantes)}")
                continue
            if _parse_timestamp_us(event["timestamp"]) is None:
                report["eventos_rechazados"] += 1
                report["detalle"].append(f"timestamp no parseable: {event['timestamp']!r}")
                continue
            report["eventos_validados"] += 1
            report["detalle"].append(
                f"OK {event['event_type']} (analysis_id={event['analysis_id']})"
            )

        if report["eventos_validados"]:
            report["veredicto"] = (
                f"Trazas legibles ({report['archivos']} archivos). Si aun así no hay "
                "observaciones, revisar tail_offsets.json: puede estar todo consumido."
            )
        else:
            report["veredicto"] = (
                "Hay archivos de traza pero sus eventos no pasan la validación: el formato "
                "no es el que este colector espera (ver 'detalle')."
            )
        return report

    # --- Persistencia de la posición de lectura -------------------------------------------

    def load_offsets(self) -> bool:
        """Restaura la posición de lectura de una corrida anterior.

        Returns:
            `True` si se restauró algo. Al restaurar se marca el arranque como ya resuelto:
            los archivos conocidos siguen desde su offset, y los que aparecieron mientras el
            proceso estuvo caído se leen desde el principio (son actividad nueva).
        """
        if self._offsets_path is None or not self._offsets_path.exists():
            return False
        try:
            data = json.loads(self._offsets_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        offsets = data.get("offsets") if isinstance(data, dict) else None
        if not isinstance(offsets, dict):
            return False
        # Las rutas se guardan RELATIVAS al directorio de trazas: así, si cambia el punto de
        # montaje del volumen observado, el progreso sigue siendo válido en vez de provocar un
        # reprocesamiento completo del histórico.
        self._offsets = {self._dir / path: int(value) for path, value in offsets.items()}
        self._bootstrapped = True
        return bool(self._offsets)

    def save_offsets(self) -> bool:
        """Baja la posición de lectura a disco (atómico). No lanza nunca."""
        if self._offsets_path is None or not self._dirty:
            return False
        try:
            self._offsets_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._offsets_path.with_suffix(self._offsets_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(
                    {"offsets": {self._relative(p): o for p, o in self._offsets.items()}},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            os.replace(tmp, self._offsets_path)
            self._dirty = False
            return True
        except OSError:
            return False

    def _relative(self, path: Path) -> str:
        """Ruta relativa al directorio de trazas, con separador estable entre plataformas."""
        try:
            return path.relative_to(self._dir).as_posix()
        except ValueError:
            return path.as_posix()

    # --- Contrato del núcleo ---------------------------------------------------------------

    def stream(self) -> Iterator[Observation]:
        """Generador infinito de observaciones.

        El núcleo consume esto con `for cycle, obs in enumerate(collector.stream())`, así que
        un generador que nunca termina es un daemon válido SIN tocar el loop: cuando no hay
        eventos nuevos, duerme y sigue. Ese es el único cambio que necesitaba el agente para
        pasar de experimento por lotes a proceso continuo en la VPS.
        """
        while True:
            produced = False
            for path in self._discover_files():
                for observation in self._read_new(path):
                    produced = True
                    yield observation
            if not produced:
                if self._idle_callback is not None:
                    self._idle_callback()
                time.sleep(self._poll)

    # --- Lectura de archivos ---------------------------------------------------------------

    def _discover_files(self) -> List[Path]:
        """Devuelve las trazas conocidas, en orden cronológico por carpeta de fecha.

        El orden entre archivos distintos es aproximado (dos análisis simultáneos se intercalan
        en tiempo real pero se leen uno después del otro). No importa: lo que el cierre del loop
        exige es el orden DENTRO de un análisis — que sus observaciones precedan a su
        `analysis_end` — y eso lo garantiza el archivo, que es append-only y por análisis.
        """
        if not self._dir.is_dir():
            self._stats["dir_existe"] = False
            self._stats["archivos_vistos"] = 0
            return []
        self._stats["dir_existe"] = True
        try:
            files = sorted(self._dir.glob("*/*.jsonl"))
        except OSError:
            return []
        self._stats["archivos_vistos"] = len(files)

        if not self._bootstrapped:
            self._bootstrapped = True
            if not self._from_start:
                # Modo "solo actividad nueva": marcar todo lo existente como ya consumido.
                for path in files:
                    try:
                        self._offsets[path] = path.stat().st_size
                    except OSError:
                        self._offsets[path] = 0
        return files

    def _read_new(self, path: Path) -> Iterator[Observation]:
        """Lee las líneas completas añadidas desde la última pasada y las traduce."""
        start = self._offsets.get(path, 0)
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size < start:
            # El archivo se truncó o rotó: volver a empezar en vez de leer basura.
            start = 0
        if size == start:
            return

        try:
            with path.open("rb") as handle:
                handle.seek(start)
                chunk = handle.read(size - start)
        except OSError:
            return

        # Consumir solo hasta el último salto de línea: si el agente observado estaba
        # escribiendo cuando leímos, la última línea puede estar incompleta y debe esperar.
        cut = chunk.rfind(b"\n")
        if cut == -1:
            return

        # El offset avanza LÍNEA POR LÍNEA, después de procesar cada una, y no de una vez por
        # bloque. La diferencia importa al apagar: el loop consume este generador de a una
        # observación, así que un SIGTERM lo corta a mitad de archivo. Con el offset adelantado
        # al bloque entero, los eventos leídos pero no entregados quedarían marcados como
        # consumidos y se perderían para siempre en el próximo arranque.
        self._stats["bytes_leidos"] += cut + 1
        consumed = 0
        for raw in chunk[: cut + 1].splitlines(keepends=True):
            if raw.strip():
                try:
                    event = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    event = None  # línea corrupta: se saltea, el tail no se rompe por eso
                    self._stats["lineas_ilegibles"] += 1
                if event is not None:
                    yield from self._observations_from(event)
            consumed += len(raw)
            self._offsets[path] = start + consumed
            self._dirty = True

    # --- Traducción traza -> observaciones -------------------------------------------------

    def _observations_from(self, event: dict) -> Iterator[Observation]:
        """Traduce UN evento de traza en las observaciones que correspondan.

        Un evento puede producir varias observaciones (por ejemplo latencia + longitud del
        razonamiento), siempre de dimensiones distintas — nunca colisionan en el emparejamiento
        del loop, que indexa por (entidad, timestamp).
        """
        self._stats["eventos_leidos"] += 1
        analysis_id = event.get("analysis_id")
        event_type = event.get("event_type")
        if not analysis_id or not event_type:
            # El JSON se leyó pero no tiene la forma de una traza del ciclo agéntico.
            self._stats["eventos_sin_campos"] += 1
            return
        ts_us = _parse_timestamp_us(event.get("timestamp", ""))
        if ts_us is None:
            # Formato de timestamp inesperado. Se cuenta aparte porque implica que el agente
            # observado cambió su serialización, no que no haya actividad.
            self._stats["eventos_sin_timestamp"] += 1
            return
        payload = event.get("payload") or {}

        if event_type == "analysis_start":
            self._ledger.open(analysis_id)
            self._cursors[analysis_id] = _AnalysisCursor(analysis_id, ts_us)
            return  # sin evento previo no hay latencia que medir: no se observa nada todavía

        cursor = self._cursors.get(analysis_id)
        if cursor is None:
            # Traza empezada por la mitad (arranque en vivo o rotación): se adopta igual.
            self._ledger.ensure_open(analysis_id)
            cursor = _AnalysisCursor(analysis_id, ts_us)
            self._cursors[analysis_id] = cursor
            return

        # --- Dimensión: latencia entre pasos (aplica a todo evento intermedio) -------------
        if event_type != "analysis_end":
            latency = (ts_us - cursor.last_event_us) / 1_000_000.0
            if latency >= 0:
                yield self._observe(DIM_STEP_LATENCY, ts_us, latency, analysis_id)
        cursor.last_event_us = ts_us
        cursor.events_in_iteration += 1

        # --- Dimensión: eventos por iteración (se emite al CERRAR la iteración anterior) ---
        if event_type == "iteration":
            iteration = payload.get("iteration")
            if isinstance(iteration, int):
                if cursor.current_iteration >= 0 and cursor.events_in_iteration > 1:
                    # Se descuenta el propio evento `iteration` que abre la nueva.
                    yield self._observe(
                        DIM_EVENTS_PER_ITERATION,
                        ts_us,
                        float(cursor.events_in_iteration - 1),
                        analysis_id,
                    )
                cursor.current_iteration = iteration
                cursor.events_in_iteration = 1
                self._ledger.note_iteration(analysis_id, iteration)

        # --- Dimensión: longitud del razonamiento -----------------------------------------
        elif event_type == "llm_thought":
            thought = payload.get("thought")
            if isinstance(thought, str):
                yield self._observe(DIM_THOUGHT_LEN, ts_us, float(len(thought)), analysis_id)

        # --- Dimensión: tamaño del resultado de tool --------------------------------------
        elif event_type in ("tool_result", "check_result"):
            result = payload.get("result")
            size = len(json.dumps(result, ensure_ascii=False, default=str)) if result is not None else 0
            yield self._observe(DIM_TOOL_RESULT_SIZE, ts_us, float(size), analysis_id)

        # --- Eventos que no son señal, pero definen el VEREDICTO --------------------------
        elif event_type == "alert_emitted":
            alert = payload.get("alert") or {}
            cves = alert.get("detected_cves") or []
            self._ledger.note_alert(
                analysis_id, len(cves), str(alert.get("severity", ""))
            )

        elif event_type == "error":
            self._ledger.note_error(analysis_id)

        elif event_type == "analysis_end":
            iterations = payload.get("iterations")
            self._ledger.close(
                analysis_id, iterations if isinstance(iterations, int) else None
            )
            self._cursors.pop(analysis_id, None)

    def _observe(
        self, dimension: str, ts_us: int, value: float, analysis_id: str
    ) -> Observation:
        """Arma la `Observation` y deja anotado a qué análisis pertenece.

        La anotación en el libro es lo que después permite que la fuente de verdad sepa qué
        desenlace corresponde a esta predicción, sin ensuciar el contrato `Observation` del
        núcleo con un campo específico de este dominio.
        """
        self._ledger.note_observation(ts_us, analysis_id)
        self._stats["observaciones"] += 1
        return Observation(
            entity_id=f"{self._agent}:{dimension}",
            timestamp=ts_us,
            values={VALUE_KEY: value},
        )
