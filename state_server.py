"""Servidor HTTP de solo lectura sobre el estado propio del meta-agente.

Qué es y qué NO es
-------------------
Es un AÑADIDO. No cambia el núcleo, no cambia el dominio, no cambia una línea de lo que el
agente aprende ni de cómo lo aprende. Expone hacia afuera lo que hasta hoy solo se podía ver
con `--status` o con `tail -f state/meta-alerts.jsonl`: el conocimiento aprendido y las
meta-alertas emitidas. Si este servidor no arranca, el meta-agente sigue funcionando igual —
igual que el dashboard del agente observado, la observabilidad es opcional por diseño.

Por qué sirve el objeto en memoria y no el archivo
---------------------------------------------------
`behavior_model.json` es un SNAPSHOT: se baja a disco cada `SAVE_EVERY` escrituras o tras
`IDLE_FLUSH_SECONDS` de inactividad. Leer el archivo mostraría un estado con minutos de atraso
y, peor, `std` no está en el snapshot (están `count` y `m2`, de los que se deriva): un lector
externo tendría que reimplementar Welford en otro lenguaje y esa duplicación se desincroniza el
día que el modelo cambie. Sirviendo `model.stats(entity)` —exactamente lo que imprime
`--status`— hay una sola definición de la verdad y está en Python, donde vive el dominio.

Las meta-alertas SÍ salen del archivo, porque el archivo es su registro canónico (append-only,
persistente entre reinicios); la traza en memoria es una ventana deslizante que olvida.

Por qué no viola la promesa de no perturbar lo observado
---------------------------------------------------------
Sirve `/state`, el volumen PROPIO del meta-agente. El territorio del agente observado sigue
montado `:ro` y este módulo ni lo referencia. Además todos los handlers son de lectura: no hay
un solo verbo que escriba, ni sobre el estado ni sobre el modelo.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

# Tope de meta-alertas devueltas por request. El JSONL de un daemon de semanas puede pesar
# megabytes; un dashboard nunca necesita más que las últimas. Se lee la cola del archivo, no
# el archivo entero.
_DEFAULT_ALERT_LIMIT = 200
_MAX_ALERT_LIMIT = 2000

# Umbral con el que nace toda entidad (ver `_EntityBaseline.threshold_sigmas`). Es el ORIGEN
# contra el que se lee cuánto se movió cada dimensión, así que se expone explícitamente en
# lugar de dejar que el dashboard lo hardcodee por su cuenta.
_INITIAL_THRESHOLD = 2.0


def _json_bytes(payload) -> bytes:
    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")


def _tail_jsonl(path: Path, limit: int) -> List[dict]:
    """Devuelve las últimas `limit` líneas parseables de un JSONL.

    Args:
        path: archivo append-only de meta-alertas.
        limit: cuántos registros retener (los más recientes).

    Returns:
        Lista de objetos, del más antiguo al más reciente. Archivo ausente = lista vacía: el
        meta-agente todavía no emitió ninguna meta-alerta, que es un estado NORMAL, no un error.
    """
    if not path.exists():
        return []
    window: deque = deque(maxlen=limit)
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    window.append(stripped)
    except OSError:
        return []
    records: List[dict] = []
    for raw in window:
        try:
            records.append(json.loads(raw))
        except ValueError:
            # Línea a medio escribir (el effector estaba anexando): se ignora sin romper.
            continue
    return records


def _make_handler(parts: dict, started_at: float):
    """Crea el handler ligado a las piezas VIVAS del agente.

    Args:
        parts: el diccionario que devuelve `build()` en main.py (modelo, ledger, auditoría...).
        started_at: epoch de arranque, para reportar uptime.

    Returns:
        Clase handler para ``ThreadingHTTPServer``.
    """
    model = parts["model"]
    ledger = parts["ledger"]
    audit = parts["audit"]
    alerts_path = Path(parts["state_dir"]) / "meta-alerts.jsonl"

    class StateHandler(BaseHTTPRequestHandler):
        """Handler de SOLO LECTURA sobre el estado del meta-agente."""

        def log_message(self, *args) -> None:  # noqa: D401, ANN002
            """Silencia el log por-request: el canal de logs es del aprendizaje, no del HTTP."""

        # --- Rutas -----------------------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 - nombre impuesto por la stdlib
            """Resuelve las cuatro rutas del estado expuesto."""
            parsed = urlparse(self.path)
            route = parsed.path.rstrip("/") or "/"
            handlers: Dict[str, Callable] = {
                "/api/health": self._health,
                "/api/config": self._config,
                "/api/model": self._model,
                "/api/meta-alerts": self._meta_alerts,
            }
            handler = handlers.get(route)
            if handler is None:
                self._send(404, {"error": "not found", "routes": sorted(handlers)})
                return
            try:
                self._send(200, handler(parse_qs(parsed.query)))
            except Exception as exc:  # noqa: BLE001 - exponer el estado no puede tumbar el agente
                self._send(500, {"error": str(exc)})

        def do_OPTIONS(self) -> None:  # noqa: N802
            """Preflight de CORS: el dashboard puede servirse desde otro origen."""
            self.send_response(204)
            self._cors()
            self.end_headers()

        # --- Payloads --------------------------------------------------------------------

        def _health(self, _query) -> dict:
            """Latido: identidad del esquema y contadores de actividad."""
            return {
                "status": "ok",
                "meta_agent": parts["meta_agent"],
                "observed_agent": parts["observed_agent"],
                "llm_enabled": parts["llm_enabled"],
                "uptime_seconds": round(time.time() - started_at, 1),
                "total_cycles": getattr(audit, "total_cycles", None),
                "entities": len(model.entities()),
                "open_analyses": ledger.open_count(),
                "traces_dir": str(parts["traces_dir"]),
            }

        def _config(self, _query) -> dict:
            """Hiperparámetros del aprendizaje: los diales que gobiernan el ajuste.

            Se exponen porque un umbral de 3.20 sigmas no dice nada sin saber que arrancó en
            2.00, que se mueve de a 0.15 y que topea en 6.00. Sin estos valores el dashboard
            mostraría números; con ellos muestra un recorrido.
            """
            return {
                "warmup": model.warmup,
                "clean_guard_sigmas": model.clean_guard_sigmas,
                "threshold_step": model.threshold_step,
                "min_threshold": model.min_threshold,
                "max_threshold": model.max_threshold,
                "initial_threshold": _INITIAL_THRESHOLD,
            }

        def _model(self, _query) -> dict:
            """El conocimiento aprendido, por `<agente>:<dimensión>`. Igual que `--status`."""
            entities = sorted(model.entities())
            return {
                "entities": [model.stats(entity) for entity in entities],
                # `is_warm` y `seen_count` no están en `stats()` y el dashboard los necesita
                # para distinguir "todavía no juzga" de "juzga y no alerta", y para exponer
                # cuántas observaciones rechazó la guarda de limpieza (seen - count). Se
                # agregan acá, en la capa de exposición, no en el dominio.
                "warm": {entity: model.is_warm(entity) for entity in entities},
                "seen": {entity: model.seen_count(entity) for entity in entities},
            }

        def _meta_alerts(self, query) -> dict:
            """Las últimas meta-alertas emitidas, desde su registro canónico en disco."""
            limit = _DEFAULT_ALERT_LIMIT
            raw = (query.get("limit") or [None])[0]
            if raw is not None:
                try:
                    limit = max(1, min(_MAX_ALERT_LIMIT, int(raw)))
                except ValueError:
                    pass  # limit inválido: se cae al default en vez de fallar el request
            records = _tail_jsonl(alerts_path, limit)
            return {
                "alerts": records,
                "returned": len(records),
                "limit": limit,
                "source": str(alerts_path),
            }

        # --- Transporte ------------------------------------------------------------------

        def _cors(self) -> None:
            """Permite el acceso desde cualquier origen.

            Es defendible porque no hay nada que proteger con una política de origen: la
            superficie es de solo lectura y no hay credenciales ni cookies en juego. Quien no
            quiera exponerla, no publica el puerto (`STATE_SERVER_ENABLED=false`) — que es el
            control real, y no un header.
            """
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")

        def _send(self, status: int, payload) -> None:
            """Escribe una respuesta JSON sin caché."""
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            self.wfile.write(body)

    return StateHandler


def start_state_server(
    parts: dict,
    host: str = "0.0.0.0",
    port: int = 8090,
    log: Optional[Callable[..., None]] = None,
) -> Optional[ThreadingHTTPServer]:
    """Inicia el servidor de estado en un thread daemon.

    Args:
        parts: piezas cableadas del agente (lo que devuelve `build()`).
        host / port: dónde escuchar.
        log: la función de log estructurado de main.py, inyectada para no duplicarla.

    Returns:
        El servidor iniciado, o `None` si el puerto está ocupado. NUNCA propaga: que el estado
        no se pueda exponer es un problema de observabilidad, no del aprendizaje, y el agente
        tiene que seguir corriendo.
    """
    handler = _make_handler(parts, started_at=time.time())
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        if log:
            log(
                "warn",
                "No se pudo exponer el estado (¿puerto ocupado?)",
                port=port,
                error=str(exc),
            )
        return None

    thread = threading.Thread(target=server.serve_forever, daemon=True, name="state-server")
    thread.start()
    if log:
        log(
            "info",
            "Estado expuesto por HTTP (solo lectura)",
            url=f"http://{host}:{port}/api/model",
        )
    return server
