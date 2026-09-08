"""Entry point del meta-agente de aprendizaje: observa a otros agentes en la VPS.

Qué hace este archivo (y qué NO)
---------------------------------
Es la RAÍZ DE COMPOSICIÓN: lee la configuración del entorno, construye las piezas del dominio
`agent_behavior` y se las inyecta al loop del núcleo. No contiene lógica de aprendizaje, ni de
detección, ni conoce el formato de las trazas. Si algo de eso apareciera acá, estaría en el
lugar equivocado.

Cómo se conecta con el agente observado
----------------------------------------
Por el sistema de archivos y en SOLO LECTURA. El agente de seguridad ya escribe su volumen
(`data/traces/<fecha>/<analysis_id>.jsonl`); este proceso lo monta como `/observed:ro` y lo
sigue con tail incremental. No hay API entre ambos, no hay puerto, no hay base de datos, y el
agente observado no se entera de que lo están mirando: no hubo que modificarle una sola línea.

Modos:
    python main.py                      # daemon: replay del histórico + seguimiento en vivo
    python main.py --live               # solo actividad nueva (ignora las trazas ya escritas)
    python main.py --no-llm             # sin LLM: exploración puramente determinista
    python main.py --max-cycles 5000    # corre N ciclos y termina (pruebas reproducibles)
    python main.py --status             # imprime el conocimiento aprendido y sale
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from core import (
    Collector,
    LearningAgentLoop,
    Observation,
    SignatureDeduper,
)
from domains.agent_behavior.audit import BoundedAuditTrail
from domains.agent_behavior.behavior_model import BehaviorNormalityModel
from domains.agent_behavior.critic import BehaviorCritic
from domains.agent_behavior.effector import MetaAlertEffector
from domains.agent_behavior.learning_element import BehaviorLearningElement
from domains.agent_behavior.ledger import AnalysisLedger
from domains.agent_behavior.llm import DeepSeekClient
from domains.agent_behavior.performance_element import BehaviorPerformanceElement
from domains.agent_behavior.problem_generator import BehaviorProblemGenerator
from domains.agent_behavior.trace_collector import TraceTailCollector
from domains.agent_behavior.truth_source import DerivedDeferredTruth
from state_server import start_state_server


class _Shutdown(Exception):
    """Señal de apagado ordenado (SIGTERM/SIGINT), para poder guardar antes de salir."""


def log(level: str, msg: str, **fields) -> None:
    """Log estructurado por stdout (una línea JSON), visible con `docker logs`."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "msg": msg,
        **fields,
    }
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


# --------------------------------------------------------------------- configuración


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "si", "sí")


# --------------------------------------------------------------------- instrumentación


class _InstrumentedCollector(Collector):
    """Envuelve al colector para emitir un latido periódico.

    Vive en la raíz de composición y no en el dominio a propósito: contar ciclos y loguear
    estado es observabilidad del PROCESO, no conocimiento del problema. El dominio no tiene por
    qué saber que alguien lo está mirando correr.
    """

    def __init__(self, inner: Collector, every: int, on_beat) -> None:
        self._inner = inner
        self._every = max(1, every)
        self._on_beat = on_beat
        self._count = 0

    def stream(self) -> Iterator[Observation]:
        for observation in self._inner.stream():
            self._count += 1
            if self._count % self._every == 0:
                self._on_beat(self._count)
            yield observation


# --------------------------------------------------------------------- construcción


def build(args) -> dict:
    """Construye el agente completo a partir del entorno y los argumentos.

    Returns:
        Diccionario con las piezas cableadas (el loop y lo que main necesita inspeccionar).
    """
    observed_agent = _env("OBSERVED_AGENT_ID", "keycloak-agent")
    meta_agent = _env("META_AGENT_ID", "meta-agent")
    traces_dir = Path(_env("OBSERVED_TRACES_DIR", "/observed/traces"))
    state_dir = Path(_env("STATE_DIR", "/state"))

    # --- Conocimiento: se restaura del snapshot para no perder lo aprendido en cada deploy.
    model = BehaviorNormalityModel(
        warmup=_env_int("WARMUP", 20),
        clean_guard_sigmas=_env_float("CLEAN_GUARD_SIGMAS", 4.0),
        threshold_step=_env_float("THRESHOLD_STEP", 0.15),
        min_threshold=_env_float("MIN_THRESHOLD", 1.5),
        max_threshold=_env_float("MAX_THRESHOLD", 6.0),
        state_path=state_dir / "behavior_model.json",
    )
    restored = model.load()
    log(
        "info",
        "Conocimiento restaurado" if restored else "Arranque en frío (sin snapshot previo)",
        entidades=len(model.entities()),
    )

    # --- Estado compartido del dominio (quién está analizando y cómo terminó).
    ledger = AnalysisLedger(
        abandon_after_seconds=_env_float("ABANDON_AFTER_SECONDS", 1800.0)
    )

    audit = BoundedAuditTrail(max_records=_env_int("AUDIT_MAX_RECORDS", 5000))

    # --- LLM: opcional por diseño. Sin credencial, el agente sigue aprendiendo igual.
    llm = None
    if not args.no_llm and _env_bool("LLM_ENABLED", True):
        candidate = DeepSeekClient(
            api_key=_env("DEEPSEEK_API_KEY", ""),
            base_url=_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=_env("DEEPSEEK_MODEL", "deepseek-chat"),
            temperature=_env_float("LLM_TEMPERATURE", 0.2),
            timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 45),
        )
        llm = candidate if candidate.enabled else None
        if llm is None:
            log("warn", "LLM habilitado pero sin DEEPSEEK_API_KEY: exploración determinista")

    # --- Las cuatro piezas de R&N + los tres puertos.
    performance = BehaviorPerformanceElement(model, ledger)
    learning = BehaviorLearningElement(
        model, ledger, save_every=_env_int("SAVE_EVERY", 200)
    )
    critic = BehaviorCritic()
    problem_generator = BehaviorProblemGenerator(
        model, audit, llm=llm, every=_env_int("EXPLORE_EVERY", 200)
    )
    effector = MetaAlertEffector(
        alerts_path=state_dir / "meta-alerts.jsonl",
        model=model,
        ledger=ledger,
        agent_id=meta_agent,
    )
    truth_source = DerivedDeferredTruth(ledger)

    # --- Inactividad: cuando el agente observado no produce eventos, el colector duerme. Se
    #     aprovecha esa ventana para bajar a disco lo aprendido (un daemon nunca "termina",
    #     así que sin este punto de control un corte de energía perdería horas de ajuste).
    idle_flush_seconds = _env_float("IDLE_FLUSH_SECONDS", 60.0)
    idle_state = {"last_flush": time.time(), "last_log": 0.0}

    def on_idle() -> None:
        now = time.time()
        if now - idle_state["last_flush"] >= idle_flush_seconds:
            idle_state["last_flush"] = now
            learning.flush()
            collector.save_offsets()
            if now - idle_state["last_log"] >= 300:
                idle_state["last_log"] = now
                diag = collector.diagnostics()
                # Si el agente nunca percibió nada, "sin actividad" es la conclusión pero no
                # el diagnóstico: se eleva a warning y se adjunta el veredicto del colector,
                # que dice en cuál eslabón de la cadena se corta la percepción.
                ciego = diag["observaciones"] == 0
                log(
                    "warn" if ciego else "info",
                    "El agente observado no produjo trazas nuevas"
                    if not ciego
                    else "Sin percepción: el agente no logró leer ninguna traza",
                    entidades=len(model.entities()),
                    analisis_abiertos=ledger.open_count(),
                    predicciones_pendientes=truth_source.pending_count(),
                    percepcion=diag,
                )

    collector = TraceTailCollector(
        traces_dir=traces_dir,
        ledger=ledger,
        agent_id=observed_agent,
        poll_interval_seconds=_env_float("POLL_INTERVAL_SECONDS", 2.0),
        from_start=not args.live,
        offsets_path=state_dir / "tail_offsets.json",
        idle_callback=on_idle,
    )
    resumed = collector.load_offsets()
    if resumed:
        log("info", "Lectura retomada desde la posición guardada (no se reprocesa el histórico)")

    def on_beat(count: int) -> None:
        log(
            "info",
            "Latido del meta-agente",
            observaciones=count,
            entidades=len(model.entities()),
            analisis_abiertos=ledger.open_count(),
            predicciones_pendientes=truth_source.pending_count(),
            umbrales={
                entity: round(model.threshold(entity), 2) for entity in model.entities()
            },
        )

    instrumented = _InstrumentedCollector(
        collector, every=_env_int("HEARTBEAT_CYCLES", 500), on_beat=on_beat
    )

    loop = LearningAgentLoop(
        collector=instrumented,
        performance=performance,
        learning=learning,
        critic=critic,
        problem_generator=problem_generator,
        effector=effector,
        truth_source=truth_source,
        audit=audit,
        deduper=SignatureDeduper(cooldown=_env_int("DEDUP_COOLDOWN_CYCLES", 50)),
    )

    return {
        "loop": loop,
        "model": model,
        "learning": learning,
        "collector": collector,
        "audit": audit,
        "ledger": ledger,
        "traces_dir": traces_dir,
        "state_dir": state_dir,
        "observed_agent": observed_agent,
        # La identidad del propio meta-agente ya se resolvía acá (la firma de las meta-alertas);
        # se publica en `parts` para que el servidor de estado la reporte sin releer el entorno.
        "meta_agent": meta_agent,
        "llm_enabled": llm is not None,
    }


# --------------------------------------------------------------------- CLI


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Meta-agente de aprendizaje sobre el comportamiento de otros agentes"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Observar solo actividad nueva (por defecto también procesa el histórico).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Exploración puramente determinista (sin llamadas al LLM).",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Cortar después de N ciclos (corridas reproducibles).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Imprimir el conocimiento aprendido y salir.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Revisar el origen de trazas y explicar por qué no hay percepción. No consume datos.",
    )
    return parser.parse_args()


def _print_status(model: BehaviorNormalityModel) -> None:
    """Vuelca el estado del conocimiento por stdout (inspección operativa)."""
    entities = model.entities()
    if not entities:
        log("info", "Todavía no hay conocimiento aprendido (sin observaciones)")
        return
    print(
        json.dumps(
            [model.stats(entity) for entity in sorted(entities)],
            ensure_ascii=False,
            indent=2,
        )
    )


def _print_diagnosis(parts: dict) -> None:
    """Explica el estado de la cadena de percepción, de punta a punta.

    Recorre los eslabones en el orden en que dependen unos de otros —montaje, archivos,
    eventos legibles, posición de lectura, conocimiento— y marca cada uno. El PRIMERO que
    falla es la causa; los de abajo son consecuencia y no hay que investigarlos.
    """
    collector = parts["collector"]
    model = parts["model"]
    probe = collector.probe()

    def mark(ok: bool) -> str:
        return "[ OK ]" if ok else "[FALLA]"

    print("\n=== Diagnóstico de la percepción ===\n")
    print(f"  Agente observado : {parts['observed_agent']}")
    print(f"  Ruta de trazas   : {probe['ruta_observada']}  (dentro del contenedor)")
    print()
    print(f"  {mark(probe['existe'])} 1. El directorio de trazas existe")
    print(f"  {mark(probe['archivos'] > 0)} 2. Hay archivos de traza          -> {probe['archivos']}")
    print(
        f"  {mark(probe['eventos_validados'] > 0)} 3. Los eventos son legibles       -> "
        f"{probe['eventos_validados']} validados, {probe['eventos_rechazados']} rechazados"
    )
    offsets = parts["state_dir"] / "tail_offsets.json"
    print(f"  {mark(True)} 4. Posición de lectura guardada   -> {'sí' if offsets.exists() else 'no (arranque limpio)'}")
    entidades = len(model.entities())
    print(f"  {mark(entidades > 0)} 5. Conocimiento aprendido         -> {entidades} dimensiones")

    if probe["archivos_recientes"]:
        print("\n  Archivos más recientes:")
        for name in probe["archivos_recientes"]:
            print(f"    {name}")
    if probe["detalle"]:
        print("\n  Validación de los primeros eventos del archivo más reciente:")
        for line in probe["detalle"]:
            print(f"    {line}")

    print(f"\n  VEREDICTO: {probe['veredicto']}\n")


def main() -> None:
    args = _parse_args()

    def _handle_signal(signum, _frame):
        raise _Shutdown(f"señal {signum}")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    parts = build(args)
    model: BehaviorNormalityModel = parts["model"]
    learning: BehaviorLearningElement = parts["learning"]

    if args.status:
        _print_status(model)
        return

    if args.diagnose:
        _print_diagnosis(parts)
        return

    traces_dir: Path = parts["traces_dir"]
    if not traces_dir.is_dir():
        # No es fatal: el agente observado puede no haber generado trazas todavía. Se avisa
        # fuerte porque es EL error de despliegue más probable (volumen mal montado).
        log(
            "warn",
            "El directorio de trazas no existe todavía; se esperará a que aparezca",
            traces_dir=str(traces_dir),
        )

    # --- Exposición del estado: AÑADIDO opcional, arranca después del `--status` porque ese
    #     modo imprime y sale (no tendría a quién servirle). Si el puerto está ocupado devuelve
    #     None y el agente sigue aprendiendo: la observabilidad nunca condiciona al aprendizaje.
    state_server = None
    if _env_bool("STATE_SERVER_ENABLED", True):
        state_server = start_state_server(
            parts,
            host=_env("STATE_SERVER_HOST", "0.0.0.0"),
            port=_env_int("STATE_SERVER_PORT", 8090),
            log=log,
        )

    log(
        "info",
        "Meta-agente iniciado",
        agente_observado=parts["observed_agent"],
        traces_dir=str(traces_dir),
        state_dir=str(parts["state_dir"]),
        modo="live" if args.live else "replay+live",
        llm=parts["llm_enabled"],
        estado_expuesto=state_server is not None,
        max_cycles=args.max_cycles,
    )

    try:
        parts["loop"].run(max_cycles=args.max_cycles)
        log("info", "Stream finalizado")
    except _Shutdown as exc:
        log("info", "Apagado ordenado", motivo=str(exc))
    except Exception as exc:  # noqa: BLE001 - se guarda lo aprendido antes de propagar
        learning.flush()
        log("error", "Fallo no controlado", error=str(exc))
        raise
    finally:
        # Guardar SIEMPRE: lo aprendido es el activo del agente y no puede depender de que la
        # salida haya sido limpia. Se guarda también hasta dónde se leyó, para que el próximo
        # arranque continúe en vez de reprocesar.
        if state_server is not None:
            # Cerrar el socket antes de persistir: es un thread daemon y moriría solo, pero
            # dejarlo escuchando mientras el agente ya no aprende haría que el dashboard
            # mostrara un estado congelado como si estuviera vivo.
            state_server.shutdown()
        saved = model.save()
        parts["collector"].save_offsets()
        audit = parts["audit"]
        log(
            "info",
            "Estado persistido" if saved else "No se pudo persistir el estado",
            ciclos_totales=getattr(audit, "total_cycles", None),
            entidades=len(model.entities()),
        )


if __name__ == "__main__":
    main()
