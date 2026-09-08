"""Generador de trazas sintéticas con ground truth, para validar el dominio sin la VPS.

Para qué existe
---------------
El equivalente de `domains/reference/data_gen.py`: permite ejercitar el loop completo y medir
la curva de aprendizaje de forma reproducible, sin depender de que el agente vigilado esté
corriendo ni de cuántos análisis haya acumulado. Es también la especificación ejecutable del
formato que este dominio consume — si el formato de traza del agente observado cambiara, este
archivo es el que lo documenta.

Qué simula
----------
Dos poblaciones de análisis, con la misma estructura de eventos que escribe el agente real:

  * PRODUCTIVOS   (~80%): el agente razona con su cadencia habitual y termina emitiendo una
                          alerta con CVEs detectadas.
  * DESCARRILADOS (~20%): latencias entre pasos varias veces mayores, razonamiento más largo,
                          resultados de tools más pobres, muchas más iteraciones, y cierre sin
                          alerta (a veces con un error). Es el patrón que el meta-agente debe
                          aprender a anticipar ANTES de que el análisis termine.

La separación entre poblaciones es deliberadamente parcial, no limpia: se solapan. Un generador
que produjera dos nubes perfectamente separables haría trivial la detección y la curva de
aprendizaje no probaría nada.
"""

from __future__ import annotations

import json
import random
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Set, Tuple

# Perfiles de comportamiento. Los valores replican el orden de magnitud de un ciclo ReAct real
# (segundos entre pasos, cientos/miles de caracteres de razonamiento) y, sobre todo, SE SOLAPAN.
# Un generador con dos nubes perfectamente separables volvería trivial la detección: el agente
# acertaría desde el primer ciclo y la curva de aprendizaje no probaría nada. El solapamiento es
# el que obliga a que el umbral se ajuste con la experiencia para separar las poblaciones.
_NORMAL = {
    "latency": (3.0, 0.9),
    "thought": (820, 180),
    "result": (2100, 450),
    "iterations": (3, 7),
    "tools_per_iteration": (1, 2),
}
_DERAILED = {
    "latency": (5.5, 2.0),
    "thought": (1450, 420),
    "result": (1450, 450),
    "iterations": (8, 16),
    "tools_per_iteration": (1, 4),
}


def _gauss(mu: float, sigma: float, floor: float) -> float:
    return max(floor, random.gauss(mu, sigma))


def generate(
    traces_dir: Path | str,
    analyses: int = 120,
    derailed_ratio: float = 0.20,
    seed: int = 1234,
) -> Tuple[Path, Set[str]]:
    """Escribe trazas sintéticas y devuelve el ground truth.

    Args:
        traces_dir: destino (se recrea desde cero).
        analyses: cuántos análisis simular.
        derailed_ratio: proporción de análisis que terminan sin producir nada.
        seed: semilla, para que el experimento sea reproducible.

    Returns:
        `(directorio, ids_de_analisis_improductivos)`. El segundo elemento es el ground truth:
        los análisis que el meta-agente debería haber marcado.
    """
    random.seed(seed)
    out = Path(traces_dir)
    if out.exists():
        shutil.rmtree(out)

    clock = datetime(2026, 8, 10, 9, 0, 0, tzinfo=timezone.utc)
    unproductive: Set[str] = set()

    for index in range(analyses):
        derailed = random.random() < derailed_ratio
        analysis_id, clock = _write_analysis(out, clock, derailed)
        if derailed:
            unproductive.add(analysis_id)
        if index % 25 == 24:
            # Salto de horas: ejercita la rotación por carpeta de fecha del agente observado.
            clock += timedelta(hours=6)

    return out, unproductive


def _write_analysis(
    out: Path, clock: datetime, derailed: bool
) -> Tuple[str, datetime]:
    """Escribe un análisis completo como JSONL y devuelve su id y el reloj avanzado."""
    profile = _DERAILED if derailed else _NORMAL
    analysis_id = f"an-{uuid.uuid4().hex[:16]}"
    day_dir = out / clock.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    seq = 0

    def emit(event_type: str, payload: dict, gap: float) -> None:
        nonlocal seq, clock
        clock += timedelta(seconds=gap)
        lines.append(
            json.dumps(
                {
                    "analysis_id": analysis_id,
                    "event_id": uuid.uuid4().hex,
                    "seq": seq,
                    "timestamp": clock.isoformat(),
                    "event_type": event_type,
                    "payload": payload,
                },
                ensure_ascii=False,
            )
        )
        seq += 1

    lat_mu, lat_sd = profile["latency"]
    th_mu, th_sd = profile["thought"]
    res_mu, res_sd = profile["result"]
    iterations = random.randint(*profile["iterations"])

    # Hueco desde el análisis anterior: el agente observado no analiza continuamente.
    emit("analysis_start", {"state_summary": "{...}"}, gap=_gauss(60, 20, 5))

    for i in range(iterations):
        emit("iteration", {"iteration": i}, gap=_gauss(lat_mu, lat_sd, 0.05))
        emit(
            "llm_thought",
            {"iteration": i, "thought": "x" * int(_gauss(th_mu, th_sd, 50))},
            gap=_gauss(lat_mu, lat_sd, 0.05),
        )
        # Cuántas tools invoca esta iteración: es lo que hace variar `events_per_iteration`.
        # Un agente que gira en falso tiende a encadenar más llamadas por iteración, pero los
        # rangos se solapan (una iteración normal también puede necesitar dos tools).
        for _ in range(random.randint(*profile["tools_per_iteration"])):
            tool = random.choice(
                ["run_deterministic_check", "get_host_metrics", "query_knowledge_base"]
            )
            emit(
                "tool_call",
                {"iteration": i, "tool": tool, "arguments": {}},
                gap=_gauss(0.4, 0.15, 0.01),
            )
            emit(
                "check_result" if tool == "run_deterministic_check" else "tool_result",
                {
                    "iteration": i,
                    "tool": tool,
                    "result": {"blob": "y" * int(_gauss(res_mu, res_sd, 20))},
                },
                gap=_gauss(lat_mu, lat_sd, 0.05),
            )

    if derailed:
        if random.random() < 0.35:
            emit("error", {"error": "LLM timeout", "where": "react_loop"}, gap=_gauss(5, 1, 0.1))
        # Sin `alert_emitted`: el análisis consumió tiempo y no concluyó nada. Esa ausencia ES
        # la verdad diferida de este dominio.
        emit("analysis_end", {"iterations": iterations}, gap=_gauss(2, 0.5, 0.1))
    else:
        emit(
            "alert_emitted",
            {
                "alert": {
                    "alert_id": f"alert-{uuid.uuid4().hex[:12]}",
                    "severity": random.choice(["high", "critical"]),
                    "detected_cves": [
                        {"cve_id": "CVE-2023-0264", "confidence": 0.93}
                    ],
                }
            },
            gap=_gauss(2, 0.5, 0.1),
        )
        emit("analysis_end", {"iterations": iterations}, gap=_gauss(1, 0.3, 0.1))

    (day_dir / f"{analysis_id}.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return analysis_id, clock
