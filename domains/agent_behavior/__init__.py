"""Dominio `agent_behavior`: el agente de aprendizaje observando a OTROS agentes.

Qué problema resuelve
---------------------
Un esquema multi-agente donde cada agente es especialista (el primero: análisis de
seguridad sobre Keycloak) necesita una capa que mire el CONJUNTO: qué comportamiento es
normal en cada agente, cuándo un agente se está descarrilando, y qué patrones se repiten.
Este dominio es esa capa.

Por qué encaja en la familia de problemas del núcleo
-----------------------------------------------------
El contrato de aplicabilidad exige tres piezas; este dominio las provee así:

  (1) observaciones por entidad en el tiempo
      -> entidad = `<agente>:<dimensión-de-comportamiento>` (ej. `keycloak-agent:step_latency_s`).
         El colector las extrae de las trazas del ciclo agéntico que el agente observado
         ya escribe en disco.

  (2) una noción de normalidad
      -> baseline móvil por dimensión de comportamiento (media/desvío en sigmas), igual que
         el dominio de referencia. "Normal" = cómo se comporta HABITUALMENTE ese agente.

  (3) un mecanismo de verdad diferida
      -> DERIVADA de los datos: mientras un análisis del agente observado está EN CURSO, el
         meta-agente juzga su comportamiento sin saber cómo termina. Cuando el análisis
         cierra, se sabe si produjo algo (alerta con CVEs) o giró en falso. Esa es la verdad,
         y llega estrictamente después de la predicción.

La decisión de modelado que hace que esto escale
-------------------------------------------------
La entidad es `<agente>:<dimensión>`, NO el CVE ni la métrica del host observado. Consecuencia:
sumar un segundo, tercer o cuarto agente especialista es simplemente MÁS `entity_id` en el
mismo stream. No cambia el núcleo y no cambia este dominio. El meta-agente no aprende sobre
Keycloak: aprende sobre AGENTES. Keycloak es el primer caso, no el objeto de estudio.

Regla de dependencia respetada
------------------------------
Este paquete importa de `core`; `core` no sabe que este dominio existe. Y NADA de este paquete
escribe sobre el agente observado: su directorio se monta de solo lectura.
"""

from .behavior_model import BehaviorNormalityModel
from .ledger import AnalysisLedger
from .trace_collector import TraceTailCollector

__all__ = ["BehaviorNormalityModel", "AnalysisLedger", "TraceTailCollector"]
