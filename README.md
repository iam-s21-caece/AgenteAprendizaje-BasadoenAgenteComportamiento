# Agente de aprendizaje sobre el comportamiento del agente analista

Agente de aprendizaje (taxonomía Russell & Norvig) que **supervisa a otro agente**: observa cómo
se comporta el agente analista de seguridad mientras razona, aprende cuál es su comportamiento
normal, y avisa cuando ese comportamiento se desvía — antes de que el análisis termine.

No analiza Keycloak. No analiza CVEs. No duplica nada de lo que hace el analista. Su objeto de
estudio es **el agente**, y la pregunta que responde es:

> *Este análisis que el analista está ejecutando ahora mismo, ¿va a terminar en algo, o está
> girando en falso?*

Es la primera pieza de un esquema **multi-agente**: mientras cada agente especialista resuelve su
problema, este lleva el análisis general del comportamiento de todos.

## Los dos agentes y cómo se comunican

| | Agente analista | Meta-agente (este repo) |
|---|---|---|
| Rol | detecta ataques sobre Keycloak | supervisa al analista |
| Taxonomía R&N | basado en modelos + en objetivos (ReAct) | **de aprendizaje** |
| Entrada | métricas del host, eventos OIDC | las trazas del analista |
| Salida | alertas de seguridad (`alerts.jsonl`) | meta-alertas (`meta-alerts.jsonl`) |
| Contenedor | `keycloak-security-agent` | `meta-agente-aprendizaje` |

La comunicación es **por el sistema de archivos, en un solo sentido y de solo lectura**:

```
   AGENTE ANALISTA                                 META-AGENTE
   (keycloak-security-agent)                       (meta-agente-aprendizaje)

   analiza un ataque
   y escribe su ciclo agentico  ─────────────┐
   en data/traces/<fecha>/<id>.jsonl         │
                                             ▼
                                    /observed (ro) ── tail incremental
                                                          │
                                                          ▼
                                             percibir → decidir → aprender
                                                          │
                                                          ▼
                                                     /state (rw)
                                                      meta-alerts.jsonl
                                                      behavior_model.json
```

Tres propiedades de esa elección, que son deliberadas:

1. **El analista no se modificó ni una línea.** Ya escribía sus trazas en disco; el meta-agente
   simplemente las lee. No hubo que agregarle un endpoint, un hook ni una dependencia.
2. **No puede perturbar lo observado.** El volumen del analista se monta `:ro`. No es una promesa
   de diseño: es una garantía del kernel.
3. **No hay acoplamiento de procesos.** No hay API, no hay puerto, no hay orden de arranque. Si el
   meta-agente se cae, el analista ni se entera; si el analista se cae, el meta-agente lo detecta
   como un análisis abandonado.

## Qué observa del analista

El analista deja un evento de traza por cada paso de su razonamiento (`analysis_start`,
`iteration`, `llm_thought`, `tool_call`, `tool_result`, `check_result`, `alert_emitted`, `error`,
`analysis_end`). De ese flujo, el colector extrae cuatro dimensiones de comportamiento:

| Entidad | Qué mide | Qué delata |
|---|---|---|
| `keycloak-agent:step_latency_s` | segundos entre dos eventos consecutivos | el analista se traba o pelea con el LLM |
| `keycloak-agent:thought_len` | caracteres del razonamiento en cada paso | razonamiento desbocado o vacío |
| `keycloak-agent:tool_result_size` | tamaño del resultado de cada herramienta | tools devolviendo nada, o volcados desmedidos |
| `keycloak-agent:events_per_iteration` | eventos que consumió la iteración que cerró | la firma clásica del ReAct girando en falso |

Las cuatro son **estacionarias** a propósito. Una magnitud que crece por construcción dentro del
análisis (el número de iteración, el total acumulado de tools) se desviaría trivialmente al final
de cada análisis, y el agente aprendería a alertar sobre el paso del tiempo.

La entidad es `<agente>:<dimensión>` y no el problema que el analista resuelve. Esa es la decisión
que hace escalar el esquema: **sumar un segundo, tercer o cuarto agente especialista es solo más
`entity_id` en el mismo stream** — no cambia el núcleo, no cambia este dominio, no cambia el
despliegue. El meta-agente no aprende sobre Keycloak: aprende sobre agentes.

## Cómo aprende

El ciclo de R&N, cerrado (`core/loop.py`):

```
percibir → decidir → aprender(continuo) → actuar
        → (más tarde) recibir verdad → criticar → aprender(correctivo) → explorar → trazar
```

**La verdad diferida se deriva del entorno, sin etiquetado humano.** Un análisis del analista fue
*productivo* si terminó emitiendo una alerta con contenido (CVEs detectadas o severidad por encima
de `info`). Si consumió iteraciones y no concluyó nada, o murió con un error, el esfuerzo que el
meta-agente observó fue esfuerzo desperdiciado:

```
was_anomaly = NOT productivo
```

**El diferimiento es estructural, no cosmético.** Es la objeción obvia contra una verdad derivada
de los mismos datos, así que conviene dejarla cerrada: el desenlace vive en el evento de cierre,
que el colector todavía no leyó cuando emite las observaciones de ese análisis. Al momento de
predecir, la información que define la verdad **no existe en el stream**. Por eso el colector lee
evento por evento y nunca hace *lookahead*: si leyera análisis completos, esto no sería
aprendizaje sino un lookup con retardo.

**Qué cambia cuando aprende.** El conocimiento es un umbral adaptativo por dimensión, legible y
auditable — no un modelo opaco. Un falso positivo lo sube (era trabajo legítimo del analista), un
falso negativo lo baja. Se puede leer, ciclo a ciclo, por qué el agente decidió distinto hoy que
ayer.

### Dónde entra el LLM

Solo en el **problem generator**: recibe la evidencia reciente y propone *dónde mirar*. Es el
único componente de R&N cuyo trabajo es generar hipótesis, y el único donde equivocarse es barato
— una mala sugerencia hace mirar en la dirección incorrecta, no emite una alerta ni mueve un
umbral.

Queda fuera de los otros tres, y por razones concretas:

- en el **performance element** volvería la decisión una caja negra y haría inmedible la curva de
  aprendizaje (no se podría atribuir un cambio de decisión al ajuste del umbral);
- en el **critic** destruiría la externalidad de la verdad: el agente se estaría calificando solo;
- en el **learning element** el conocimiento dejaría de ser un umbral legible.

Sin API key el agente sigue aprendiendo igual, con exploración determinista (`--no-llm`).

## Evidencia de que aprende

```bash
python experiments/behavior_learning_curve.py 400
```

Simula trazas del analista con ground truth conocido —con poblaciones **solapadas** a propósito:
si fueran separables, acertaría desde el primer ciclo y no probaría nada— y mide por episodio.
Salida real:

```
episodios    0+ | precision= 54.5% [################..............] recall=100.0%
episodios   20+ | precision=100.0% [##############################] recall=100.0%
...
episodios  380+ | precision=100.0% [##############################] recall=100.0%

keycloak-agent:tool_result_size   umbral=3.20 sigmas  (FP=8 FN=0)   <- aprendio aca
keycloak-agent:step_latency_s     umbral=2.15 sigmas  (FP=1 FN=0)
```

Se mide **por episodio** y no por observación: un análisis produce decenas de observaciones y la
verdad es del análisis entero, así que medir por observación mezclaría la calidad del juicio con
cuántas veces se vio el mismo episodio. Los umbrales muestran *dónde* aprendió — `tool_result_size`
era la dimensión ruidosa y su umbral subió de 2.0 a 3.2 sigmas; las demás casi no se movieron.

Esa asimetría (verdad por episodio, predicciones por observación) obligó a dos reglas de
**asignación de crédito** en el learning element, documentadas en el módulo: un episodio mueve el
umbral una sola vez por dimensión, y un episodio ya detectado por alguna dimensión no cuenta como
*miss* de las demás. Sin ellas el umbral se hunde al piso y el agente no aprende: la precisión
queda plana oscilando alrededor del 50%.

## Despliegue en la VPS

Junto al agente analista, como **carpetas hermanas**:

```bash
cd /root/keycloak
git clone https://github.com/iam-s21-caece/Agente-Aprendizaje-General.git agenteAprendizajeGn
cd agenteAprendizajeGn
docker compose -f docker-compose.meta.yml up -d --build
docker logs -f meta-agente-aprendizaje
```

**No hace falta ningún archivo de configuración para arrancar.** El compose no usa `env_file:`
(que haría fallar a un clon recién bajado, porque `.env` está gitignoreado por contener secretos):
usa interpolación con defaults. Para configurar algo —la API key del LLM, otra ruta al analista,
otros umbrales— se copia `.env.example` a `.env` y se edita.

Los volúmenes:

```
/root/keycloak/agente/data  ->  /observed  (ro)   # territorio del analista
./state                     ->  /state     (rw)   # lo unico que este agente escribe
```

Si el analista **no** está en `../agente/data`, hay que fijar `OBSERVED_DATA_DIR`. Ojo con ese
caso: si la ruta no existe, Docker crea un directorio vacío en vez de fallar y el meta-agente
queda mirando la nada. Se detecta en el log de arranque:

```json
{"level": "warn", "msg": "El directorio de trazas no existe todavía"}
```

### Verificar que está aprendiendo

```bash
docker compose -f docker-compose.meta.yml run --rm meta-agente --status
```

- *"Todavía no hay conocimiento aprendido"* → el analista nunca escribió trazas.
- `count` por debajo de `WARMUP` (20) → hay señal pero no alcanza. Bajar `WARMUP`, o encender
  `always_analyze` en el `.env` **del analista** (es una variable de entorno, no un cambio de
  código: el analista solo escribe trazas cuando un check determinista dispara).
- `count > 20` y umbrales distintos de `2.00` → está aprendiendo. Ahí ya sirve
  `tail -f state/meta-alerts.jsonl`.

En `/state` quedan `behavior_model.json` (el conocimiento aprendido), `meta-alerts.jsonl` y
`tail_offsets.json` (hasta dónde leyó). Los tres sobreviven al reinicio: el conocimiento no se
pierde en cada deploy y el histórico no se reprocesa.

La imagen no ejecuta `pip install`: es todo librería estándar, incluido el cliente del LLM (habla
HTTP con `urllib`).

### El estado, por HTTP

`--status` y `tail -f` alcanzan para operar, pero no para mirar. Para eso el agente expone su
estado —el **propio**, nunca el del observado— en un puerto de solo lectura (`state_server.py`):

```
GET :8090/api/health        latido: entidades vigiladas, análisis abiertos, ciclos, uptime
GET :8090/api/config        warmup, guarda de limpieza, paso y topes del umbral
GET :8090/api/model         estado por <agente>:<dimensión> — idéntico a `--status`
GET :8090/api/meta-alerts   últimas meta-alertas (?limit=N, tope 2000)
```

Es lo que consume el dashboard de `pro-inves-frontend`. Tres cosas que lo mantienen honesto:

1. **Sirve el objeto en memoria, no el archivo.** `behavior_model.json` es un snapshot que se
   baja cada `SAVE_EVERY` escrituras; leerlo mostraría minutos de atraso, y además `std` no está
   ahí (se deriva de `count` y `m2`), así que un lector externo tendría que reimplementar Welford
   en otro lenguaje. Sirviendo `model.stats()` hay una sola definición de la verdad. Las
   meta-alertas sí salen del archivo, que es su registro canónico.
2. **No toca la promesa de no perturbar lo observado.** El puerto sirve `/state`; el territorio
   del analista sigue montado `:ro` y este módulo ni lo referencia. Todos los handlers son GET.
3. **Es opcional.** Si el puerto está ocupado, loguea y el agente sigue aprendiendo. Con
   `STATE_SERVER_ENABLED=false` el agente vuelve a su forma anterior: sin puerto y sin superficie
   de red. Para no exponerlo a internet: `STATE_SERVER_PUBLISH=127.0.0.1:8090` y túnel SSH.

## Arquitectura: por qué el núcleo no sabe nada de esto

El agente está partido en un **núcleo agnóstico** y un **dominio**. La dirección de dependencia es
siempre `domains → core`, nunca al revés: el núcleo no importa nada de `domains/` y no sabe qué
dominio corre.

```
core/                      # 100% agnostico. Solo CONTRATOS + orquestacion.
  contracts.py             #   Observation, Prediction, DeferredTruth, Correction...
  performance_element.py   #   R&N: percibe y decide              (ABC)
  learning_element.py      #   R&N: modifica el conocimiento      (ABC)
  critic.py                #   R&N: verdad diferida -> correccion (ABC)
  problem_generator.py     #   R&N: sugiere exploracion           (ABC)
  ports.py                 #   sensores / actuadores / fuente de verdad (ABC)
  audit.py                 #   transversal: traza auditable por ciclo
  dedup.py                 #   transversal: dedupe por firma + cooldown
  loop.py                  #   el ciclo R&N que CIERRA el bucle

domains/agent_behavior/    # ESTE caso: supervisar al agente analista.
  trace_collector.py       #   UNICO lugar que conoce el formato de traza; tail incremental
  behavior_model.py        #   normalidad por <agente>:<dimension> + persistencia en disco
  ledger.py                #   estado de los analisis del analista y su desenlace
  performance_element.py / learning_element.py / critic.py / problem_generator.py
  effector.py              #   emite la meta-alerta
  truth_source.py          #   verdad DERIVADA: el analisis produjo algo?
  llm.py                   #   cliente LLM opcional (stdlib), solo para exploracion
  audit.py                 #   traza acotada, para operacion continua
  trace_gen.py             #   trazas sinteticas con ground truth

domains/reference/         # molde minimo sobre CSV: documenta como pivotar a otro dominio
main.py                    # entry point (daemon de la VPS)
state_server.py            # expone el estado PROPIO por HTTP, solo lectura. Opcional.
experiments/               # curvas de aprendizaje de ambos dominios
```

Esa separación no es ceremonia: es lo que permite que **sumar los próximos agentes al esquema no
toque el núcleo**, y lo que deja abierta la migración de persistencia. Hoy el bus entre agentes es
el archivo; el día que sean cuatro agentes y el volumen lo justifique, un
`PostgresTraceCollector` al lado del actual respeta el mismo puerto `Collector` y ni el núcleo ni
el resto del dominio se enteran.

**Regla no negociable:** si un dominio nuevo obliga a cambiar `core/`, probablemente el problema
no pertenece a esta familia — o se está metiendo lógica de dominio donde no va.

### La familia de problemas que el núcleo soporta

Detección sobre normalidad con verdad diferida. Un dominio encaja si puede proveer:

1. **Observaciones por entidad en el tiempo** → `core.Observation`
2. **Una noción de normalidad** → cuantificable como desvío en *sigmas*
3. **Un mecanismo de verdad diferida** → `core.DeferredTruth` vía `DeferredTruthSource`

Este caso las provee así: (1) dimensiones de comportamiento por análisis del analista,
(2) baseline móvil por dimensión, (3) el desenlace del análisis observado.

## Extensiones futuras (documentadas, no implementadas a propósito)

- **Más agentes supervisados**: cada uno aporta sus `entity_id` al mismo stream. Es el caso para
  el que se diseñó y no requiere cambios.
- **Otros orígenes de traza** (API, base de datos): son *colectores adicionales* que respetan
  `Collector.stream() → Iterator[Observation]`. Nada más cambia.
- **Modelos de normalidad más ricos** (estacionalidad, multivariado): reemplazan el baseline del
  dominio sin tocar el núcleo.
- **Canal de vuelta hacia el analista** (que el meta-agente ajuste sus parámetros en caliente): hoy
  NO existe, y es deliberado. La supervisión es de solo lectura; un orquestador que *manda*
  requiere un contrato distinto del actual `Effector.act(prediction)`.

## Referencias

- Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
- Yao, S., et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023.

## Requisitos

Python 3.9+ (solo librería estándar; sin dependencias externas). Docker para el despliegue.
