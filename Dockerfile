# Imagen del meta-agente de aprendizaje.
#
# No hay `pip install` y no es un descuido: el proyecto usa solo librería estándar (incluido el
# cliente del LLM, que habla HTTP con `urllib`). Consecuencias prácticas: la imagen se construye
# sin red hacia PyPI, arranca en segundos y no tiene un árbol de dependencias que se rompa entre
# versiones. Si algún día entra una dependencia, entra por el dominio — nunca por el núcleo.
FROM python:3.11-slim

# Salida sin buffer: los logs y las meta-alertas aparecen en `docker logs` al instante, no
# cuando se llena el buffer. En un agente que se observa por consola esto importa.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Solo el código que el agente necesita para correr. Los experimentos, los datos de ejemplo y
# el dominio de referencia no viajan a la VPS.
COPY core/ /app/core/
COPY domains/__init__.py /app/domains/
COPY domains/agent_behavior/ /app/domains/agent_behavior/
COPY main.py /app/main.py
# Exposicion de solo lectura del estado propio (ver state_server.py). Viaja porque es lo que
# alimenta al dashboard; sigue siendo stdlib pura, asi que no agrega un `pip install`.
COPY state_server.py /app/state_server.py

# Volumen propio del meta-agente: acá viven el conocimiento aprendido y las meta-alertas.
# El directorio del agente OBSERVADO se monta aparte y de solo lectura (ver compose).
RUN mkdir -p /state
VOLUME ["/state"]

# Daemon por defecto: procesa el histórico de trazas y sigue en vivo.
ENTRYPOINT ["python", "main.py"]
