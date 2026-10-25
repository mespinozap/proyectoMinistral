# Dockerfile — Entorno de inferencia del proyecto Mistral AI
# Contiene Python + dependencias + script. El modelo NO se copia:
# el script consulta el servidor Ollama del equipo anfitrión.

# Buena práctica 1: imagen base ligera y con versión fija
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Buena práctica 2: capas ordenadas (dependencias estables primero,
# código después) para aprovechar la caché de Docker
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/inference2.py src/inference2.py
COPY src/inference.py src/inference.py

COPY prompts/suite.json prompts/suite.json
COPY prompts/suite2.json prompts/suite2.json

# Buena práctica 3: usuario no root
RUN useradd --create-home appuser \
    && mkdir -p /app/results \
    && chown -R appuser:appuser /app/results
USER appuser

ENTRYPOINT ["python", "src/inference2.py"]

CMD ["--help"]