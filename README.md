# Evaluación de desempeño del modelo Mistral 3 3B en inferencia local

Proyecto del curso de **Lengiajes de 4ta Generación**: suite de 10 prompts de
prueba para evaluar el modelo **ministral-3:3b** ejecutándose de forma
local mediante Ollama, con medición de TTFT (tiempo al primer token) y
velocidad de generación (tokens/s), ejecutado de forma nativa y
contenerizado con Docker.

## 1. Requisitos

- Windows 10/11 (probado en Windows 11), 8 GB de RAM
- [Python 3.12](https://www.python.org/downloads/) (o superior)
- [Git](https://git-scm.com/download/win)
- [Ollama para Windows](https://ollama.com/download/windows)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (para la ejecución contenerizada)
- ~5 GB de espacio en disco (modelo + herramientas)

## 2. Descarga del modelo

Luego de instalar Ollama, descargar el modelo (≈2 GB):

    ollama pull ministral-3:3b

Verificar:

    ollama list

## 3. Instalación del proyecto

    git clone https://github.com/lcedeno1975/proyecto-mistral.git
    cd proyecto-mistral
    pip install -r requirements.txt

## 4. Ejecución nativa

Iniciar Ollama y ejecutar la suite:

    python src/inference2.py --suite prompts/suite.json --output corrida1

O un prompt único de prueba:

    python src/inference2.py --prompt "dime la dirección de la sede central de la Universidad Central, ubicada en Costa Rica"

### Configuración de Ollama para Docker

Para que el contenedor pueda comunicarse con Ollama, este debe
escuchar en todas las interfaces (una sola vez):

    setx OLLAMA_HOST "0.0.0.0"

…y luego reiniciar Ollama (Quit desde la bandeja y volver a abrir).

## 5. Ejecución con Docker

Construir la imagen (el modelo NO se copia en la imagen; se consume
en tiempo de ejecución desde el Ollama del anfitrión):

    docker build -t mistral-inference:v2 .

Ejecutar la suite de 3 prompts dentro del contenedor:

    docker run --rm -e OLLAMA_URL=http://host.docker.internal:11434/api/generate -v %CD%\results:/app/results mistral-inference:v2 --suite prompts/suite2.json --output docker_corrida1

> Nota: `host.docker.internal` es la dirección con la que un
> contenedor alcanza a la máquina anfitrión. Los resultados se
> montan como volumen en `results/` del proyecto.

## 6. Estructura del repositorio

    README.md            esta guía
    requirements.txt     dependencias de Python (requests)
    Dockerfile           definición de la imagen del entorno de inferencia
    .dockerignore        archivos excluidos del contexto de build
    src/inference.py     script de inferencia (suite, métricas, errores)
    src/inference2.py    script de inferencia con corrección a la captura del TTFT(suite, métricas, errores)
    prompts/suite.json  suite de 10 prompts por capacidad
    prompts/suite2.json  suite de 3 prompts por capacidad
    results/             salidas JSON de las corridas (nativa y Docker)
    docs/                capturas y evidencias

## 7. Ejemplo de salida

    [GEN-01] (generacion_texto) Ejecutando: Escribe una publicación breve...
          -> TTFT 2.018s   3.46 tokens/s | 206 tokens
    ...
    ============================================================
    Corrida finalizada: 10 prompts OK, 0 con error.
    Resultados guardados en: results\corrida1_20260101_120000.json

Y el JSON de resultados conserva la estructura de la suite, con el
bloque `resultado` por prompt: salida, tokens_generados, ttft_seg,
tokens_por_segundo, temperatura, estado, error_detalle y fecha.

## 8. Problemas conocidos y soluciones
 | Problema | Solución |
 |---|---|
 | `failed to connect to the docker API` | Iniciar Docker Desktop y esperar "Engine running" |
 | `pull access denied` al `docker run` | Ejecutar antes `docker build -t mistral-inference:v2 .` |
 | Contenedor no conecta con Ollama | `setx OLLAMA_HOST "0.0.0.0"` + reiniciar Ollama |
 | `${PWD}` no se expande en PowerShell | Usar la ruta absoluta en el parámetro `-v` |

## 9. Licencia / uso

Proyecto académico para fines educativos.
