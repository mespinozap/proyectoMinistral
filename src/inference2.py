"""
inference.py — Script propio de inferencia para el proyecto Mistral AI.

Recorre el archivo JSON de la suite (prompts con estructura: id, categoria,
prompt, esperado, resultado, calificacion, comentario), ejecuta cada prompt
contra la API local de Ollama, registra salidas, métricas (TTFT y tokens/s)
y errores por prompt, y guarda en results/ un archivo con la MISMA estructura.

Uso:
  python src/inference.py --suite prompts/suite.json --output corrida1
  python src/inference.py --suite prompts/suite.json --temperature 0.2 --output corrida_temp02
  python src/inference.py --prompt "Escribe un haiku" --temperature 0.9
"""

import argparse          # parseo de argumentos de línea de comandos
import copy             # copia profunda: el JSON original queda intacto
import json             # lectura de la suite y escritura de resultados
import time             # medición de tiempos
from datetime import datetime   # marcas de tiempo de ejecución
from pathlib import Path        # manejo portable de rutas relativas
import os
import requests         # cliente HTTP hacia la API de Ollama

# API local de Ollama: localhost, sin credenciales ni rutas absolutas
#OLLAMA_URL = "http://localhost:11434/api/generate"

#OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434/api/generate")


def consultar_ollama(modelo, prompt, temperatura, max_tokens):
    """Envía un prompt a Ollama en modo streaming.

    Devuelve un diccionario con la misma forma del bloque 'resultado'
    de la suite: salida, tokens_generados, ttft_seg, tokens_por_segundo,
    temperatura, estado, error_detalle y fecha_ejecucion.

    Métricas:
      - TTFT: load_duration + prompt_eval_duration reportados por Ollama
        (tiempo real hasta el primer token generado, incluye la carga
        del modelo si este no estaba en memoria).
      - tokens/s: eval_count / eval_duration (cifras exactas de Ollama).

    Los errores se capturan y se devuelven dentro de la estructura, de modo
    que un prompt fallido no interrumpe la corrida de la suite completa.
    """
    # Estructura de resultado (idéntica al bloque 'resultado' del suite.json)
    resultado = {
        "salida": None,
        "tokens_generados": None,
        "ttft_seg": None,
        "tokens_por_segundo": None,
        "temperatura": temperatura,
        "estado": "ok",
        "error_detalle": None,
        "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"),
    }

    # --- 1. Envío de la petición (con captura de errores de conexión/API) ---
    t_inicio = time.perf_counter()   # reloj ANTES de enviar la petición
    try:
        respuesta = requests.post(
            OLLAMA_URL,
            json={
                "model": modelo,
                "prompt": prompt,
                "stream": True,                  # streaming: permite medir TTFT
                "options": {
                    "temperature": temperatura,   # temperatura de muestreo
                    "num_predict": max_tokens,    # máximo de tokens a generar
                },
            },
            stream=True,     # consume la respuesta fragmento a fragmento
            timeout=300,     # margen amplio: en CPU la generación es lenta
        )
        respuesta.raise_for_status()   # detecta 404 (modelo no instalado), 500, etc.

    except requests.exceptions.ConnectionError:
        # Servidor caído: Ollama no corre o el puerto no responde
        resultado["estado"] = "error"
        resultado["error_detalle"] = ("No se pudo conectar con Ollama en "
                                      "localhost:11434. Verifique que el "
                                      "servicio esté en ejecución (bandeja del sistema).")
        return resultado
    except requests.exceptions.HTTPError as e:
        resultado["estado"] = "error"
        if "404" in str(e):
            # Ollama devuelve 404 cuando el modelo no existe
            resultado["error_detalle"] = (f"El modelo '{modelo}' no está instalado. "
                                          f"Ejecute: ollama pull {modelo}")
        else:
            resultado["error_detalle"] = f"Error HTTP al consultar Ollama: {e}"
        return resultado
    except requests.exceptions.Timeout:
        resultado["estado"] = "error"
        resultado["error_detalle"] = "La petición excedió el tiempo de espera (300 s)."
        return resultado

    # --- 2. Procesamiento del streaming, fragmento a fragmento ---
    ttft_streaming = None      # tiempo de reloj hasta el primer fragmento recibido
    fragmentos = []
    metricas_finales = {}      # aquí queda el resumen final que envía Ollama

    try:
        for linea in respuesta.iter_lines():
            if not linea:
                continue
            fragmento = json.loads(linea.decode("utf-8"))
            fragmentos.append(fragmento.get("response", ""))
            if ttft_streaming is None and fragmento.get("response"):
                # primer token recibido: tiempo desde el envío de la petición
                ttft_streaming = time.perf_counter() - t_inicio
            if fragmento.get("done"):
                # el último fragmento trae todas las métricas de Ollama
                metricas_finales = fragmento
    except json.JSONDecodeError:
        resultado["estado"] = "error"
        resultado["error_detalle"] = "Respuesta de Ollama con formato inesperado (JSON inválido)."
        return resultado

    # --- 3. Verificación de que la generación terminó bien ---
    if not metricas_finales:
        # El flujo terminó sin la marca 'done': se registra como error suave
        resultado["salida"] = "".join(fragmentos)
        resultado["tokens_generados"] = len(fragmentos)
        resultado["ttft_seg"] = round(ttft_streaming or 0, 3)
        resultado["estado"] = "error"
        resultado["error_detalle"] = "El flujo terminó de forma inesperada (sin marca 'done')."
        return resultado

    # --- 4. Métricas calculadas con los datos exactos que reporta Ollama ---
    # Ollama reporta las duraciones en nanosegundos (ns); se convierten a segundos.
    carga = metricas_finales.get("load_duration", 0) / 1e9          # carga del modelo en RAM
    eval_prompt = metricas_finales.get("prompt_eval_duration", 0) / 1e9  # lectura del prompt

    resultado["salida"] = "".join(fragmentos)
    resultado["tokens_generados"] = metricas_finales.get("eval_count", len(fragmentos))

    # TTFT real: lo que tarda el sistema en producir el primer token de salida.
    # Incluye la carga del modelo si no estaba en memoria (primera consulta);
    # en consultas siguientes 'carga' es ~0 y solo cuenta la evaluación del prompt.
    resultado["ttft_seg"] = round(carga + eval_prompt, 3)

    # Velocidad de generación: tokens generados / tiempo de generación
    duracion = metricas_finales.get("eval_duration", 0) / 1e9
    resultado["tokens_por_segundo"] = (round(resultado["tokens_generados"] / duracion, 2)
                                       if duracion > 0 else None)
    return resultado


def ejecutar_suite(modelo, ruta_suite, temperatura, max_tokens, salida):
    """Recorre la suite JSON, ejecuta cada prompt y guarda los resultados
    en results/ conservando la estructura original del archivo."""
    # --- 1. Carga y validación del archivo de la suite ---
    ruta_suite = Path(ruta_suite)
    if not ruta_suite.exists():
        raise FileNotFoundError(f"ERROR: No se encontró el archivo de suite: {ruta_suite}")

    try:
        with open(ruta_suite, encoding="utf-8") as f:
            suite = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"ERROR: {ruta_suite} no es un JSON válido. Revise la sintaxis.")

    if "prompts" not in suite or not suite["prompts"]:
        raise ValueError(f"ERROR: {ruta_suite} no contiene la lista 'prompts' o está vacía.")

    # --- 2. Copia profunda: el archivo original en prompts/ queda intacto ---
    suite_resultado = copy.deepcopy(suite)
    suite_resultado["model"] = modelo
    suite_resultado["temperatura"] = temperatura
    suite_resultado["max_tokens"] = max_tokens
    suite_resultado["fecha_corrida"] = datetime.now().isoformat(timespec="seconds")

    # --- 3. Recorrido de los prompts y ejecución uno por uno ---
    total_ok, total_error = 0, 0
    for item in suite_resultado["prompts"]:
        id_prompt = item.get("id", "SIN-ID")
        print(f"[{id_prompt}] ({item.get('categoria', '?')}) "
              f"Ejecutando: {item['prompt'][:60]}...")

        # Ejecución contra la API; el bloque 'resultado' del JSON se sobrescribe
        item["resultado"] = consultar_ollama(modelo, item["prompt"],
                                             temperatura, max_tokens)

        if item["resultado"]["estado"] == "ok":
            total_ok += 1
            print(f"      -> TTFT {item['resultado']['ttft_seg']}s | "
                  f"{item['resultado']['tokens_por_segundo']} tokens/s | "
                  f"{item['resultado']['tokens_generados']} tokens\n")
        else:
            total_error += 1
            print(f"      -> ERROR: {item['resultado']['error_detalle']}\n")

    # --- 4. Escritura del archivo de resultados (misma estructura que la suite) ---
    carpeta = Path("results")
    carpeta.mkdir(exist_ok=True)   # crea results/ si no existe
    nombre = f"{salida}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    ruta_salida = carpeta / nombre
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(suite_resultado, f, ensure_ascii=False, indent=2)

    # --- 5. Resumen en consola ---
    print("=" * 60)
    print(f"Corrida finalizada: {total_ok} prompts OK, {total_error} con error.")
    print(f"Resultados guardados en: {ruta_salida}")
    return ruta_salida


def main():
    """Parseo de argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Script de inferencia local con Ollama")
    parser.add_argument("--model", default="ministral-3:3b",
                        help="Modelo de Ollama a usar (default: ministral-3:3b)")
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--prompt", help="Prompt único para ejecutar")
    grupo.add_argument("--suite", help="Ruta al archivo JSON de la suite")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperatura de muestreo (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Máximo de tokens a generar (default: 256)")
    parser.add_argument("--output", default="corrida",
                        help="Nombre base del archivo de resultados")
    args = parser.parse_args()

    try:
        if args.prompt:
            # --- Modo prompt único: muestra salida y métricas en consola ---
            r = consultar_ollama(args.model, args.prompt,
                                 args.temperature, args.max_tokens)
            if r["estado"] == "ok":
                print(f"\nSalida:\n{r['salida']}\n")
                print(f"TTFT: {r['ttft_seg']}s | Tokens: {r['tokens_generados']} "
                      f"| {r['tokens_por_segundo']} tokens/s")
            else:
                print(f"ERROR: {r['error_detalle']}")
                raise SystemExit(1)
        else:
            # --- Modo suite completa ---
            ejecutar_suite(args.model, args.suite, args.temperature,
                           args.max_tokens, args.output)
    except (FileNotFoundError, ValueError) as e:
        # Errores de archivo/estructura: mensaje claro y salida con código 1
        print(e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()