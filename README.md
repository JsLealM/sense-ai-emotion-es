# SENSE AI / AuraVoice — Analizador Emocional de Voz en Tiempo Real

<div align="center">

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NVIDIA CUDA 12.4](https://img.shields.io/badge/NVIDIA_CUDA-12.4-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-emotion2vec-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/emotion2vec/emotion2vec_plus_large)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

**Prototipo de Reconocimiento de Emociones en el Habla (SER) con Aceleración en Paralelo sobre Silicio NVIDIA CUDA**  
*Asignatura: Computación Paralela y Distribuida — 7mo Semestre (2026)*

</div>

---

## Tabla de Contenidos
- [SENSE AI / AuraVoice — Analizador Emocional de Voz en Tiempo Real](#sense-ai--auravoice--analizador-emocional-de-voz-en-tiempo-real)
  - [Tabla de Contenidos](#tabla-de-contenidos)
  - [1. Modelo Fundacional Hugging Face: emotion2vec+ Large](#1-modelo-fundacional-hugging-face-emotion2vec-large)
    - [¿Por Que se Eligio Este Modelo?](#por-que-se-eligio-este-modelo)
    - [Restricciones Tecnicas y Solucion de Arquitectura](#restricciones-tecnicas-y-solucion-de-arquitectura)
  - [2. Descripcion de la Aplicacion y Problema que Resuelve](#2-descripcion-de-la-aplicacion-y-problema-que-resuelve)
    - [¿Que Problema Resuelve?](#que-problema-resuelve)
    - [¿Que Hace AuraVoice Engine?](#que-hace-auravoice-engine)
  - [3. Pruebas y Funcionamiento en Vivo](#3-pruebas-y-funcionamiento-en-vivo)
    - [Resumen de Pruebas con Audios Reales](#resumen-de-pruebas-con-audios-reales)
  - [4. Aceleracion y Funcionamiento de NVIDIA CUDA](#4-aceleracion-y-funcionamiento-de-nvidia-cuda)
  - [5. Arquitectura del Sistema](#5-arquitectura-del-sistema)
  - [6. Instalacion y Puesta en Marcha](#6-instalacion-y-puesta-en-marcha)
    - [Prerrequisitos](#prerrequisitos)
    - [Paso a Paso con `uv`](#paso-a-paso-con-uv)
    - [Streaming Continuo en Tiempo Real (WebSocket)](#streaming-continuo-en-tiempo-real-websocket)
  - [7. Contribuidores y Licencia](#7-contribuidores-y-licencia)
    - [Integrantes del Grupo](#integrantes-del-grupo)
    - [Licencias](#licencias)

---

## 1. Modelo Fundacional Hugging Face: emotion2vec+ Large

* **Identificador Oficial:** [`emotion2vec/emotion2vec_plus_large`](https://huggingface.co/emotion2vec/emotion2vec_plus_large)
* **Entidad Desarrolladora (*Issued By*):** **Alibaba Group / FunASR Team** (Interspeech 2024 SOTA).
* **Backbone Arquitectural:** **Data2Vec 2.0 Large** (Transformer contextual de auto-atención profunda).
* **Parametros Totales:** **~315.0 Millones de parametros** (315.0M params verificados fisicamente en VRAM).
* **Licencia de los Pesos:** Apache 2.0 / Open Source.

### ¿Por Que se Eligio Este Modelo?
1. **Acustico Puro (Cero Trampas Multimodales):** No recurre a la transcripcion textual a palabras (ASR con Whisper + LLMs de texto). Evalua de forma directa la **senal fisica y bioacustica de la voz** (frecuencia fundamental F0, entonacion, formantes espectrales, armonicos y dinamica temporal).
2. **Representaciones Latentes Universales:** Entrenado mediante aprendizaje autosupervisado sobre mas de 37.000 horas de habla, extrayendo representaciones acusticas robustas invariantes a dialectos, idiolectos y entornos acusticos.
3. **9 Clases Emocionales Canonicas:** Cobertura amplia de estados afectivos: *Alegria, Tristeza, Enojo, Miedo, Disgusto, Neutral, Sorpresa, Otro* y *Neutro / Desc.*

### Restricciones Tecnicas y Solucion de Arquitectura
* **Complejidad Cuadratica en Atencion Transformer:** En audios de mas de 15 segundos, la atencion $O(T^2)$ genera matrices masivas que provocaban desbordamiento de memoria en GPUs de 4 GB (`CUDA Out of Memory`).
* **Segmentacion Deslizante O(1) en Memoria:** Se diseno una segmentacion adaptativa en ventanas uniformes (3.0s a 5.0s con solapamiento) que acota el consumo de VRAM en un maximo estricto de **~770 MiB**, permitiendo procesar audios de cualquier duracion sin saturar la GPU.
* **Carga 100% Offline:** El sistema apunta directamente a la copia local en disco (`~/.cache/huggingface/hub/models--emotion2vec--emotion2vec_plus_large`), garantizando arranque instantaneo e independencia total de servidores remotos.

---

## 2. Descripcion de la Aplicacion y Problema que Resuelve

### ¿Que Problema Resuelve?
La clasificacion tradicional de afecto y prosodia depende frecuentemente de llamadas a servicios en la nube (APIs propietarias), generando latencias inaceptables (> 1.5s), altos costos por invocacion y graves vulnerabilidades de privacidad al transmitir senales biometricas de voz a centros de datos ajenos.

### ¿Que Hace AuraVoice Engine?
* **Ingesta Acustica Flexible:** Soporta archivos de audio (`.wav`, `.mp3`, `.ogg`, `.m4a`, `.flac`) y grabacion en vivo desde el microfono local con normalizacion a 16.000 Hz Mono.
* **Procesamiento de Senal en Tiempo Real:** Extraccion de nivel RMS en dB, amplitud pico y descomposicion espectral de 128 Mel-Bands.
* **Espacio Circunflejo de Russell 2D:** Mapeo matematico continuo sobre los ejes de **Valencia** (placer/displacer) y **Activacion / Arousal** (calma/agitacion).
* **Consola Dual AuraTerminal:** Alternancia dinamica entre telemetria de silicio CUDA (VRAM, latencia, temperatura) y trazabilidad explicable (**XAI**) con cadena de razonamiento algoritmico paso a paso.

---

## 3. Pruebas y Funcionamiento en Vivo

### Resumen de Pruebas con Audios Reales

| Fichero de Prueba                    | Duracion | Estado Predominante | Confianza | Latencia CUDA | Diagnostico Acustico                                                              |
| :----------------------------------- | :------: | :-----------------: | :-------: | :-----------: | :-------------------------------------------------------------------------------- |
| **Audio 1 (`WhatsApp Audio...`)**    |  1.39 s  |      **Enojo**      | **98.4%** |    32.1 ms    | Tension glotal marcada y alta concentracion armonica superior.                    |
| **Audio 2 (`WhatsApp Ptt 59s...`)**  | 59.05 s  |     **Neutral**     | **94.2%** |   1220.0 ms   | Discurso conversacional prolongado con modulacion controlada. Peak VRAM: 770 MiB. |
| **Audio 3 (`WhatsApp Ptt 8.8s...`)** |  8.81 s  |    **Tristeza**     | **95.8%** |   347.2 ms    | Caida de formantes acusticos y ritmo prosodico atenuado.                          |

---

## 4. Aceleracion y Funcionamiento de NVIDIA CUDA

En el contexto de **Computacion Paralela**, el proyecto explota la arquitectura de hardware de la GPU NVIDIA GeForce RTX 3050 Laptop mediante las siguientes tecnicas:

```mermaid
flowchart TD
    AudioIn["Senal Acustica Cruda (16 kHz Mono)"] --> PreProc["Preprocesamiento y Normalizacion de Amplitud"]
    PreProc --> Chunking["Segmentacion Deslizante Adaptativa O(1) VRAM"]
    Chunking --> VRAMAlloc["Asignacion de Tensores en VRAM (cuda:0, FP16)"]
    VRAMAlloc --> D2VKernels["Kernels Transformer Data2Vec 2.0 (24 Capas)"]
    D2VKernels --> TensorCores["Multi-Head Self-Attention en Tensor Cores"]
    TensorCores --> CUDASync["Sincronizacion de Streams CUDA (torch.cuda.synchronize)"]
    CUDASync --> LatentEmbeds["Embeddings Temporales Latentes [B, T, 1024]"]
    LatentEmbeds --> SoftmaxOut["Agrupacion Temporal + Softmax (9 Clases)"]
    SoftmaxOut --> Dashboard["Renderizado UI + Telemetria AuraTerminal"]
```

1. **Paralelismo Masivo con Nucleos CUDA y Tensor Cores:** Las operaciones tensoriales del codificador Data2Vec 2.0 se distribuyen sobre los **2048 nucleos CUDA** de la GPU, reduciendo la latencia de inferencia en ordenes de magnitud frente al procesador central.
2. **Gestion de Memoria Acotada ($O(1)$ VRAM):** En lugar de cargar la senal completa de audios extensos en una sola matriz cuadratica de atencion, el motor subdivide el flujo acustico y ejecuta pases eficientes en bucles tensoriales, manteniendo el consumo pico de VRAM estrictamente por debajo de **~770 MiB**.
3. **Sincronizacion Cientifica de Streams:** Medicion estricta del tiempo de inferencia con llamadas a `torch.cuda.synchronize()` antes y despues del kernel de ejecucion para eliminar la asincronia inherente entre CPU y GPU.

---

## 5. Arquitectura del Sistema

```text
sense-ai-emotion-es/
|
|-- app.py                      # Interfaz reactiva AuraVoice (Streamlit)
|-- requirements.txt            # Dependencias oficiales con indice PyTorch CUDA 12.4
|-- README.md                   # Documentacion tecnica del proyecto y del parcial
|-- MODEL_CARD_NOTES.md         # Ficha tecnica formal de emotion2vec+ Large
|-- LICENSE                     # Licencia de codigo abierto MIT
|
|-- src/                        # Modulos de logica y computacion paralela
|   |-- cuda_monitor.py         # Telemetria de GPU, VRAM y estado de streams
|   |-- emotion2vec_engine.py   # Motor de inferencia emotion2vec+ Large (9 clases)
|   |-- system_telemetry.py     # Monitoreo de CPU y trazas XAI de decision
|   |-- audio_telemetry.py      # Extraccion de RMS, pico dB y Mel-Bands
|   |-- audio_processor.py      # Conversion de formatos y normalizacion 16kHz
|   |-- result_parser.py        # Modelo circunflejo de Russell 2D y 9 clases
|   |-- evaluator.py            # Metricas supervisadas (Accuracy, F1, Matriz 9x9) - sin uso en la UI
|   |-- websocket_server.py     # Servidor WebSocket asincrono de streaming continuo (ventana 3 s / paso 1 s)
|   |-- streaming_ui.py         # Pestana de streaming: cliente JS (AudioWorklet + WebSocket) y panel en vivo
|   |-- diarization_engine.py   # Motor de diarizacion acustica
|
|-- scripts/                    # Scripts de soporte y puesta en marcha
|   `-- download_model.py       # Descarga y precarga de pesos para nuevos entornos (WSL)
|
|-- tests/                      # Banco de pruebas de terminal
|   |-- test_cuda.py            # Test de estres tensorial y sincronizacion CUDA
|   |-- test_inference.py       # Test de inferencia real de emotion2vec+ Large
|
|-- samples/                    # Audios de prueba en espanol
|   |-- sample_es.wav           # Muestra acustica de referencia
|
`-- .streamlit/
    `-- config.toml             # Configuracion de logging y supresion de observadores
```

---

## 6. Instalacion y Puesta en Marcha

### Prerrequisitos
* Sistema Operativo: Linux (nativo), WSL2 o Windows 10/11.
* GPU NVIDIA con soporte CUDA (RTX 3050 Laptop o superior).
* Controladores NVIDIA instalados (Driver 535+).
* FFmpeg disponible en el PATH del sistema.

### Paso a Paso con `uv`
```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/sense-ai-emotion-es.git
cd sense-ai-emotion-es

# 2. Crear entorno virtual con Python 3.10
uv venv --python 3.10 .venv

# 3. Activar el entorno
source .venv/bin/activate       # Linux / WSL2 / macOS

# 4. Instalar dependencias con soporte CUDA 12.4
uv pip install -r requirements.txt

# 5. Precargar y verificar los pesos del modelo (Solo la primera vez en un equipo nuevo / WSL)
python scripts/download_model.py

# 6. Ejecutar la aplicacion web
streamlit run app.py
```

La interfaz queda accesible en `http://localhost:8501`.

### Streaming Continuo en Tiempo Real (WebSocket)

La pestana **Streaming en Tiempo Real** analiza la voz del microfono de forma continua. Necesita un
segundo proceso: un servidor WebSocket asincrono que mantiene el modelo cargado en la GPU.

```bash
# Terminal 1: servidor de inferencia (carga emotion2vec+ Large en cuda:0 y escucha en ws://0.0.0.0:8765)
source .venv/bin/activate
python src/websocket_server.py

# Terminal 2: interfaz web
source .venv/bin/activate
streamlit run app.py
```

Flujo del pipeline:

1. **Navegador:** captura el microfono con un `AudioWorklet`, remuestrea a 16 kHz mono y envia tramas PCM Float32 de 250 ms por WebSocket.
2. **Servidor (`src/websocket_server.py`):** acumula el audio en un bufer circular acotado a 5 s. Cada 1.0 s de audio nuevo toma la ventana de los ultimos 3.0 s.
3. **Concurrencia:** la inferencia corre en un hilo aparte (`asyncio.to_thread`), asi el bucle de eventos sigue recibiendo audio mientras la GPU trabaja. Un `asyncio.Lock` serializa el acceso al unico modelo en GPU, y si la GPU esta ocupada se descartan las ventanas atrasadas para analizar siempre la mas reciente.
4. **Respuesta:** JSON con la emocion predominante, las 9 probabilidades, valencia/activacion (Russell 2D), latencia y VRAM. El panel se actualiza sin recargar Streamlit.

Tolerancia a fallos: si el servidor esta apagado, la pestana muestra instrucciones en lugar de un error; si se niega el microfono o se pierde la conexion, el panel lo indica y la app sigue funcionando. Las ventanas en silencio (RMS < 0.004) no se envian al modelo.

> El navegador solo permite usar el microfono en `https://` o en `http://localhost`. Abre la app como `http://localhost:8501`.

---

## 7. Contribuidores y Licencia

### Integrantes del Grupo
* **Integrante 1:** JsLealMJohan -> Sebastian Leal Mendez
* **Integrante 2:** Hades0131Andres -> David Benitez
* **Integrante 3:** jhayderstiven03-collab -> Jhayder Florez 

### Licencias
* **Codigo Fuente:** [MIT License](LICENSE).
* **Pesos del Modelo emotion2vec+ Large:** Publicado por Alibaba Group / FunASR bajo licencia **Apache 2.0**.
