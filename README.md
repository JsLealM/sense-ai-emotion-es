# sense-ai-emotion-es
## SENSE AI — Analizador Emocional de Voz en Español
> **Prototipo Funcional de Reconocimiento de Emociones en el Habla (SER) con Aceleración NVIDIA CUDA**  
> Proyecto Universitario: *Computación Paralela / Arquitecturas Paralelas* (7mo Semestre)  
> **Nombre del Repositorio:** `sense-ai-emotion-es`

---

## Enlaces Oficiales del Modelo

* **Hugging Face Model Hub:** [https://huggingface.co/UMUTeam/w2v-bert-emotion-es](https://huggingface.co/UMUTeam/w2v-bert-emotion-es)
* **Entidad Desarrolladora:** [UMUTeam (Grupo de Lenguaje Natural, Universidad de Murcia)](https://huggingface.co/UMUTeam)
* **Repositorio de Referencia (Toolkit):** [NLP-UMUTeam/umuteam-speech-emotion](https://github.com/NLP-UMUTeam/umuteam-speech-emotion)
* **Licencia de los Pesos:** Apache 2.0 / Código MIT

---

## 1. Descripción del Proyecto

Este proyecto implementa y evalúa un prototipo funcional de **Reconocimiento de Emociones en el Habla (Speech Emotion Recognition - SER)** entrenado y ajustado específicamente para el **idioma español**. 

A diferencia de modelos multilingües generalistas o centrados en transcripción (ASR) que carecen de soporte prosódico nativo para el español, este prototipo integra el modelo fundacional **W2V-BERT Emotion ES** (`UMUTeam/w2v-bert-emotion-es`) acelerado por hardware mediante **NVIDIA CUDA** para clasificar la acústica y entonación de la voz en seis estados emocionales fundamentales.

### Categorías Emocionales Soportadas:
El modelo infiere una distribución de probabilidad Softmax sobre las siguientes 6 clases:

| ID | Etiqueta Original (Config) | Etiqueta en Español | Significado / Manifestación Acústica |
| :---: | :---: | :---: | :--- |
| `0` | `anger` | **Enojo / Molestia** | Elevada energía espectral, mayor intensidad sonora y pitch elevado. |
| `1` | `disgust` | **Disgusto** | Inflexiones vocales de aversión o rechazo marcado. |
| `2` | `fear` | **Miedo** | Modulación inestable, temblor tonal y energía contenida. |
| `3` | `joy` | **Alegría** | Entonación melódica ascendente, ritmo ágil y energía positiva. |
| `4` | `neutral` | **Neutral** | Entonación plana y uniforme, ritmo pausado y balance armónico. |
| `5` | `sadness` | **Tristeza** | Pitch decaído, tempo lento y atenuación de formantes superiores. |

> [!IMPORTANT]
> **Descargo Ético Obligatorio**:  
> El sistema clasifica patrones bioacústicos y prosódicos presentes en la señal sonora de la voz (frecuencia fundamental F0, energía, formantes). **Bajo ninguna circunstancia** debe interpretarse como un diagnóstico psicológico concluyente ni una evaluación psiquiátrica del estado mental interno de una persona.

---

## 2. Ficha Técnica y Arquitectura del Modelo

* **Arquitectura:** `CustomAudioClassification` basada en **Wav2Vec2-BERT** (Conformer + Self-Attention de 24 capas).
* **Parámetros Totales:** **~581.877.062 parámetros** (~582M params verificados físicamente en VRAM).
* **Tamaño de los Pesos en Disco:** `model.safetensors` (~2.33 GB en precisión completa FP32).
* **Preprocesamiento Acústico:** Extractor `SeamlessM4TFeatureExtractor` que calcula bancos de filtros Mel logarítmicos de **80 canales** muestreados a **16.000 Hz (16 kHz, canal mono)**.
* **Cabezal de Clasificación (`ClassificationHead`):** Proyector lineal (1024 -> 768) + Capa Densa (768 -> 768) con activación Tanh + Proyección de salida (768 -> 6 clases).

---

## 3. Aceleración con NVIDIA CUDA en GPUs de 4 GB (RTX 3050 Laptop)

Un modelo de 582M de parámetros en precisión completa (FP32) puede copar más de 3.5 GB al sumar los tensores de activaciones intermedias. Para garantizar una ejecución estable y veloz:

1. **Inferencia en Media Precisión (`torch.float16`)**:
   - Reduce la huella del modelo en VRAM a solo **~1.12 GB**.
   - Deja más de **2.6 GB de VRAM libres** para cálculos tensoriales.
2. **Sincronización de Streams (`torch.cuda.synchronize()`)**:
   - Medición precisa de latencia en milisegundos evitando el sesgo de las llamadas asíncronas de la CPU.
3. **Manejo Preventivo de Memoria**:
   - Captura de excepciones `torch.cuda.OutOfMemoryError` con vaciado inmediato de fragmentación (`torch.cuda.empty_cache()`).
4. **Modo Alternativo CPU**:
   - Permite conmutar a CPU para verificar en vivo el factor de aceleración (*speedup*) frente a la GPU.

---

## 4. Instalación del Entorno Paso a Paso

El proyecto está diseñado para funcionar en **Linux nativo**, **WSL2 (Windows Subsystem for Linux)** y **Windows nativo**.

---

### Opción A: En Linux Nativo o WSL2 (Ubuntu / Debian / CachyOS / Arch)

#### 1. Prerrequisitos de sistema (FFmpeg)
- **Arch / CachyOS:** `sudo pacman -S ffmpeg`
- **Ubuntu / Debian / WSL2:** `sudo apt update && sudo apt install -y ffmpeg`

> **Nota para WSL2:** Si usas WSL2, asegúrate de tener instalado el controlador NVIDIA más reciente en Windows. Windows pasa automáticamente el soporte de CUDA a WSL2; **no instales drivers de GPU dentro de WSL2**.

#### 2. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/sense-ai-emotion-es.git
cd sense-ai-emotion-es
```

#### 3. Instalar `uv` y crear entorno Python 3.10
```bash
# Instalar uv en el espacio de usuario (sin requerir sudo)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Crear entorno con Python 3.10
uv venv --python 3.10 .venv
```

#### 4. Activar el entorno virtual
- **En Fish Shell (CachyOS por defecto):**
  ```fish
  source .venv/bin/activate.fish
  ```
- **En Bash o Zsh (Ubuntu / WSL2 por defecto):**
  ```bash
  source .venv/bin/activate
  ```

#### 5. Instalar dependencias con CUDA 12.4
```bash
uv pip install -r requirements.txt
```

---

### Opción B: En Windows Nativo (PowerShell)

#### 1. Prerrequisitos en Windows
- **Instalar FFmpeg:** Abre PowerShell como Administrador y ejecuta:
  ```powershell
  winget install Gyan.FFmpeg
  ```
  *(O descarga el binario de ffmpeg.org y agrégalo al PATH del sistema).*

#### 2. Instalar `uv` en Windows
En PowerShell (terminal normal):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 3. Clonar y crear entorno virtual
```powershell
git clone https://github.com/TU_USUARIO/sense-ai-emotion-es.git
cd sense-ai-emotion-es

# Crear entorno aislado con Python 3.10
uv venv --python 3.10 .venv
```

#### 4. Activar entorno en Windows
- **En PowerShell:**
  ```powershell
  # Si PowerShell bloquea scripts, habilitar para la sesión actual:
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

  # Activar entorno:
  .venv\Scripts\Activate.ps1
  ```
- **En Símbolo del Sistema (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

#### 5. Instalar dependencias con CUDA
```powershell
uv pip install -r requirements.txt
```

---

## 5. Verificación Técnica en Terminal

### A. Prueba de Estrés y Telemetría CUDA
Comprueba que PyTorch detecta la GPU física y ejecuta operaciones tensoriales en VRAM:
```bash
python tests/test_cuda.py
```
*Salida esperada:* Detección de la RTX 3050 Laptop, multiplicación de matrices 4000x4000 en `cuda:0` en ~116 ms y reporte de VRAM.

### B. Prueba de Inferencia Real sobre Audio en Español
Ejecuta la clasificación emocional en terminal sobre una grabación de muestra:
```bash
python tests/test_inference.py samples/sample_es.wav
```
*Salida esperada:*
- Verificación física de los **581.877.062 parámetros** en `cuda:0` en formato `float16`.
- Predicción predominante (ej. Neutral: 99.94%).
- Tabla de probabilidades de las 6 categorías.
- Latencia en milisegundos (~497 ms), RTF (~0.20) y VRAM pico (~1122 MB).

---

## 6. Ejecución de la Interfaz Web (Streamlit)

Inicia la aplicación web interactiva:
```bash
streamlit run app.py
```
Se abrirá automáticamente en tu navegador en:  
`http://localhost:8501`

### Pestañas de la Aplicación AuraVoice:
1. **Pestaña 1: Sesión en Vivo**: Dashboard principal AuraVoice con captura y streaming de audio, visualización de ondas espectrales (RMS/Pico dB), tarjeta de **Estado Emocional Latente** (modelo circunflejo Valence/Arousal, desglose de Softmax y métricas acústicas), **AuraTerminal** con telemetría de CPU y GPU en tiempo real, e **Historial de Diarización** (línea temporal de Alex y Dra. Silva).
2. **Pestaña 2: Historial & Diarización**: Monitoreo de cuota de habla y simulador de turnos conversacionales dinámicos preparado para streaming multi-hablante.
3. **Pestaña 3: Métricas de Inferencia**: Benchmark comparativo CUDA vs CPU y evaluación supervisada formal (Matriz de Confusión 6x6, Accuracy, Precision, Recall y F1-Score).
4. **Pestaña 4: Configuración de Modelo**: Ficha técnica, ajuste de ventanas deslizantes, selector de hardware y optimización FP16.

---

## 7. Estructura Modular del Proyecto

```text
sense-ai-emotion-es/
│
├── app.py                  # Interfaz web interactiva AuraVoice (Streamlit)
├── requirements.txt        # Dependencias oficiales con índice CUDA 12.4
├── README.md               # Documentación general y guía de instalación
├── MODEL_CARD_NOTES.md     # Ficha técnica académica y funcionamiento de CUDA
├── .gitignore              # Reglas de exclusión de caché, entorno y pesos
│
├── src/                    # Módulos modulares del sistema
│   ├── cuda_monitor.py     # Telemetría de GPU, VRAM y verificación física de tensores
│   ├── system_telemetry.py # Monitoreo de CPU, hilos y generador de logs AuraTerminal
│   ├── audio_telemetry.py  # Extracción de ondas visuales, nivel RMS, pico dB y F0
│   ├── diarization_engine.py# Seguimiento temporal multi-hablante (Alex y Dra. Silva)
│   ├── audio_processor.py  # Conversión y metadatos de audio (16kHz mono)
│   ├── model_loader.py     # Definición CustomAudioClassification y carga FP16 en VRAM
│   ├── inference.py        # Orquestador con sincronización torch.cuda.synchronize()
│   ├── result_parser.py    # Modelo circunflejo (Valence/Arousal) y métricas Softmax
│   └── evaluator.py        # Matriz de confusión y métricas supervisadas (Scikit-Learn)
│
├── tests/                  # Pruebas automatizadas de terminal
│   ├── test_cuda.py        # Verificación de hardware tensorial en CUDA
│   └── test_inference.py   # Test de inferencia real de W2V-BERT en GPU
│
├── samples/                # Audios de muestra en español
│   ├── README.md           # Explicación del banco de pruebas
│   └── sample_es.wav       # Audio en español de prueba ("República de Colombia")
│
└── assets/                 # Recursos multimedia y evidencias
    └── screenshots/        # Capturas de pantalla para la sustentación
```

---

## Colaboradores y Licencia
* **Proyecto Universitario:** Computación Paralela (2026).
* **Código del Prototipo:** Licencia MIT.
* **Modelo W2V-BERT Emotion ES:** Desarrollado por **UMUTeam** (Universidad de Murcia), Licencia Apache 2.0.
