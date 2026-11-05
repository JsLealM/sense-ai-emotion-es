# FICHA TÉCNICA DEL MODELO: W2V-BERT Emotion ES (UMUTeam)

Documento técnico elaborado para la asignatura de **Computación Paralela / Arquitecturas Paralelas** (Universidad).

---

## 1. Identificación General del Modelo

| Parámetro | Detalle |
| :--- | :--- |
| **Nombre Oficial** | **W2V-BERT Emotion ES** (`UMUTeam/w2v-bert-emotion-es`) |
| **Entidad Desarrolladora** | **UMUTeam** (Grupo de Lenguaje Natural y Ciencias de la Computación, **Universidad de Murcia**, España) |
| **Repositorio Hugging Face** | [https://huggingface.co/UMUTeam/w2v-bert-emotion-es](https://huggingface.co/UMUTeam/w2v-bert-emotion-es) |
| **Tarea Principal** | Clasificación de Audio / Reconocimiento de Emociones en el Habla (**Speech Emotion Recognition - SER**) |
| **Idioma Objetivo** | **Español (`es`)** |
| **Licencia** | **Apache 2.0 / Open Source** (Uso académico y comercial con atribución) |

---

## 2. Descripción y Problema que Resuelve

### ¿Qué problema resuelve?
El reconocimiento de emociones en la voz (SER) es uno de los desafíos más complejos en computación afectiva y procesamiento de señales de audio. Gran parte de los modelos fundacionales multilingües (como Whisper o SenseVoice) se centran en la transcripción textual (ASR) y carecen de sensibilidad o soporte nativo para el español en su faceta prosódica y tonal.

### Propuesta de W2V-BERT Emotion ES:
**W2V-BERT Emotion ES** fue entrenado y ajustado específicamente sobre grabaciones de habla en español para capturar la modulación tonal, la energía, el ritmo y la frecuencia fundamental (F0). A partir de una señal de audio cruda de la voz humana, el modelo infiere la distribución probabilística de seis estados emocionales fundamentales.

---

## 3. Arquitectura y Parámetros

* **Arquitectura Base**: **Wav2Vec2-BERT (Conformer + Self-Attention)**.
* **Extractor de Características**: `SeamlessM4TFeatureExtractor` (calcula bancos de filtros Mel logarítmicos de **80 canales** muestreados a **16.000 Hz**).
* **Capas Ocultas**: 24 capas de atención Conformer con tamaño oculto `hidden_size=1024` y 16 cabezas de atención (`num_attention_heads=16`).
* **Dimensión Intermedia**: 4096 (`intermediate_size=4096`).
* **Número de Parámetros**: Aprox. **580 Millones de parámetros** (~580M params).
* **Tamaño de los Pesos en Disco**:
  - Precisión Completa (FP32, `model.safetensors`): **~2.33 GB**.
  - Media Precisión (FP16 en GPU): **~1.16 GB**.
* **Preprocesamiento Acústico**: Audio de entrada remuestreado a **16.000 Hz (16 kHz), canal mono**.
* **Capa de Salida**: Cabezal de clasificación de secuencia para audio con normalización Softmax sobre 6 clases.

---

## 4. Categorías Emocionales y Mapeo de Etiquetas

El modelo produce puntuaciones numéricas (logits) correspondientes a las siguientes **seis categorías**:

| ID | Etiqueta Original (Config) | Etiqueta en Español | Significado / Manifestación Acústica |
| :---: | :---: | :---: | :--- |
| `0` | `anger` | **Enojo / Molestia** | Alta energía, mayor volumen y tono de voz elevado. |
| `1` | `disgust` | **Disgusto** | Tono de aversión o desagrado pronunciado. |
| `2` | `fear` | **Miedo** | Tono tembloroso, variación rápida de tono y volumen moderado. |
| `3` | `joy` | **Alegría** | Entonación ascendente, ritmo dinámico y energía positiva. |
| `4` | `neutral` | **Neutral** | Entonación uniforme, ritmo pausado y balance armónico. |
| `5` | `sadness` | **Tristeza** | Tono bajo, ritmo pausado, caída de energía en formantes. |

> [!IMPORTANT]
> **Descargo Ético Obligatorio**:  
> El modelo clasifica patrones acústicos y prosódicos en la señal de audio. **Bajo ninguna circunstancia** debe interpretarse como una evaluación psicológica concluyente ni un diagnóstico médico del estado anímico o mental real de una persona.

---

## 5. Dataset de Entrenamiento y Métricas Reportadas

* **Entorno de Datos**: Entrenado por el equipo UMUTeam en corpus de habla en español (incluyendo datasets conversacionales y de actuación emocional controlada como MAV-ES y corpus en lengua castellana).
* **Métricas Típicas de Referencia**:
  - Unweighted Average Recall (UAR): ~68% - 74% (estado del arte en SER en español).
  - Accuracy ponderado: ~70% - 76% según las condiciones acústicas del entorno.

---

## 6. Aceleración con NVIDIA CUDA en GPU de 4 GB (RTX 3050 Laptop)

### Desafío de Memoria en GPU de 4 GB:
Un modelo de 580M de parámetros en precisión completa (FP32) ocupa ~2.4 GB solo en pesos fijos. Al sumar las activaciones intermedias de la red Conformer de 24 capas, la VRAM requerida puede rozar los 3.5 - 3.8 GB, incrementando el riesgo de desbordamiento (`CUDA Out of Memory`).

### Solución Implementada:
1. **Inferencia en Media Precisión (`torch.float16`)**:
   - Carga del modelo en GPU con `torch_dtype=torch.float16`.
   - Reduce el tamaño de los tensores de pesos a **~1.16 GB de VRAM**.
   - Deja más de **2.5 GB de VRAM libres** para buffers y activaciones.
2. **Sincronización de Streams (`torch.cuda.synchronize()`)**:
   - Medición precisa de latencia en milisegundos sin sesgos causados por la asincronía del host (CPU).
3. **Manejo de Errores Preventivo**:
   - En caso de excepción `torch.cuda.OutOfMemoryError`, se ejecuta `torch.cuda.empty_cache()` y se notifica al usuario con instrucciones de segmentación.
4. **Modo Alternativo CPU**:
   - Permite conmutar a CPU para propósitos de comparación y diagnóstico de rendimiento.

---

## 7. Limitaciones Técnicas Conocidas

1. **Ruido de Fondo**: Grabaciones con ruido ambiental elevado o música de fondo pueden desviar la clasificación hacia neutral o disgusto.
2. **Sarcasmo e Ironía**: Al tratarse de un modelo puramente acústico (audio-only), no analiza el contexto semántico del texto, por lo que frases irónicas pueden clasificarse con la emoción de la entonación literal.
3. **Variación Dialectal**: Aunque está optimizado para español general, ciertas inflexiones regionales marcadas pueden tener ligeras variaciones en la confianza de la predicción.
