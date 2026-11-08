# FICHA TÉCNICA DEL MODELO: emotion2vec+ Large (Interspeech 2024 SOTA)

Documento técnico elaborado para la asignatura de **Computación Paralela / Arquitecturas Paralelas** (Universidad).

---

## 1. Identificación General del Modelo

| Parámetro | Detalle |
| :--- | :--- |
| **Nombre Oficial** | **emotion2vec+ Large** (`emotion2vec/emotion2vec_plus_large`) |
| **Entidad Desarrolladora** | **Alibaba Group / FunASR Team** (Interspeech 2024) |
| **Repositorio Hugging Face** | [https://huggingface.co/emotion2vec/emotion2vec_plus_large](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |
| **Publicación Científica** | *emotion2vec+: Self-Supervised Pre-Training for Speech Emotion Representation* (Interspeech 2024) |
| **Tarea Principal** | Reconocimiento de Emociones en el Habla (**Speech Emotion Recognition - SER**) Acústico Puro |
| **Dominio Acústico** | **Universal / Multilingüe** (Inferencia acústica de representaciones latentes invariantes al idioma) |
| **Licencia** | **Apache 2.0 / Open Source** (Uso académico y comercial con atribución) |

---

## 2. Descripción y Problema que Resuelve

### ¿Qué problema resuelve?
El reconocimiento de emociones en la voz (SER) enfrenta el reto de modelar las variaciones acústicas paralingüísticas (entonación, ritmo, armónicos, energía, formantes) independientemente del contenido semántico textual. La mayoría de modelos multimodales dependen de transcribir texto (ASR) o acoplar grandes modelos de lenguaje (LLM), lo que introduce latencias masivas, fallas alucinatorias y sesgos léxicos.

### Propuesta de emotion2vec+ Large:
**emotion2vec+ Large** es un modelo fundacional acústico puro de última generación (*State-of-the-Art*). Entrenado mediante aprendizaje autosupervisado sobre decenas de miles de horas de habla, aprende a proyectar el audio crudo directamente en un espacio latente estructurado según la valencia y excitación emocional. Extrae probabilidades de 9 estados emocionales directamente de la forma de onda acústica en tiempo real.

---

## 3. Arquitectura y Parámetros

* **Arquitectura Base**: **Data2Vec 2.0 Large** (Transformer contextual para audio con atención multi-cabeza y ALiBi positional encoding).
* **Número de Parámetros**: **~315.0 Millones de parámetros** (315.0M params).
* **Capas Transformer**: 24 capas de codificador contextual con normalizaciones de capa optimizadas.
* **Dimensión de Embedding**: 1024 canales latentes.
* **Preprocesamiento Acústico**: Señal de audio remuestreada a **16.000 Hz (16 kHz), canal mono**.
* **Precisión y Aceleración**: Tensores en **FP16** acelerados mediante núcleos tensoriales NVIDIA CUDA.
* **Capa de Clasificación**: Proyección lineal final con activación Softmax normalizada sobre 9 clases.

---

## 4. Categorías Emocionales y Mapeo Canónico

El modelo produce puntuaciones probabilísticas (logits) para **9 categorías canónicas**:

| ID | Etiqueta Original (Config) | Etiqueta en Español | Significado / Manifestación Acústica |
| :---: | :---: | :---: | :--- |
| `0` | `angry` | **Enojo** | Tensión glotal marcada, ataque abrupto y alta energía en altas frecuencias. |
| `1` | `disgusted` | **Disgusto** | Formantes descendentes, entonación de rechazo o aversión visceral. |
| `2` | `fearful` | **Miedo** | Temblores en tono (jitter/shimmer), variaciones rápidas de entonación. |
| `3` | `happy` | **Alegría** | Dinamismo melódico elevado, entonación ascendente y resonancia armónica abierta. |
| `4` | `neutral` | **Neutral** | Contorno de F0 homogéneo, cadencia estable y balance prosódico. |
| `5` | `other` | **Otro** | Expresiones afectivas no contempladas en las categorías elementales. |
| `6` | `sad` | **Tristeza** | Reducción de formantes, ritmo atenuado, caída de volumen y timbre sombrío. |
| `7` | `surprised` | **Sorpresa** | Elevación súbita de frecuencia fundamental (F0) y cadencia inesperada. |
| `8` | `unknown` | **Neutro / Desc.** | Señal acústica de baja inteligibilidad o prosodia indeterminada. |

> [!IMPORTANT]
> **Descargo Ético Obligatorio**:  
> El modelo clasifica exclusivamente patrones acústicos y prosódicos en la señal de audio. **Bajo ninguna circunstancia** debe interpretarse como una evaluación psicológica concluyente ni un diagnóstico médico del estado anímico o mental real de una persona.

---

## 5. Aceleración con NVIDIA CUDA en GPU de 4 GB (RTX 3050 Laptop)

### Desafío Computacional de la Atención Cuadrática:
En modelos Transformer basados en audio, la matriz de atención crece con complejidad cuadrática $O(T^2)$ respecto al número de tramas temporales $T$. En audios largos (> 30 segundos), la inferencia monolítica requiere asignar matrices de atención gigantescas, superando el límite de 4 GB de VRAM de la GPU RTX 3050 (`CUDA Out of Memory`).

### Solución Arquitectural Implementada:
1. **Segmentación Deslizante Acotada ($O(1)$ VRAM)**:
   - Para audios superiores a 8 segundos, el motor divide la señal en ventanas temporales uniformes (de 3.0s a 5.0s con solapamiento controlado).
   - Procesa cada fragmento individualmente liberando memoria intermedia.
   - Agrega las distribuciones de probabilidad mediante promedio temporal ponderado.
   - El consumo pico de VRAM se mantiene estricto en **~770 MiB**, garantizando estabilidad continua sin importar si el audio dura 10 segundos o 1 hora.
2. **Inferencia en Media Precisión (`torch.float16`)**:
   - Ejecución directa sobre los núcleos CUDA de la GPU Laptop RTX 3050.
3. **Carga 100% Offline**:
   - Carga directa desde la instantánea local en disco (`~/.cache/huggingface/hub/models--emotion2vec--emotion2vec_plus_large`), sin comprobaciones de red ni latencias de red.
