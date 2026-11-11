# FICHA TÉCNICA DEL MODELO: emotion2vec+ Large

Documento técnico elaborado para la asignatura de **Computación Paralela / Arquitecturas Paralelas** (Universidad).

---

## 1. Identificación General del Modelo

| Parámetro | Detalle |
| :--- | :--- |
| **Nombre Oficial** | **emotion2vec+ Large** (`emotion2vec/emotion2vec_plus_large`) |
| **Entidad Desarrolladora** | Organización **emotion2vec** (Ziyang Ma et al., SJTU + Alibaba / FunASR) |
| **Repositorio Hugging Face** | [https://huggingface.co/emotion2vec/emotion2vec_plus_large](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |
| **Publicación Científica** | *emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation* (arXiv:2312.15185) |
| **Tarea Principal** | Reconocimiento de Emociones en el Habla (**Speech Emotion Recognition - SER**) Acústico Puro |
| **Dominio Acústico** | **Universal / Multilingüe** (Inferencia acústica de representaciones latentes invariantes al idioma) |
| **Licencia** | `model-license` de FunASR (ver `MODEL_LICENSE.txt`): uso citando la fuente y conservando el nombre del modelo |

---

## 2. Descripción y Problema que Resuelve

### ¿Qué problema resuelve?
El reconocimiento de emociones en la voz (SER) enfrenta el reto de modelar las variaciones acústicas paralingüísticas (entonación, ritmo, armónicos, energía, formantes) independientemente del contenido semántico textual. La mayoría de modelos multimodales dependen de transcribir texto (ASR) o acoplar grandes modelos de lenguaje (LLM), lo que introduce latencias masivas, fallas alucinatorias y sesgos léxicos.

### Propuesta de emotion2vec+ Large:
**emotion2vec+ Large** es un modelo fundacional acústico puro de última generación (*State-of-the-Art*). Entrenado mediante aprendizaje autosupervisado sobre decenas de miles de horas de habla, aprende a proyectar el audio crudo directamente en un espacio latente estructurado según la valencia y excitación emocional. Extrae probabilidades de 9 estados emocionales directamente de la forma de onda acústica en tiempo real.

---

## 3. Arquitectura y Parámetros

* **Arquitectura Base**: **Data2Vec 2.0** — CNN de 7 capas sobre la forma de onda + 12 bloques Transformer (4 prenet + 8), 16 cabezas, posiciones ALiBi.
* **Número de Parámetros**: **~300M** según la model card; **162.140.185** contados en el checkpoint de inferencia (`sum(p.numel())`, sin el decoder de preentrenamiento).
* **Dimensión de Embedding**: 1024 canales latentes, 50 frames por segundo.
* **Preprocesamiento Acústico**: Señal de audio remuestreada a **16.000 Hz (16 kHz), canal mono**.
* **Precisión en la app**: **FP32** (`torch.float32`), ~626 MiB de pesos en VRAM.
* **Capa de Clasificación**: Proyección lineal 1024→9 con Softmax.

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

### Desafío: atención cuadrática
La matriz de atención crece con O(T²) respecto al número de frames T (50 por segundo). Con 60 s de audio, T = 3.000 y los puntajes de una sola capa ocupan ~576 MB en FP32 (16 cabezas × 3.000² × 4 bytes), lo que en 4 GB termina en `CUDA out of memory`.

### Solución implementada
1. **Ventanas de duración fija** (3 s, paso 2 s, configurables): la memoria por ventana no depende de la duración total del audio. Las probabilidades de las ventanas se promedian. Pico medido: ~720–830 MiB.
2. **FP32 sin cuantizar**: el modelo cabe holgado; FP16 queda como trabajo futuro.
3. **Carga offline** desde `~/.cache/huggingface/hub/models--emotion2vec--emotion2vec_plus_large` cuando los pesos ya están descargados.
