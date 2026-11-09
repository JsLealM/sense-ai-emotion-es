# Banco de Audios de Prueba (Samples)

Este directorio contiene el conjunto de grabaciones de voz y muestras acústicas utilizadas para la evaluación, validación y benchmarking del sistema **AuraVoice** con los modelos fundacionales **emotion2vec+ Large** (315M params) y **W2V-BERT Emotion ES** (582M params).

---

## 1. Muestras Recientes de Evaluación Emocional (Grabaciones Multiclase)

Estas grabaciones fueron incorporadas para evaluar la capacidad del modelo de discriminar entre diferentes polaridades de valencia y activación emocional tanto en **español** como en **inglés**:

| Archivo | Idioma | Estado Emocional Objetivo | Formato / Tamaño | Fenómeno Acústico y Propósito de Evaluación |
| :--- | :---: | :---: | :---: | :--- |
| `alegria-entusiasmo.m4a` | Español (`es`) | **Alegría / Entusiasmo** | M4A (~201 KB) | Entonación ascendente, tempo ágil y alta energía espectral en frecuencias altas. Evalúa la detección de alta valencia (+V) y moderada-alta activación (+A). |
| `neutro-calma.m4a` | Español (`es`) | **Neutro / Calma** | M4A (~222 KB) | Prosodia uniforme, pausas regulares y balance armónico sin modulación extrema. Evalúa la clasificación de neutralidad y serenidad acústica. |
| `frustración-enojo.m4a` | Español (`es`) | **Frustración / Enojo** | M4A (~206 KB) | Modulación tensa, inflexiones cortantes y picos de intensidad vocal. Evalúa la detección de baja valencia (-V) y alta activación (+A). |
| `happy-en.m4a` | Inglés (`en`) | **Happy / Joy** | M4A (~208 KB) | Tono brillante y variabilidad en $F_0$. Comprueba la capacidad de generalización multilingüe (cross-lingual) para alegría en inglés. |
| `neutral-en.m4a` | Inglés (`en`) | **Neutral / Calm** | M4A (~202 KB) | Cadencia conversacional pausada y tono plano. Comprueba la estabilidad de la predicción neutral en inglés. |
| `frustation-en.m4a` | Inglés (`en`) | **Frustration / Anger** | M4A (~222 KB) | Elevada energía prosódica con tensión en formantes vocales. Comprueba la discriminación de tensión acústica en inglés. |

---

## 2. Muestras de Referencia Base y Benchmark de Hardware

Archivos normalizados a 16.000 Hz PCM mono utilizados como línea base para medir latencia de inferencia, RTF (*Real-Time Factor*), consumo de VRAM y aceleración GPU (CUDA) frente a CPU:

| Archivo | Idioma | Formato | Duración | Propósito en el Benchmark |
| :--- | :---: | :---: | :---: | :--- |
| `sample_es.wav` | Español (`es`) | WAV (16 kHz Mono) | ~2.44 s | **Línea Base Principal**: Discurso formal en español (*"República de Colombia"*). Utilizado en la pestaña *Benchmark CUDA vs CPU* y en `tests/test_inference.py`. |
| `sample_es.ogg` | Español (`es`) | OGG Vorbis | ~2.44 s | Validación de decodificación y remuestreo automático desde formatos comprimidos a 16 kHz PCM. |
| `sample_en.wav` | Inglés (`en`) | WAV (16 kHz Mono) | ~7.18 s | Prueba de audio de media duración para validar agregación temporal por ventaneo continuo $O(1)$ sin desbordamiento de memoria VRAM. |
| `sample_en.mp3` | Inglés (`en`) | MP3 | ~7.18 s | Verificación de compatibilidad con contenedores de compresión MPEG Layer-3. |
| `sample_zh.wav` | Chino (`zh`) | WAV (16 kHz Mono) | ~5.62 s | Validación en el idioma nativo de pre-entrenamiento de los pesos base de Alibaba FunAudioLLM. |
| `sample_zh.mp3` | Chino (`zh`) | MP3 | ~5.62 s | Variante comprimida para evaluación de robustez del extractor acústico. |

---

## 3. Uso en la Aplicación Web

Para utilizar cualquiera de estas muestras dentro de la interfaz gráfica interactiva:

1. Inicia la aplicación con:
   ```bash
   streamlit run app.py
   ```
2. En la pestaña **Sesión en Vivo (Análisis Emocional)**, localiza la tarjeta izquierda **Captura de Voz**.
3. Selecciona la opción **Muestra Pregrabada**.
4. En el menú desplegable, selecciona la grabación que desees evaluar (ej. `Grabación: alegria-entusiasmo.m4a`, `Grabación: neutro-calma.m4a` o `sample_es.wav`).
5. Pulsa el botón **▶ Analizar Emoción en Audio** para observar:
   * La forma de onda espectral coloreada según el estado capturado.
   * El desglose de probabilidades Softmax y el vector de Russell (Valencia / Activación).
   * La telemetría de inferencia en milisegundos y el consumo de VRAM en la **AuraTerminal**.

---

## 4. Licencia y Uso Académico

Estas muestras se distribuyen bajo licencia permisiva exclusivamente con fines académicos, de investigación y evaluación de prototipos en la asignatura de **Computación Paralela** (7mo Semestre).
