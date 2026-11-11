# Banco de Audios de Prueba (Samples)

Este directorio contiene el conjunto de grabaciones de voz y muestras acústicas utilizadas para la evaluación, validación y benchmarking del sistema **AuraVoice** con el modelo **emotion2vec+ Large** (`emotion2vec/emotion2vec_plus_large`, ~300M según la model card, 162,1M contados en el checkpoint).

---

## 1. Muestras Recientes de Evaluación Emocional (Grabaciones Multiclase)

Estas grabaciones fueron incorporadas para evaluar la capacidad del modelo de discriminar entre diferentes polaridades de valencia y activación emocional tanto en **español** como en **inglés**:

| Archivo | Idioma | Emoción actuada | Predicción medida (top 2) | ¿Acierta? |
| :--- | :---: | :---: | :--- | :---: |
| `alegria-entusiasmo.m4a` | Español | Alegría | Tristeza 36,6 % · Neutral 26,3 % (incierta) | ❌ |
| `neutro-calma.m4a` | Español | Neutral | Tristeza 77,1 % · Neutral 20,3 % | ❌ |
| `frustración-enojo.m4a` | Español | Enojo | Tristeza 45,0 % · Disgusto 19,9 % (incierta) | ❌ |
| `happy-en.m4a` | Inglés | Alegría | Tristeza 48,2 % · Neutral 25,5 % (incierta) | ❌ |
| `neutral-en.m4a` | Inglés | Neutral | Neutral 72,5 % · Tristeza 26,1 % | ✅ |
| `frustation-en.m4a` | Inglés | Enojo | Tristeza 69,2 % · Neutral 17,1 % | ❌ |

Medido el 10-nov-2026 en GPU (FP32) con `python tests/test_inference.py <archivo>`. Son grabaciones actuadas por el equipo con micrófono de portátil: el modelo tiende a clasificarlas como **Tristeza**, con confianza baja cuando se equivoca. Es el caso límite documentado en la sección 3 del README principal.

---

## 2. Muestras de Referencia Base y Benchmark de Hardware

Archivos normalizados a 16.000 Hz PCM mono utilizados como línea base para medir latencia de inferencia, RTF (*Real-Time Factor*), consumo de VRAM y aceleración GPU (CUDA) frente a CPU:

| Archivo | Idioma | Formato | Duración | Propósito en el Benchmark |
| :--- | :---: | :---: | :---: | :--- |
| `sample_es.wav` | Español (`es`) | WAV (16 kHz Mono) | ~2.44 s | **Línea Base Principal**: Discurso formal en español (*"República de Colombia"*). Utilizado en la pestaña *Benchmark CUDA vs CPU* y en `tests/test_inference.py` (predicción medida: Tristeza 95,1 %). |
| `sample_es.ogg` | Español (`es`) | OGG Vorbis | ~2.44 s | Validación de decodificación y remuestreo automático desde formatos comprimidos a 16 kHz PCM. |
| `sample_en.wav` | Inglés (`en`) | WAV (16 kHz Mono) | ~7.18 s | Audio de media duración: se procesa en 4 ventanas de 3 s y se promedian las probabilidades (predicción medida: Neutral 69,3 %). |
| `sample_en.mp3` | Inglés (`en`) | MP3 | ~7.18 s | Verificación de compatibilidad con contenedores de compresión MPEG Layer-3. |
| `sample_zh.wav` | Chino (`zh`) | WAV (16 kHz Mono) | ~5.62 s | Prueba en otro idioma (predicción medida: Neutral 97,4 %). |
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
5. Pulsa el botón **Ejecutar Inferencia de Emoción en Audio** para observar:
   * La forma de onda coloreada según el estado capturado, con F0 y % de voz medidos (pYIN).
   * El desglose de probabilidades Softmax y el vector de Russell (Valencia / Activación).
   * En la **AuraTerminal**: latencia total, tiempo del modelo, bytes copiados CPU↔GPU y VRAM pico.

---

## 4. Licencia y Uso Académico

Estas muestras se distribuyen bajo licencia permisiva exclusivamente con fines académicos, de investigación y evaluación de prototipos en la asignatura de **Computación Paralela** (7mo Semestre).
