# Banco de Audios de Prueba (Samples)

Este directorio contiene audios de muestra diseñados para evaluar y evidenciar el comportamiento del modelo **SenseVoice Small** en el prototipo académico.

## Estructura de Muestras

| Archivo | Idioma Esperado | Contenido / Fenómeno Acústico | Propósito en la Evaluación |
| :--- | :--- | :--- | :--- |
| `sample_en.wav` | Inglés (`en`) | Discurso claro en inglés en tono neutral | Comprobar precisión de ASR y velocidad de inferencia en idioma oficial. |
| `sample_zh.wav` | Chino Mandarín (`zh`) | Frase en mandarín estándar | Validar el idioma nativo de entrenamiento de Alibaba FunAudioLLM. |
| `sample_es.wav` | Español (`es`) | Discurso en español latinoamericano | **Evaluación de limitación**: Demostrar empíricamente que el español no está soportado nativamente y observar la respuesta del modelo (etiquetado fonético o asignación errónea). |
| `sample_laughter_en.wav` | Inglés (`en`) | Frase casual acompañada de risas | Evaluar detección de eventos de audio (`<|Laughter|>`) y emoción (`<|HAPPY|>`). |

## Generación y Licencia
Los audios sintéticos o grabaciones de prueba se distribuyen bajo licencia permisiva (Creative Commons / Dominio Público) exclusivamente para fines académicos y de investigación.
