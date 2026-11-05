"""
SENSE AI — Analizador emocional de voz en español
Prototipo universitario para la evaluación del modelo UMUTeam/w2v-bert-emotion-es
con aceleración NVIDIA CUDA en GPU GeForce RTX 3050 (4 GB VRAM).
"""

import os
import sys
import tempfile
import time
import streamlit as st
import pandas as pd
import numpy as np

# Configurar ruta raíz del proyecto
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.cuda_monitor import get_cuda_status, verify_model_device
from src.audio_processor import get_audio_metadata, convert_to_wav_16k_mono
from src.result_parser import EMOTION_TRANSLATION, EMOTION_COLORS, EMOTION_EMOJIS, ETHICAL_DISCLAIMER
from src.inference import run_emotion_inference
from src.model_loader import load_emotion_model
from src.evaluator import compute_evaluation_metrics, ALL_EMOTIONS

# Configuración de Streamlit
st.set_page_config(
    page_title="SENSE AI — Analizador emocional de voz",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS modernos y limpios
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .badge-cuda {
        background-color: #10B981;
        color: white;
        padding: 6px 14px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-cpu {
        background-color: #F59E0B;
        color: white;
        padding: 6px 14px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .pred-card {
        padding: 20px;
        border-radius: 12px;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        text-align: center;
        margin-bottom: 15px;
    }
    .pred-title {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
    }
    .pred-emotion {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 6px 0;
    }
    .pred-score {
        font-size: 1.2rem;
        font-weight: 600;
        opacity: 0.95;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #92400E;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_cached_model(device_choice: str, use_fp16: bool):
    """Carga y mantiene en memoria los pesos de W2V-BERT Emotion ES."""
    return load_emotion_model(device_name=device_choice, use_fp16=use_fp16)


# Inicializar historial de pruebas en st.session_state
if "test_history" not in st.session_state:
    st.session_state.test_history = []


# ==========================================
# BARRA LATERAL: Telemetría y Configuración GPU
# ==========================================
st.sidebar.markdown("### Panel de Control y GPU")

cuda_info = get_cuda_status()
is_cuda_avail = cuda_info["is_available"]

device_options = ["cuda:0", "cpu"] if is_cuda_avail else ["cpu"]
device_labels = {
    "cuda:0": f"NVIDIA CUDA ({cuda_info.get('device_name', 'GPU')})",
    "cpu": "CPU (Modo Diagnóstico / Fallback)"
}

selected_device = st.sidebar.selectbox(
    "Dispositivo de Inferencia:",
    options=device_options,
    format_func=lambda d: device_labels.get(d, d),
    index=0
)

use_fp16_toggle = st.sidebar.checkbox(
    "Optimización FP16 (Media Precisión)",
    value=True,
    help="Reduce el uso de VRAM de ~2.4 GB a ~1.2 GB, ideal para GPUs de 4 GB como la RTX 3050 Laptop."
)

if "cuda" in selected_device:
    st.sidebar.markdown('<span class="badge-cuda">GPU CUDA ACTIVA</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="badge-cpu">MODO CPU ACTIVO</span>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Hardware y CUDA")
st.sidebar.write(f"**Modelo:** `UMUTeam/w2v-bert-emotion-es`")
st.sidebar.write(f"**GPU:** {cuda_info.get('device_name', 'No disponible')}")
st.sidebar.write(f"**PyTorch:** {cuda_info['pytorch_version']}")
st.sidebar.write(f"**CUDA Runtime:** {cuda_info.get('cuda_version', 'N/A')}")

if is_cuda_avail:
    total_vram = cuda_info["vram_total_mb"]
    used_vram = cuda_info["vram_reserved_mb"]
    free_vram = cuda_info["vram_free_mb"]
    usage_pct = min(1.0, max(0.0, used_vram / total_vram)) if total_vram > 0 else 0.0

    st.sidebar.write(f"**VRAM Total:** {total_vram} MB (~{round(total_vram/1024, 2)} GB)")
    st.sidebar.write(f"**VRAM Ocupada:** {used_vram} MB")
    st.sidebar.write(f"**VRAM Libre:** {free_vram} MB")
    st.sidebar.progress(usage_pct, text=f"Uso VRAM: {int(usage_pct * 100)}%")
else:
    st.sidebar.warning("CUDA no está disponible en PyTorch. Ejecutando en CPU.")

st.sidebar.markdown("---")
st.sidebar.caption("Proyecto Universitario de Computación Paralela | Speech Emotion Recognition")


# ==========================================
# ENCABEZADO PRINCIPAL
# ==========================================
st.markdown('<div class="main-title">SENSE AI — Analizador emocional de voz en español</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Clasificación de emociones en el habla mediante el modelo fundacional '
    '<b>W2V-BERT Emotion ES</b> (UMUTeam) acelerado por hardware con NVIDIA CUDA.</div>',
    unsafe_allow_html=True
)

# Pestañas principales
tab_analysis, tab_lab, tab_eval, tab_docs = st.tabs([
    "Análisis Emocional",
    "Laboratorio de Pruebas & Benchmark",
    "Evaluación Experimental (Métricas)",
    "Ficha Técnica & Documentación"
])


# ==========================================
# PESTAÑA 1: ANÁLISIS EMOCIONAL
# ==========================================
with tab_analysis:
    col_up, col_play = st.columns([1.2, 1])

    with col_up:
        st.markdown("#### 1. Cargar Archivo de Voz en Español")
        uploaded_file = st.file_uploader(
            "Selecciona un audio con voz en español (WAV, MP3, M4A, OGG, FLAC):",
            type=["wav", "mp3", "m4a", "ogg", "flac"]
        )

    temp_audio_path = None
    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_audio_path = tmp_file.name

        with col_play:
            st.markdown("#### 2. Reproductor y Señal")
            st.audio(temp_audio_path)
            meta = get_audio_metadata(temp_audio_path)
            st.markdown(f"""
            - **Archivo:** `{uploaded_file.name}`
            - **Duración:** `{meta['duration_seconds']} s`
            - **Frecuencia original:** `{meta['sample_rate']} Hz` *(normalizada a 16.000 Hz)*
            - **Canales:** `{'Mono' if meta['channels'] == 1 else 'Estéreo'}`
            - **Tamaño:** `{meta['size_kb']} KB`
            """)

        st.markdown("---")
        btn_analyze = st.button("Analizar emoción en la voz", type="primary", use_container_width=True)

        if btn_analyze:
            with st.spinner(f"Cargando modelo W2V-BERT y ejecutando inferencia en {selected_device}..."):
                try:
                    # 1. Cargar modelo
                    model, feat_extractor, load_meta = get_cached_model(selected_device, use_fp16_toggle)

                    # 2. Ejecutar inferencia
                    results = run_emotion_inference(
                        model=model,
                        feature_extractor=feat_extractor,
                        audio_path=temp_audio_path,
                        device=selected_device
                    )

                    if results["status"] != "EXITOSA":
                        st.error(f"Error durante el procesamiento: {results.get('error_message')}")
                    else:
                        parsed = results["parsed_result"]
                        metrics = results["metrics"]

                        # Guardar en historial
                        st.session_state.test_history.append({
                            "Audio": uploaded_file.name,
                            "Duración (s)": metrics["audio_duration_seconds"],
                            "Emoción Predominante": parsed['predominant_emotion'],
                            "Confianza": f"{parsed['predominant_percent']}%",
                            "Dispositivo": metrics["device_used"],
                            "Latencia (ms)": metrics["latency_ms"],
                            "RTF": metrics["real_time_factor_rtf"],
                            "VRAM Pico (MB)": metrics["vram_stats"]["peak_mb"]
                        })

                        st.success("Inferencia completada con éxito.")

                        # Mostrar Resultados
                        st.markdown("### Diagnóstico Emocional")
                        res_col1, res_col2 = st.columns([1, 1.4])

                        with res_col1:
                            st.markdown(f"""
                            <div class="pred-card">
                                <div class="pred-title">Emoción Predominante</div>
                                <div class="pred-emotion">{parsed['predominant_emotion']}</div>
                                <div class="pred-score">Nivel de Confianza: {parsed['predominant_percent']}%</div>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown("#### Detalle de Puntuaciones:")
                            df_scores = pd.DataFrame(parsed["ranking"])[["categoria", "porcentaje", "probabilidad"]]
                            df_scores.columns = ["Categoría", "Porcentaje (%)", "Probabilidad"]
                            st.dataframe(df_scores, hide_index=True, use_container_width=True)

                        with res_col2:
                            st.markdown("#### Distribución en las 6 Categorías")

                            # Gráfica horizontal de barras
                            chart_df = pd.DataFrame({
                                "Emoción": [item["categoria"] for item in parsed["ranking"]],
                                "Porcentaje": [item["porcentaje"] for item in parsed["ranking"]]
                            })
                            st.bar_chart(chart_df.set_index("Emoción"), horizontal=True, color="#3B82F6")

                            st.markdown(f'<div class="disclaimer-box">{parsed["disclaimer"]}</div>', unsafe_allow_html=True)

                        st.markdown("---")

                        # Métricas Técnicas y CUDA
                        st.markdown("#### Rendimiento Técnico y Evidencia de Hardware")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Latencia de Inferencia", f"{metrics['latency_ms']} ms")
                        m2.metric("Factor Tiempo Real (RTF)", f"{metrics['real_time_factor_rtf']}")
                        m3.metric("Aceleración vs Tiempo Real", metrics["speedup_vs_realtime"])
                        m4.metric("Dispositivo Físico", metrics["device_used"])

                        vram = metrics["vram_stats"]
                        st.markdown(f"""
                        **Detalle de Consumo de Memoria VRAM:**
                        - **VRAM Inicial:** `{vram['start_mb']} MB`
                        - **VRAM Pico durante inferencia:** `{vram['peak_mb']} MB`
                        - **VRAM Final tras ejecución:** `{vram['end_mb']} MB`
                        - **Precisión utilizada:** `{'FP16 (Media Precisión)' if use_fp16_toggle and 'cuda' in selected_device else 'FP32'}`
                        """)

                except Exception as exc:
                    st.error(f"Error inesperado: {exc}")
                    import traceback
                    st.code(traceback.format_exc())

    else:
        st.info("Sube un archivo de audio en español para iniciar el análisis o consulta las muestras en el Laboratorio de Pruebas.")


# ==========================================
# PESTAÑA 2: LABORATORIO DE PRUEBAS & BENCHMARK
# ==========================================
with tab_lab:
    st.markdown("### Laboratorio de Pruebas Experimentales y Benchmark")
    st.markdown("Evalúa grabaciones en español con diferentes tonos emocionales y compara el desempeño CUDA vs CPU.")

    samples_dir = os.path.join(CURRENT_DIR, "samples")
    sample_options = {
        "Audio Español 1 (sample_es.wav)": os.path.join(samples_dir, "sample_es.wav"),
    }

    # Detectar audios adicionales en samples/
    if os.path.exists(samples_dir):
        for f in os.listdir(samples_dir):
            if f.endswith((".wav", ".mp3", ".ogg")) and f not in ["sample_en.wav", "sample_zh.wav"]:
                key = f"Muestra en Español: {f}"
                sample_options[key] = os.path.join(samples_dir, f)

    selected_sample = st.selectbox("Selecciona una grabación de prueba:", options=list(sample_options.keys()))
    sample_path = sample_options[selected_sample]

    if os.path.exists(sample_path):
        st.audio(sample_path)
        s_meta = get_audio_metadata(sample_path)
        st.caption(f"Duración: {s_meta['duration_seconds']} s | Muestreo: {s_meta['sample_rate']} Hz")

        l_col1, l_col2 = st.columns(2)
        with l_col1:
            btn_lab_cuda = st.button("Inferencia en GPU NVIDIA (CUDA)", use_container_width=True, disabled=not is_cuda_avail)
        with l_col2:
            btn_lab_compare = st.button("Benchmark Comparativo (CUDA vs CPU)", use_container_width=True, disabled=not is_cuda_avail)

        if btn_lab_cuda:
            with st.spinner("Procesando en GPU CUDA..."):
                m_cuda, fe_cuda, _ = get_cached_model("cuda:0", use_fp16_toggle)
                res_c = run_emotion_inference(m_cuda, fe_cuda, sample_path, device="cuda:0")

                p_c = res_c["parsed_result"]
                m_c = res_c["metrics"]

                st.markdown("#### Resultado en GPU NVIDIA CUDA:")
                st.write(f"**Emoción Predominante:** **{p_c['predominant_emotion']}** ({p_c['predominant_percent']}%)")
                st.write(f"**Latencia:** {m_c['latency_ms']} ms | **RTF:** {m_c['real_time_factor_rtf']} | **VRAM Pico:** {m_c['vram_stats']['peak_mb']} MB")

        if btn_lab_compare:
            with st.spinner("Ejecutando en CUDA y posteriormente en CPU (esto puede tardar unos segundos)..."):
                # Ejecutar en CUDA
                m_cuda, fe_cuda, _ = get_cached_model("cuda:0", use_fp16_toggle)
                res_c = run_emotion_inference(m_cuda, fe_cuda, sample_path, device="cuda:0")

                # Ejecutar en CPU
                m_cpu, fe_cpu, _ = get_cached_model("cpu", False)
                res_cpu = run_emotion_inference(m_cpu, fe_cpu, sample_path, device="cpu")

                st.markdown("#### Tabla Comparativa de Aceleración por Hardware")

                compare_data = {
                    "Parámetro / Métrica": [
                        "Dispositivo de Cómputo",
                        "Tiempo de Inferencia (Latencia)",
                        "Real-Time Factor (RTF)",
                        "Factor de Aceleración",
                        "Memoria VRAM Utilizada",
                        "Emoción Predominante",
                        "Nivel de Confianza"
                    ],
                    "NVIDIA GPU (CUDA)": [
                        cuda_info.get("device_name", "GPU"),
                        f"{res_c['metrics']['latency_ms']} ms",
                        str(res_c['metrics']['real_time_factor_rtf']),
                        res_c['metrics']['speedup_vs_realtime'],
                        f"{res_c['metrics']['vram_stats']['peak_mb']} MB",
                        res_c["parsed_result"]["predominant_emotion"],
                        f"{res_c['parsed_result']['predominant_percent']}%"
                    ],
                    "Procesador (CPU)": [
                        "CPU del Sistema",
                        f"{res_cpu['metrics']['latency_ms']} ms",
                        str(res_cpu['metrics']['real_time_factor_rtf']),
                        res_cpu['metrics']['speedup_vs_realtime'],
                        "0 MB (Memoria RAM)",
                        res_cpu["parsed_result"]["predominant_emotion"],
                        f"{res_cpu['parsed_result']['predominant_percent']}%"
                    ]
                }
                st.table(pd.DataFrame(compare_data))

                lat_cuda = res_c["metrics"]["latency_seconds"]
                lat_cpu = res_cpu["metrics"]["latency_seconds"]
                if lat_cuda > 0:
                    speedup = round(lat_cpu / lat_cuda, 2)
                    st.success(f"**Aceleración por Hardware:** CUDA ejecutó la inferencia **{speedup}x veces más rápido** que la CPU.")

    # Historial de pruebas
    if st.session_state.test_history:
        st.markdown("---")
        st.markdown("#### Historial de Pruebas de la Sesión")
        df_hist = pd.DataFrame(st.session_state.test_history)
        st.dataframe(df_hist, use_container_width=True)


# ==========================================
# PESTAÑA 3: EVALUACIÓN EXPERIMENTAL (MÉTRICAS)
# ==========================================
with tab_eval:
    st.markdown("### Evaluación Experimental del Clasificador de Emociones")
    st.markdown("""
    Esta sección implementa el cálculo riguroso de métricas de desempeño supervisado:
    **Accuracy, Precision, Recall, F1-Score** y la **Matriz de Confusión**.

    > [!IMPORTANT]
    > **Principio Metodológico**: Las métricas de evaluación exigen un conjunto de prueba con **etiquetas reales (Ground Truth)**
    > anotadas previamente por evaluadores humanos o provenientes de un corpus de referencia (ej. MAV-ES, Emo-DB).
    > **Nunca** se deben calcular métricas utilizando las predicciones del propio modelo como verdad absoluta.
    """)

    st.markdown("#### 1. Cargar Archivo de Etiquetas de Referencia (CSV)")
    st.caption("El archivo CSV debe contener dos columnas: `etiqueta_real` y `etiqueta_predicha` (o nombre de archivo para evaluar en lote).")

    eval_file = st.file_uploader("Cargar dataset de validación (.csv)", type=["csv"], key="eval_csv")

    demo_eval = st.button("Cargar conjunto de demostración para sustentación")

    y_true_data, y_pred_data = None, None

    if eval_file is not None:
        try:
            df_eval = pd.read_csv(eval_file)
            if "etiqueta_real" in df_eval.columns and "etiqueta_predicha" in df_eval.columns:
                y_true_data = df_eval["etiqueta_real"].tolist()
                y_pred_data = df_eval["etiqueta_predicha"].tolist()
            else:
                st.error("El CSV debe contener las columnas: 'etiqueta_real' y 'etiqueta_predicha'.")
        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {e}")

    elif demo_eval:
        # Conjunto de demostración representativo para mostrar la funcionalidad ante el profesor
        y_true_data = [
            "Alegría", "Alegría", "Alegría", "Tristeza", "Tristeza", "Enojo",
            "Enojo", "Enojo", "Miedo", "Disgusto", "Neutral", "Neutral"
        ]
        y_pred_data = [
            "Alegría", "Alegría", "Neutral", "Tristeza", "Neutral", "Enojo",
            "Enojo", "Disgusto", "Miedo", "Disgusto", "Neutral", "Tristeza"
        ]
        st.info("Se ha cargado un conjunto representativo de 12 muestras de validación.")

    if y_true_data and y_pred_data:
        metrics_res = compute_evaluation_metrics(y_true_data, y_pred_data, labels=ALL_EMOTIONS)

        st.markdown("#### 2. Resumen de Métricas Globales")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Accuracy Global", f"{metrics_res['accuracy_percent']}%")
        col_m2.metric("Precision (Macro)", f"{metrics_res['precision_macro']}")
        col_m3.metric("Recall (Macro)", f"{metrics_res['recall_macro']}")
        col_m4.metric("F1-Score (Macro)", f"{metrics_res['f1_macro']}")

        st.markdown("#### 3. Matriz de Confusión")
        cm_matrix = np.array(metrics_res["confusion_matrix"])
        df_cm = pd.DataFrame(cm_matrix, index=ALL_EMOTIONS, columns=ALL_EMOTIONS)
        st.write("Filas: **Etiqueta Real (Ground Truth)** | Columnas: **Predicción del Modelo**")
        st.dataframe(df_cm, use_container_width=True)

        st.markdown("#### 4. Reporte por Categoría Emocional")
        rep_dict = metrics_res["classification_report"]
        rows = []
        for emo in ALL_EMOTIONS:
            if emo in rep_dict:
                item = rep_dict[emo]
                rows.append({
                    "Emoción": emo,
                    "Precision": round(item["precision"], 4),
                    "Recall": round(item["recall"], 4),
                    "F1-Score": round(item["f1-score"], 4),
                    "Muestras de Soporte": int(item["support"])
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ==========================================
# PESTAÑA 4: FICHA TÉCNICA & DOCUMENTACIÓN
# ==========================================
with tab_docs:
    st.markdown("### Ficha Técnica del Modelo (W2V-BERT Emotion ES)")

    model_notes_path = os.path.join(CURRENT_DIR, "MODEL_CARD_NOTES.md")
    if os.path.exists(model_notes_path):
        with open(model_notes_path, "r", encoding="utf-8") as f:
            notes_content = f.read()
        st.markdown(notes_content)
    else:
        st.info("El archivo MODEL_CARD_NOTES.md no está disponible.")
