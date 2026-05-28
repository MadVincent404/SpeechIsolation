import os
import glob
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Audio Benchmark Viewer", layout="wide")

# DICTIONNAIRE DE TRADUCTION

LANGUAGES = {
    "FR": {
        "title": "Dashboard de Comparaison des Modèles Audio",
        "nav_header": "Navigation",
        "go_to": "Aller vers :",
        "page_global": "Résultats Globaux",
        "page_samples": "Écoute des Samples",
        "global_avg": "Moyennes globales (Sur 5 samples)",
        "filter_lang": "Filtrer la langue :",
        "all": "Toutes",
        "choose_metric": "Choisir la métrique cible :",
        "graph_title": "Comparaison du {metric} moyen",
        "raw_data": "Données brutes",
        "audio_analysis": "Analyse Audio Individuelle",
        "dir_error": "Le dossier `{base_dir}/` est introuvable au niveau du script.",
        "no_subdir": "Aucun sous-dossier trouvé dans `{base_dir}/`.",
        "lang_label": "Langue du dataset :",
        "sample_num": "Numéro de Sample :",
        "display_folder": "Affichage du dossier : `{folder}`",
        "sec1_title": "1. Signaux de Référence & Entrées",
        "voice_clean": "**Voix Originale (Propre)**",
        "voice_interf": "**Voix Interférante**",
        "no_interf": "Pas d'interférent (Tâche Noise pure)",
        "bg_noise": "**Bruit de fond**",
        "no_noise": "Pas de bruit de fond isolé",
        "input_noise": "**ENTRÉE Bruité (`With Noise`)**",
        "input_mix": "**ENTRÉE Mélangée (`Speech Mix`)**",
        "absent": "Absent",
        "not_avail": "Non disponible",
        "sec2_title": "2. Sorties Nettoyées par les Modèles",
        "no_output": "Aucun fichier commençant par `OUTPUT_` trouvé dans ce dossier."
    },
    "EN": {
        "title": "Audio Model Comparison Dashboard",
        "nav_header": "Navigation",
        "go_to": "Go to:",
        "page_global": "Global Results",
        "page_samples": "Sample Listening",
        "global_avg": "Global Averages (Over 5 samples)",
        "filter_lang": "Filter language:",
        "all": "All",
        "choose_metric": "Choose target metric:",
        "graph_title": "Comparison of average {metric}",
        "raw_data": "Raw Data",
        "audio_analysis": "Individual Audio Analysis",
        "dir_error": "The folder `{base_dir}/` cannot be found.",
        "no_subdir": "No subfolder found in `{base_dir}/`.",
        "lang_label": "Dataset Language:",
        "sample_num": "Sample Number:",
        "display_folder": "Displaying folder: `{folder}`",
        "sec1_title": "1. Reference Signals & Inputs",
        "voice_clean": "**Original Voice (Clean)**",
        "voice_interf": "**Interfering Voice**",
        "no_interf": "No interferent (Pure Noise task)",
        "bg_noise": "**Background Noise**",
        "no_noise": "No isolated background noise",
        "input_noise": "**Noisy INPUT (`With Noise`)**",
        "input_mix": "**Mixed INPUT (`Speech Mix`)**",
        "absent": "Missing",
        "not_avail": "Not available",
        "sec2_title": "2. Cleaned Outputs by Models",
        "no_output": "No file starting with `OUTPUT_` found in this folder."
    }
}

lang_choice = st.sidebar.selectbox("Langue / Language", ["FR", "EN"])
t = LANGUAGES[lang_choice]

# Titre dynamique de la page
st.title(t["title"])

# DONNEES SYNTHETIQUES

@st.cache_data
def load_summary_data():
    raw_data = [
        {"Language": "english", "Task": "Speech_Mix", "Model": "Ast-ConvTasNet-Libri2Mix", "PESQ": 1.823706, "STOI": 0.859592, "SI-SDR": 7.156403, "L1_Mel": 27.259010},
        {"Language": "english", "Task": "Speech_Mix", "Model": "SB-SepFormer-WHAMR16k", "PESQ": 1.772913, "STOI": 0.903081, "SI-SDR": 8.552147, "L1_Mel": 1.883515},
        {"Language": "english", "Task": "Speech_With_Noise", "Model": "Ast-ConvTasNet-Libri1Mix", "PESQ": 2.574313, "STOI": 0.944640, "SI-SDR": 10.748401, "L1_Mel": 17.545628},
        {"Language": "english", "Task": "Speech_With_Noise", "Model": "SB-SepFormer-DNS4", "PESQ": 2.719426, "STOI": 0.935153, "SI-SDR": 14.369688, "L1_Mel": 1.908773},
        {"Language": "english", "Task": "Speech_With_Noise", "Model": "SB-SepFormer-Enhancement", "PESQ": 2.229427, "STOI": 0.937691, "SI-SDR": 11.472964, "L1_Mel": 1.905832},
        {"Language": "french", "Task": "Speech_Mix", "Model": "Ast-ConvTasNet-Libri2Mix", "PESQ": 1.746410, "STOI": 0.885949, "SI-SDR": 5.565150, "L1_Mel": 28.448732},
        {"Language": "french", "Task": "Speech_Mix", "Model": "SB-SepFormer-WHAMR16k", "PESQ": 1.480719, "STOI": 0.853526, "SI-SDR": 5.197864, "L1_Mel": 2.709585},
        {"Language": "french", "Task": "Speech_With_Noise", "Model": "Ast-ConvTasNet-Libri1Mix", "PESQ": 2.763246, "STOI": 0.961309, "SI-SDR": 13.187845, "L1_Mel": 17.472190},
        {"Language": "french", "Task": "Speech_With_Noise", "Model": "SB-SepFormer-DNS4", "PESQ": 2.105685, "STOI": 0.882000, "SI-SDR": 7.980865, "L1_Mel": 3.024743},
        {"Language": "french", "Task": "Speech_With_Noise", "Model": "SB-SepFormer-Enhancement", "PESQ": 1.816939, "STOI": 0.889641, "SI-SDR": 8.206975, "L1_Mel": 3.001628},
    ]
    return pd.DataFrame(raw_data)

df_summary = load_summary_data()

# NAVIGATION VIA LA SIDEBAR

st.sidebar.markdown("---")
st.sidebar.header(t["nav_header"])
page = st.sidebar.radio(t["go_to"], [t["page_global"], t["page_samples"]])

base_dir = "output_audio"

# PAGE 1 : RESULTATS GLOBAUX

if page == t["page_global"]:
    st.subheader(t["global_avg"])

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_lang = st.selectbox(t["filter_lang"], [t["all"], "english", "french"])
    with col_f2:
        selected_metric = st.selectbox(t["choose_metric"], ["SI-SDR", "PESQ", "STOI", "L1_Mel"])

    df_filtered = df_summary.copy()
    if selected_lang != t["all"]:
        df_filtered = df_filtered[df_filtered["Language"] == selected_lang]

    fig = px.bar(
        df_filtered,
        x="Model",
        y=selected_metric,
        color="Task",
        facet_col="Language" if selected_lang == t["all"] else None,
        barmode="group",
        title=t["graph_title"].format(metric=selected_metric),
        text_auto=".2f"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(t["raw_data"])
    st.dataframe(
        df_summary.style.background_gradient(subset=["PESQ", "STOI", "SI-SDR"], cmap="viridis"),
        use_container_width=True
    )

# PAGE 2 : ECOUTE DES SAMPLES

elif page == t["page_samples"]:
    st.subheader(t["audio_analysis"])

    if not os.path.exists(base_dir):
        st.error(t["dir_error"].format(base_dir=base_dir))
    else:
        avail_langs = sorted([
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ])

        if not avail_langs:
            st.warning(t["no_subdir"].format(base_dir=base_dir))
            st.stop()

        lang = st.sidebar.selectbox(t["lang_label"], avail_langs)
        lang_dir = os.path.join(base_dir, lang)

        avail_samples = sorted([
            d for d in os.listdir(lang_dir)
            if os.path.isdir(os.path.join(lang_dir, d))
        ])

        if not avail_samples:
            st.warning(t["no_subdir"].format(base_dir=lang_dir))
            st.stop()

        sample_folder = st.sidebar.selectbox(t["sample_num"], avail_samples)
        target_folder = os.path.join(lang_dir, sample_folder)

        st.info(t["display_folder"].format(folder=target_folder))

        #  SECTION 1 : LES ENTREES ET REFERENCES 
        st.markdown(t["sec1_title"])

        col_ref1, col_ref2, col_ref3 = st.columns(3)
        with col_ref1:
            st.markdown(t["voice_clean"])
            p = os.path.join(target_folder, "01_ORIGINAL_clean.wav")
            if os.path.exists(p):
                st.audio(p)
            else:
                st.caption(t["absent"])

        with col_ref2:
            st.markdown(t["voice_interf"])
            p = os.path.join(target_folder, "02_ORIGINAL_interferent.wav")
            if os.path.exists(p):
                st.audio(p)
            else:
                st.caption(t["no_interf"])

        with col_ref3:
            st.markdown(t["bg_noise"])
            p = os.path.join(target_folder, "03_ORIGINAL_noise.wav")
            if os.path.exists(p):
                st.audio(p)
            else:
                st.caption(t["no_noise"])

        col_in1, col_in2 = st.columns(2)
        with col_in1:
            st.markdown(t["input_noise"])
            p = os.path.join(target_folder, "04_INPUT_speech_with_noise.wav")
            if os.path.exists(p):
                st.audio(p)
            else:
                st.caption(t["not_avail"])

        with col_in2:
            st.markdown(t["input_mix"])
            p = os.path.join(target_folder, "05_INPUT_speech_mix.wav")
            if os.path.exists(p):
                st.audio(p)
            else:
                st.caption(t["not_avail"])

        st.markdown("---")

        #  SECTION 2 : LES RESULTATS DES MODELES 
        st.markdown(t["sec2_title"])

        output_files = sorted(glob.glob(os.path.join(target_folder, "OUTPUT_*.wav")))

        if not output_files:
            st.warning(t["no_output"])
        else:
            for file_path in output_files:
                model_name = os.path.basename(file_path).replace("OUTPUT_", "").replace(".wav", "")

                with st.container():
                    col_txt, col_aud = st.columns([1, 2])
                    with col_txt:
                        st.write(f"**{model_name}**")
                    with col_aud:
                        st.audio(file_path)