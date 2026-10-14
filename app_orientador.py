"""Orientador académico educativo con reglas explicables y Random Forest."""

from __future__ import annotations

import streamlit as st

from orientacion_engine import (
    INTERESTS,
    PATHS,
    evaluate_expert_rules,
    predict_with_model,
    train_demo_models,
)

st.set_page_config(
    page_title="Brújula | Orientador académico",
    page_icon="🧭",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(150deg, #f3f7f5 0%, #f8f7f2 58%, #edf3f4 100%); }
    .block-container { max-width: 1120px; padding-top: 2.2rem; }
    .hero { padding: 1.8rem 2rem; border-radius: 22px; color: #f5fbf8;
            background: linear-gradient(115deg, #173d39, #287567); margin-bottom: 1.3rem; }
    .hero h1 { color: #fff; margin: 0; font-size: 2.5rem; }
    .hero p { margin: .5rem 0 0; color: #d9eee7; }
    [data-testid="stMetric"] { background: #ffffffb8; border: 1px solid #dce8e3;
                               border-radius: 14px; padding: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🧭 Brújula</h1>
      <p>Explora estudios y profesiones que podrían encajar con tus intereses.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Una herramienta educativa para explorar opciones; no es un test psicométrico "
    "ni sustituye la orientación académica profesional."
)

@st.cache_resource
def load_models():
    return train_demo_models()


tab_test, tab_results, tab_method = st.tabs(
    ["1 · Tus intereses", "2 · Orientación", "3 · Cómo funciona"]
)

with tab_test:
    st.subheader("¿Qué cosas te interesan?")
    st.write("Valora cada área del **0 (nada)** al **5 (mucho)**. No hay respuestas correctas.")
    with st.form("interest_profile"):
        left, right = st.columns(2)
        profile = {}
        for index, (key, label) in enumerate(INTERESTS.items()):
            with left if index % 2 == 0 else right:
                profile[key] = st.slider(
                    label,
                    min_value=0,
                    max_value=5,
                    value=3,
                    help="Indica cuánto te interesa esta actividad o temática.",
                    key=f"interest_{key}",
                )
        submitted = st.form_submit_button("Ver mi orientación", type="primary", use_container_width=True)
        if submitted:
            st.session_state["orientation_profile"] = profile

    if "orientation_profile" not in st.session_state:
        st.info("Completa las seis áreas y pulsa «Ver mi orientación» para ver sugerencias.")

with tab_results:
    profile = st.session_state.get("orientation_profile")
    if profile is None:
        st.info("Primero completa el cuestionario de la pestaña «Tus intereses».")
    else:
        decision_tree, forest = load_models()
        tree_results = predict_with_model(decision_tree, profile)
        forest_results = predict_with_model(forest, profile)
        rule_results = evaluate_expert_rules(profile)
        tree_path, tree_share = tree_results[0]
        forest_path, forest_share = forest_results[0]

        st.subheader("Tu mapa inicial")
        summary = st.columns(3)
        summary[0].metric("Árbol de decisión", tree_path)
        summary[1].metric("Random Forest", forest_path)
        summary[2].metric(
            "Sistema experto",
            rule_results[0].path if rule_results else "Sin regla activa",
        )
        st.caption(
            f"Proporción en la hoja del árbol: {tree_share:.0%}. "
            f"Proporción de árboles que votan por la primera ruta del bosque: {forest_share:.0%}. "
            "Son estimaciones de este modelo simulado, no medidas de aptitud ni de éxito."
        )

        st.markdown("#### Coincidencias del sistema experto")
        if rule_results:
            top_rule = rule_results[0]
            metrics = st.columns(3)
            metrics[0].metric("Ruta con más reglas", top_rule.path)
            metrics[1].metric("Puntos de reglas", str(top_rule.score))
            metrics[2].metric("Reglas activadas", str(len(top_rule.evidence)))

            path_details = PATHS[top_rule.path]
            st.write(path_details["description"])
            st.write("**Estudios para explorar:** " + " · ".join(path_details["studies"]))
            st.write("**Señales detectadas:** " + "; ".join(top_rule.evidence) + ".")

            if len(rule_results) > 1:
                with st.expander("Ver otras coincidencias de las reglas"):
                    for result in rule_results[1:]:
                        st.markdown(f"**{result.path}** — {result.score} puntos")
                        st.caption("; ".join(result.evidence) + ".")
        else:
            st.info(
                "Todavía no se activa ninguna regla con tus respuestas. "
                "Prueba a subir alguna valoración o usa las rutas del modelo como ideas para explorar."
            )

        st.markdown("#### Rutas sugeridas por Random Forest")
        chart_data = {
            "Ruta": [path for path, _ in forest_results],
            "Voto de árboles (%)": [round(probability * 100, 1) for _, probability in forest_results],
        }
        st.bar_chart(chart_data, x="Ruta", y="Voto de árboles (%)")
        for path, _ in forest_results[:3]:
            details = PATHS[path]
            st.markdown(f"**{path}** — {details['description']}")
            st.caption("Algunas opciones: " + " · ".join(details["studies"]))

with tab_method:
    st.subheader("Tres métodos, tres perspectivas")
    st.markdown(
        """
        **Árbol de decisión.** Aprende una secuencia de divisiones y asigna la ruta
        correspondiente a la hoja final. Es el modelo más sencillo de inspeccionar.

        **Random Forest.** Entrena muchos árboles sobre muestras con reemplazo y combina
        sus resultados; la app muestra el reparto de votos entre las rutas.

        **Sistema experto (reglas IF/THEN).** Compara tus valoraciones con condiciones
        definidas de antemano. Por ejemplo: si el interés por naturaleza es 4 o 5 y el
        interés por ciencia es al menos 3, suma puntos a «Medio ambiente». La app muestra
        qué señales activaron cada recomendación. Las reglas constituyen su base de
        conocimiento y la función de evaluación actúa como motor de inferencia.

        Los dos modelos supervisados se entrenan al iniciar la app con perfiles simulados
        alrededor de arquetipos definidos para cada ruta; no se usan datos personales ni
        un estudio real de orientación.

        **Cómo interpretar el resultado.** Las rutas sirven para iniciar una conversación
        o investigar asignaturas y salidas profesionales. Contrasta las sugerencias con
        tus experiencias, información oficial de estudios y una persona orientadora.
        """
    )
    st.warning(
        "La parte de Random Forest es una demostración técnica con datos sintéticos. "
        "Sus votos no deben interpretarse como diagnóstico, aptitud, admisión o garantía laboral."
    )
