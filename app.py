"""Détecteur d'hallucinations basé sur l'analyse des logprobs (UI Streamlit).

Pose une question à un LLM local (Ollama), analyse la confiance de chaque token
généré, et bloque la réponse si un mot grammaticalement critique (nom, nombre,
verbe...) a été produit avec une probabilité inférieure au seuil défini.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from config import DEFAULT_MODEL, DEFAULT_THRESHOLD, OLLAMA_BASE_URL, SAFETY_MESSAGE
from confidence_analyzer import find_critical_tokens, load_spacy_model
from llm_client import build_client, query_model
from plots import plot_token_probabilities


@st.cache_resource(show_spinner="Chargement du modèle linguistique français...")
def _cached_nlp():
    return load_spacy_model()


@st.cache_resource
def _cached_client():
    return build_client()


def render_response(raw_answer: str, is_blocked: bool) -> None:
    st.subheader("Réponse")
    if is_blocked:
        st.error(SAFETY_MESSAGE)
        with st.expander("Voir la réponse brute du modèle (bloquée)"):
            st.write(raw_answer or "(vide)")
    else:
        st.success(raw_answer or "(vide)")


def render_diagnostic(weakest_prob: float, weakest_token: str, threshold: float, critical_count: int) -> None:
    st.subheader("Diagnostic")
    st.metric("Maillon faible", f"{weakest_prob:.1f}%", help=f"Token : {weakest_token!r}")
    st.metric("Seuil", f"{threshold:.0f}%")
    st.metric("Mots critiques", critical_count)


def render_critical_details(analyses) -> None:
    if not analyses:
        return
    with st.expander("Détails des mots critiques (POS)"):
        st.dataframe(
            [
                {
                    "Token": a.token,
                    "Mot complet": a.word,
                    "Type": a.pos,
                    "Confiance (%)": round(a.probability, 2),
                }
                for a in analyses
            ],
            use_container_width=True,
        )


def run_demo(question: str, model: str, threshold: float) -> None:
    client = _cached_client()
    nlp = _cached_nlp()

    try:
        response = query_model(client, model, question)
    except Exception as exc:
        st.error(
            f"Impossible d'interroger Ollama ({OLLAMA_BASE_URL}). "
            f"Vérifie que le service est lancé (`ollama serve`).\n\n"
            f"Détail : {exc}"
        )
        return

    token_data = response.choices[0].logprobs.content
    raw_answer = response.choices[0].message.content.strip()
    result = find_critical_tokens(token_data, nlp)
    is_blocked = result.weakest_probability < threshold

    col_left, col_right = st.columns([2, 1])
    with col_left:
        render_response(raw_answer, is_blocked)
    with col_right:
        render_diagnostic(
            result.weakest_probability,
            result.weakest_token,
            threshold,
            result.critical_count,
        )

    st.subheader("Probabilités par token")
    fig = plot_token_probabilities(token_data)
    st.pyplot(fig)
    plt.close(fig)

    render_critical_details(result.analyses)


def render_sidebar() -> tuple[str, float]:
    with st.sidebar:
        st.header("Paramètres")
        model = st.text_input("Modèle Ollama", value=DEFAULT_MODEL)
        threshold = st.slider(
            "Seuil de confiance minimum (%)",
            min_value=0,
            max_value=100,
            value=DEFAULT_THRESHOLD,
            step=5,
            help=(
                "Si la probabilité du token critique le plus faible passe sous ce "
                "seuil, la réponse est bloquée et un transfert humain est proposé."
            ),
        )
        st.markdown("---")
        st.caption(
            "Le filtre cible les mots grammaticalement critiques (noms, "
            "nombres, verbes, noms propres) via spaCy `fr_core_news_md`."
        )
    return model, float(threshold)


def main() -> None:
    st.set_page_config(
        page_title="Détecteur d'hallucinations LLM",
        page_icon=":mag:",
        layout="wide",
    )
    st.title("Détecteur d'hallucinations LLM")
    st.caption(
        "Analyse les logprobs pour bloquer les réponses peu fiables avant qu'elles "
        "n'atteignent l'utilisateur."
    )

    model, threshold = render_sidebar()

    question = st.text_area(
        "Pose une question (de préférence factuelle, ex: une statistique précise) :",
        value="Quelle est la population de la France en 2026 ?",
        height=80,
    )

    if st.button("Analyser", type="primary"):
        if not question.strip():
            st.warning("Saisis une question.")
            return
        run_demo(question.strip(), model, threshold)


if __name__ == "__main__":
    main()
