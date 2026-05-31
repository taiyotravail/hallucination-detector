"""Constantes globales du détecteur d'hallucinations."""

OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen:0.5b"

CRITICAL_POS_TAGS = ("PROPN", "NOUN", "NUM", "VERB")

SAFETY_MESSAGE = (
    "Je ne dispose pas de cette information avec une fiabilité suffisante. "
    "Je transfère votre demande à un opérateur humain."
)

DEFAULT_THRESHOLD = 70
