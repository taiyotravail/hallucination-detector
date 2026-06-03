# Détecteur d'hallucinations LLM

> Comment empêcher une IA d'halluciner avec aplomb quand elle ne sait pas ?

Démo Streamlit qui analyse les **logprobs** d'un LLM local pour détecter les
réponses peu fiables avant qu'elles n'atteignent l'utilisateur. Si la confiance
du modèle tombe sous un seuil défini, la réponse est bloquée et remplacée par un
message de transfert vers un opérateur humain.

Cible : **legaltech, finance, santé, assurance** — tous les secteurs où une IA
qui invente avec assurance n'est pas une option.

## Le problème

Un LLM ne dit jamais "je ne sais pas". Quand il manque d'information, il
hallucine avec la même assurance que lorsqu'il connaît la réponse. Exemple
réel avec `qwen:0.5b` sur la question *"Selon le rapport officiel de l'ARCEP, 
quel était le nombre exact, à l'unité près, de foyers raccordables à la fibre optique (FttH) en France au premier trimestre 2021?"* :

> 1600 foyers.

Le nombre est faux. Pourtant le modèle l'affirme
sans broncher.

## L'approche

Plutôt que de demander au LLM s'il est sûr de lui (il dira toujours oui), on
inspecte directement les **probabilités de chaque token généré** :

1. On demande au modèle de retourner ses `logprobs` (top 3 alternatives par token).
2. On reconstruit le texte et on l'analyse avec spaCy (`fr_core_news_md`).
3. On filtre les tokens correspondant à des **mots grammaticalement critiques** :
   noms (`NOUN`), noms propres (`PROPN`), nombres (`NUM`), verbes (`VERB`).
4. Le **maillon faible** = la probabilité la plus basse parmi ces mots critiques.
5. Si maillon faible < seuil (par défaut **70 %**) → la réponse est bloquée.

Le filtre POS est crucial : la ponctuation, les articles et autres mots
fonctionnels ont souvent une confiance basse sans que ce soit problématique.
On ne veut bloquer que sur du contenu factuel.

## Prérequis

- Python 3.13+
- [Ollama](https://ollama.com/) installé localement
- Un modèle Ollama

## Installation

```bash
git clone <repo-url>
cd logprobs
python -m venv .venv
source .venv/bin/activate
pip install streamlit openai spacy matplotlib
python -m spacy download fr_core_news_md
ollama pull qwen:0.5b
```

## Lancement

Dans un terminal, démarrer le serveur Ollama :

```bash
ollama serve
```

Dans un autre terminal, lancer l'app :

```bash
streamlit run app.py
```

L'interface est accessible sur http://localhost:8501.

## Utilisation

1. Ajuster le seuil de confiance dans la sidebar (défaut : 70 %).
2. Saisir une question — idéalement piège (statistique précise, fait pointu).
3. Cliquer sur **Analyser**.
4. L'app affiche :
   - La réponse du modèle (ou le message de sécurité si seuil non respecté)
   - Le diagnostic : maillon faible, seuil, nombre de mots critiques
   - Un scatter plot des top-3 probabilités par token
   - Le détail des mots critiques avec leur catégorie grammaticale

## Architecture

Découpage modulaire (cf. standards `CLAUDE.md`) :

```
.
├── app.py                   # UI Streamlit (orchestration uniquement)
├── config.py                # constantes (URL, modèle, seuil, message)
├── llm_client.py            # client Ollama (API compat OpenAI)
├── confidence_analyzer.py   # logique métier : POS + maillon faible
├── plots.py                 # visualisations matplotlib
└── logprobs.ipynb           # notebook d'exploration initial
```

Les modules `confidence_analyzer` et `plots` n'ont aucune dépendance vis-à-vis
de Streamlit : ils sont réutilisables (API REST, batch jobs, tests unitaires).

## Limites connues

- Le filtre POS dépend de la qualité du tagger spaCy ; sur du texte mal formé
  produit par un petit modèle, les tags peuvent être imprécis (mais ça reste
  exploitable car les *mauvais* tokens sont précisément ceux qui rendent la
  phrase incohérente).
- Les modèles fermés (Anthropic, OpenAI) exposent les logprobs avec des
  restrictions variables ; le code utilise l'API compat OpenAI d'Ollama mais
  doit être adapté pour d'autres backends.
- Le seuil optimal dépend du modèle et du domaine — à calibrer sur un jeu
  d'évaluation propre au cas d'usage.

## Pistes d'extension

- Empaqueter `confidence_analyzer` dans une API FastAPI pour intégration prod.
- Calibrer automatiquement le seuil sur un dataset annoté (vraies vs. fausses
  réponses).
- Ajouter un mode "streaming" pour bloquer la réponse dès qu'un token critique
  passe sous le seuil, sans attendre la fin de la génération.
- Tester d'autres modèles Ollama pour comparer les profils de confiance.
