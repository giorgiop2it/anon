#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from html import escape

# Caricamento modello NER
tokenizer = AutoTokenizer.from_pretrained("DeepMount00/Italian_NER_XXL")
model = AutoModelForTokenClassification.from_pretrained("DeepMount00/Italian_NER_XXL", ignore_mismatched_sizes=True)
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Categorie e colori per le entità
ENTITY_COLORS = {
    "INDIRIZZO": "#FFCCCC",
    "VALUTA": "#FF9999",
    "CVV": "#FF6666",
    "NUMERO_CONTO": "#FF3333",
    "BIC": "#FF0000",
    "IBAN": "#CCFFCC",
    "STATO": "#99FF99",
    "NOME": "#66FF66",
    "COGNOME": "#33FF33",
    "CODICE_POSTALE": "#00FF00",
    "IP": "#CCCCFF",
    "ORARIO": "#9999FF",
    "URL": "#6666FF",
    "LUOGO": "#CCCCFF",
    "IMPORTO": "#66FFFF",
    "EMAIL": "#FFCC99",
    "PASSWORD": "#FF9966",
    "NUMERO_CARTA": "#FF6633",
    "TARGA_VEICOLO": "#FF3300",
    "DATA_NASCITA": "#FFFF99",
    "DATA_MORTE": "#FFFF66",
    "RAGIONE_SOCIALE": "#FFFF33",
    "ETA": "#FFFF00",
    "DATA": "#CCFFFF",
    "PROFESSIONE": "#99FFFF",
    "PIN": "#66FFFF",
    "NUMERO_TELEFONO": "#33FFFF",
    "FOGLIO": "#00FFFF",
    "PARTICELLA": "#FFCCFF",
    "CARTELLA_CLINICA": "#FF99FF",
    "MALATTIA": "#FF66FF",
    "MEDICINA": "#FF33FF",
    "CODICE_FISCALE": "#FF00FF",
    "NUMERO_DOCUMENTO": "#CC99FF",
    "STORIA_CLINICA": "#9966FF",
    "AVV_NOTAIO": "#6633FF",
    "P_IVA": "#66FFFF",
    "LEGGE": "#CCCCCC",
    "TASSO_MUTUO": "#999999",
    "N_SENTENZA": "#666666",
    "MAPPALE": "#FFCCFF",
    "SUBALTERNO": "#33FFF",
}

# Funzione per evidenziare il testo con colori
def highlight_entities(text, entities):
    annotated_text = text
    for entity in sorted(entities, key=lambda x: x["start"], reverse=True):
        color = ENTITY_COLORS.get(entity["entity"], "#E0E0E0")
        start, end, label = entity["start"], entity["end"], entity["entity"]
        annotated_text = (
            annotated_text[:start]
            + f'<span style="background-color:{color}; padding:2px; border-radius:4px;">{escape(text[start:end])} ({label})</span>'
            + annotated_text[end:]
        )
    return annotated_text

# Funzione per ottenere le entità dal testo
def extract_entities(text):
    ner_results = nlp(text)
    grouped_entities = []
    current_entity = {"word": "", "entity": "", "start": 0, "end": 0}

    for entity in ner_results:
        entity_type = entity["entity"].split("-")[-1]
        prefix = entity["entity"].split("-")[0]
        word = entity["word"].replace("##", "")
        if prefix == "B":
            if current_entity["word"]:
                grouped_entities.append(current_entity)
            current_entity = {"word": word, "entity": entity_type, "start": entity["start"], "end": entity["end"]}
        elif prefix == "I" and entity_type == current_entity["entity"]:
            current_entity["word"] += f" {word}"
            current_entity["end"] = entity["end"]
        else:
            if current_entity["word"]:
                grouped_entities.append(current_entity)
            current_entity = {"word": word, "entity": entity_type, "start": entity["start"], "end": entity["end"]}
    if current_entity["word"]:
        grouped_entities.append(current_entity)
    return grouped_entities

# Funzione per anonimizzare il testo
def anonymize_text(text, entities):
    anonymized_text = text
    for entity in sorted(entities, key=lambda x: x["start"], reverse=True):
        start, end, label = entity["start"], entity["end"], entity["entity"]
        anonymized_text = anonymized_text[:start] + f"[{label}]" + anonymized_text[end:]
    return anonymized_text

# Streamlit App
def main():
    st.title("Anonimizzazione del Testo con Evidenziazione delle Entità")
    
    # Legenda
    with st.expander("Legenda delle Categorie"):
        st.write("Il modello è capace di identificare le seguenti categorie:")
        for category, color in ENTITY_COLORS.items():
            st.markdown(f"<span style='background-color:{color}; padding:4px; border-radius:4px;'>{category}</span>", unsafe_allow_html=True)
    
    # Input testo
    text = st.text_area("Inserisci il testo da processare (max 2000 caratteri):", height=200, max_chars=2000)
    
    if st.button("Elabora"):
        if text.strip():
            entities = extract_entities(text)
            highlighted_text = highlight_entities(text, entities)
            anonymized_text = anonymize_text(text, entities)
            
            st.subheader("Testo Originale con Entità Evidenziate:")
            st.markdown(highlighted_text, unsafe_allow_html=True)
            
            st.subheader("Testo Anonimizzato:")
            st.text(anonymized_text)
        else:
            st.warning("Inserisci un testo valido!")

if __name__ == "__main__":
    main()
