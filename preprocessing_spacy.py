import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_text_spacy(text):
    text = " ".join(text.split())

    doc = nlp(text)

    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]

    print(f"Tokens after lemmatization: {tokens}")

    clean_text = " ".join(tokens)

    return clean_text.strip()
