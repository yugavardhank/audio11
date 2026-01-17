import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

def generate_titles(topics):
    docs = []

    for t in topics:
        doc = nlp(t["text"])
        phrases = [
            chunk.text.lower()
            for chunk in doc.noun_chunks
            if len(chunk.text.split()) <= 4
        ]
        docs.append(" ".join(phrases))

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5)
    X = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names_out()

    for i, t in enumerate(topics):
        scores = X[i].toarray()[0]
        top = scores.argsort()[-2:][::-1]
        t["title"] = " / ".join(terms[j] for j in top if scores[j] > 0).title()

    return topics