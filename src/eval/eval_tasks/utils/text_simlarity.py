import spacy
from sklearn.metrics.pairwise import cosine_similarity

# python -m spacy download en_core_web_md

# Load a pre-trained Word2Vec model from spaCy
nlp = spacy.load("en_core_web_md")

def match_by_similarity(sentence1, sentence2, threshold=0.5):
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    vector1 = doc1.vector
    vector2 = doc2.vector
    similarity = cosine_similarity([vector1], [vector2])[0][0]
    if similarity >= threshold:
        print(f"Sentence 1: {sentence1}, Sentence 2: {sentence2}, Similarity: {similarity}")
        return True
    else:
        print(f"Sentence 1: {sentence1}, Sentence 2: {sentence2}, Similarity: {similarity}")
        return False