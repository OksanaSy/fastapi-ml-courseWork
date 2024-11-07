from fastapi import APIRouter
from pydantic import BaseModel
from gensim.models.doc2vec import Doc2Vec
from sklearn.cluster import KMeans

model = Doc2Vec.load("doc2vec_model.model")

router = APIRouter()

class SentencesInput(BaseModel):
    sentences: list[str]

@router.post("/group/")
async def group_sentences(input: SentencesInput):
    vectors = [model.infer_vector(sentence.split()) for sentence in input.sentences]
    print("Vectors:", vectors)

    n_clusters = min(len(input.sentences), 3)
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(vectors).tolist()

    grouped_sentences = {}
    for label, sentence in zip(labels, input.sentences):
        label = int(label)
        grouped_sentences.setdefault(label, []).append(sentence)

    return {"grouped_sentences": grouped_sentences}
