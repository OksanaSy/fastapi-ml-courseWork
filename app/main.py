from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from app.predict import router as predict_router
from app.group import router as group_router
from textdistance import Cosine, Jaccard, Levenshtein
from train_doc2vec import preprocess_text
from preprocessing_spacy import preprocess_text_spacy
from app.api.api import api_router
from app.api.heartbeat import heartbeat_router
from app.core.config import settings
from app.core.event_handler import start_app_handler, stop_app_handler
from app.models.predict import SimilarityMethod, SimilarityRequest, SimilarityResponse
import spacy

nlp = spacy.load("en_core_web_sm")

app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(heartbeat_router)
app.include_router(predict_router)
app.include_router(group_router)
app.include_router(api_router, prefix=settings.API_V1_STR, tags=["ML API"])

app.add_event_handler("startup", start_app_handler(app, settings.MODEL_PATH))
app.add_event_handler("shutdown", stop_app_handler(app))


class TextInput(BaseModel):
    text: str


@app.post("/preprocess/")
async def preprocess(input: TextInput, method: str = Query("spacy", enum=["spacy", "default"])):
    if method == "spacy":
        preprocessed_text = preprocess_text_spacy(input.text)
    else:
        preprocessed_text = preprocess_text(input.text)

    if isinstance(preprocessed_text, list):
        preprocessed_text = " ".join(preprocessed_text)

    cleaned_text = " ".join(preprocessed_text.split())
    return {"preprocessed_text": cleaned_text}



METHOD_TO_FUNCTION = {
    SimilarityMethod.cosine: Cosine(qval=2),
    SimilarityMethod.jaccard: Jaccard(qval=2),
    SimilarityMethod.levenshtein: Levenshtein(),
}


@app.post("/calculate_similarity")
async def calculate_similarity(request: SimilarityRequest) -> SimilarityResponse:
    method = request.method
    line1 = request.line1
    line2 = request.line2

    if method not in METHOD_TO_FUNCTION:
        raise HTTPException(status_code=400, detail=f"Метод '{method}' не підтримується.")

    similarity_function = METHOD_TO_FUNCTION[method]
    similarity_value = similarity_function.normalized_similarity(line1, line2)

    response = SimilarityResponse(
        method=method,
        line1=line1,
        line2=line2,
        similarity=similarity_value
    )
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
