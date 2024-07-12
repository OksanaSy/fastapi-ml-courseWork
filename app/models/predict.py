from fastapi import FastAPI, HTTPException
from textdistance import Cosine, Jaccard, Levenshtein
from typing import Union

from app.api.api import api_router
from app.api.heartbeat import heartbeat_router
from app.core.config import settings
from app.core.event_handler import start_app_handler, stop_app_handler
from app.models.predict import SimilarityMethod, SimilarityRequest

app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(heartbeat_router)
app.include_router(api_router, prefix=settings.API_V1_STR, tags=["ML API"])

app.add_event_handler("startup", start_app_handler(app, settings.MODEL_PATH))
app.add_event_handler("shutdown", stop_app_handler(app))

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

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
