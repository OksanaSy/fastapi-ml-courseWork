from enum import Enum
from pydantic import BaseModel, Field, StrictStr


class PredictRequest(BaseModel):
    input_text: StrictStr = Field(..., title="input_text", description="Input text", example="Input text for ML")


class PredictResponse(BaseModel):
    result: float = Field(..., title="result", description="Predict value", example=0.9)

class SimilarityMethod(str, Enum):
    cosine = "cosine"
    jaccard = "jaccard"
    levenshtein = "levenshtein"


class SimilarityRequest(BaseModel):
    method: SimilarityMethod
    line1: str
    line2: str


class SimilarityResponse(BaseModel):
    method: SimilarityMethod
    line1: str
    line2: str
    similarity: float
