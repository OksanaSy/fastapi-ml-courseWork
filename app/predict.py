from fastapi import APIRouter
from pydantic import BaseModel
import joblib
from train_model import preprocess_text

model = joblib.load("sentiment_model.joblib")
router = APIRouter()

class TextInput(BaseModel):
    text: str

@router.post("/predict/")
async def predict_sentiment(input: TextInput):
    preprocessed_text = preprocess_text(input.text)
    prediction = model.predict([preprocessed_text])
    sentiment = "positive" if prediction[0] == "positive" else "negative"
    return {"sentiment": sentiment}
