from fastapi import FastAPI # type: ignore
from pydantic import BaseModel # type: ignore
from transformers import pipeline

# Initialize FastAPI
app = FastAPI()

# Load a BERT model fine-tuned for sentiment analysis (replace with a bias-detection model later)
classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Define the request schema
class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_text(request: TextRequest):
    text = request.text

    # Run the text through BERT
    result = classifier(text)[0]
    label = result["label"]

    # Convert labels to bias categories
    if "1 star" in label or "2 star" in label:
        bias = "Right-Wing"
        color = "blue"
    elif "4 star" in label or "5 star" in label:
        bias = "Left-Wing"
        color = "red"
    else:
        bias = "Neutral"
        color = "yellow"

    return {
        "text": text,
        "bias": bias,
        "color": color,
        "confidence": result["score"]
    }
