from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from ml_service import predict_delay

app = FastAPI()


# ══════════════════════════════════════════════════════
# INPUT SCHEMA
# ══════════════════════════════════════════════════════
class Order(BaseModel):
    order_date: str
    expected_delivery_date: str
    actual_delivery_date: str


class PredictRequest(BaseModel):
    history: List[Order]


# ══════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════
@app.get("/")
def home():
    return {
        "message": "Supply Chain Delay Prediction API is running"
    }


# ══════════════════════════════════════════════════════
# PREDICTION ENDPOINT
# ══════════════════════════════════════════════════════
@app.post("/predict")
def predict(request: PredictRequest):

    try:
        # ─────────────────────────────
        # VALIDATION
        # ─────────────────────────────
        if not request.history or len(request.history) < 5:
            return {
                "error": "Not enough history",
                "message": "Minimum 5 orders required"
            }

        # ─────────────────────────────
        # CONVERT TO DICT
        # ─────────────────────────────
        history = [order.model_dump() for order in request.history]

        # ─────────────────────────────
        # CALL ML SERVICE
        # ─────────────────────────────
        result = predict_delay(history)

        return result

    except Exception as e:
        return {
            "error": "Prediction failed",
            "details": str(e)
        }