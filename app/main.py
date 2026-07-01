from __future__ import annotations

import base64

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import STATIC_DIR
from app.enrich import BreedEnricher
from app.model import BreedClassifier
from app.profile_store import ProfileStore

app = FastAPI(title="Dog Breed Teacher", version="1.0.0")

classifier = BreedClassifier()
profile_store = ProfileStore()
enricher = BreedEnricher(profile_store)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class Base64ImageRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image from camera capture")


def _decode_base64_image(payload: str) -> bytes:
    cleaned = payload.strip()
    if "," in cleaned:
        cleaned = cleaned.split(",", 1)[1]
    try:
        return base64.b64decode(cleaned, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload") from exc


async def _build_response(prediction: dict) -> dict:
    breed_id = prediction["predicted_breed"]
    profile = None
    if breed_id != "unknown":
        base_profile = profile_store.get_profile(breed_id)
        if base_profile:
            api_data = await enricher.enrich(breed_id)
            profile = profile_store.merge_with_api_data(base_profile, api_data)

    alternatives = []
    for item in prediction["top_k"]:
        alt_profile = profile_store.get_profile(item["breed_id"])
        alternatives.append(
            {
                **item,
                "display_name": alt_profile["display_name"] if alt_profile else item["breed_id"],
            }
        )

    display_name = "Unknown / Low confidence"
    if profile:
        display_name = profile["display_name"]
    elif breed_id != "unknown":
        fallback = profile_store.get_profile(breed_id)
        display_name = fallback["display_name"] if fallback else breed_id

    return {
        "prediction": {
            "predicted_breed": breed_id,
            "display_name": display_name,
            "confidence": prediction["confidence"],
            "is_confident": prediction["is_confident"],
            "top_k": alternatives,
        },
        "profile": profile,
        "disclaimer": (
            "Temperament and behavior information reflects typical breed traits, "
            "not the mood or personality of the individual dog in the photo."
        ),
        "api_enrichment_enabled": enricher.enabled,
    }


@app.get("/")
async def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index_path)


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_ready": classifier.is_ready,
        "api_enrichment_enabled": enricher.enabled,
    }


@app.get("/api/breeds")
async def list_breeds() -> dict:
    return {"breeds": profile_store.list_breeds()}


@app.post("/api/predict")
async def predict_upload(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    try:
        prediction = classifier.predict(image_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return await _build_response(prediction)


@app.post("/api/predict/base64")
async def predict_base64(payload: Base64ImageRequest) -> dict:
    image_bytes = _decode_base64_image(payload.image_base64)
    try:
        prediction = classifier.predict(image_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return await _build_response(prediction)
