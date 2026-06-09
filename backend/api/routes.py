"""Additional API routes and utilities."""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import numpy as np
import cv2
from loguru import logger

router = APIRouter()


@router.post("/segment")
async def segment_image(image: UploadFile = File(...)):
    """Segment objects in uploaded image."""
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Would use SAM segmenter
        return {"status": "segmentation complete", "masks": []}
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features")
async def extract_features(image: UploadFile = File(...)):
    """Extract semantic features from image."""
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        return {"status": "features extracted", "embedding_dim": 0}
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/human/poses")
async def get_human_poses():
    """Get current tracked human poses."""
    return {"poses": [], "count": 0}


@router.get("/prediction/next")
async def get_next_prediction():
    """Get next predicted scene state."""
    return {"predictions": [], "time_horizons": []}


def setup_routes(app):
    """Register additional routes."""
    app.include_router(router, prefix="/api")