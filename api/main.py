
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference import InferenceService

app = FastAPI(
    title="Mammography Risk Classification API",
    description="API for classifying breast cancer risk from mammogram images.",
    version="1.0.0"
)

# Initialize service (Singleton pattern for model loading)
# In production, might want to load this on startup event
service = None

@app.on_event("startup")
async def startup_event():
    global service
    try:
        service = InferenceService()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: Model could not be loaded: {e}")
        # We might still run for documentation purposes, but endpoints will fail

@app.get("/")
def read_root():
    return {"message": "Welcome to the Mammography Risk Classification API"}

@app.post("/predict")
async def predict_risk(file: UploadFile = File(...)):
    if not service:
        raise HTTPException(status_code=503, detail="Model service not initialized.")
    
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        result = service.predict(image)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
