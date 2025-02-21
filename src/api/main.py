from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
import joblib
from pathlib import Path
from skimage.color import rgb2lab, deltaE_cie76
import logging
from src.predict.predict_rf import ColorPredictionSystem

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()

# Enable CORS to allow frontend (HTML/JS) to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端域访问（可指定特定域名，如 ["http://localhost:5500"]）
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有请求方法（如 GET, POST）
    allow_headers=["*"],  # 允许所有请求头
)

logging.info("CORS settings applied. API is now accessible from any origin.")

# Define root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
logging.info(f"Root directory set to: {ROOT_DIR}")

# Load model
model_path = ROOT_DIR / "data/models/random_forest/model_v1.pkl"

if not model_path.exists():
    logging.error(f"Model file not found at: {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")

logging.info(f"Loading model from: {model_path}")
model = joblib.load(model_path)

# Initialize color prediction system
predictor = ColorPredictionSystem(model)
logging.info("ColorPredictionSystem initialized successfully.")

def process_image(image_bytes):
    """Processes the uploaded image and predicts color."""
    try:
        logging.info("Decoding image from byte stream...")
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            logging.error("Failed to decode image. Possible corrupt or unsupported format.")
            return {"error": "Invalid image format. Ensure you upload a valid PNG or JPG file."}

        logging.info("Image successfully decoded. Running color prediction...")

        # Perform color prediction
        captured_color, predicted_color = predictor.predict_true_color(image)

        # ✅ Convert numpy.int64 to Python int
        captured_color = [int(x) for x in captured_color]
        predicted_color = [int(x) for x in predicted_color]

        logging.info(f"Prediction completed: Captured Color={captured_color}, Predicted Color={predicted_color}")
        return captured_color, predicted_color

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return {"error": f"Image processing failed: {str(e)}"}

@app.post("/predict/")
async def predict_color(file: UploadFile = File(...)):
    """Receives an image, processes it, and returns color predictions."""
    try:
        logging.info(f"Received file: {file.filename}")

        image_bytes = await file.read()
        captured_color, predicted_color = process_image(image_bytes)

        # If processing fails, return an error response
        if isinstance(captured_color, dict) and "error" in captured_color:
            raise HTTPException(status_code=400, detail=captured_color["error"])

        return {
            "captured_color": captured_color,
            "predicted_color": predicted_color
        }

    except Exception as e:
        logging.error(f"Error handling request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    logging.info("Starting FastAPI server on port 8080...")  
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)  # listen 8080 port for google cloud run



