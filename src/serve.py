import io
import yaml
from pathlib import Path
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torchvision.transforms as transforms
from src.model import get_model

app = FastAPI(title="CIFAR-10 PyTorch Serving API")


model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

@app.on_event("startup")
def load_resources():
    """Loads the model checkpoint on application startup."""
    global model
    
    
    config_path = Path("/app/configs/training_config.yaml")
    if not config_path.exists():
        config_path = Path("configs/training_config.yaml")
        
    try:
        config = load_config(str(config_path))
        
        
        model = get_model(
            architecture=config["model"]["architecture"],
            num_classes=config["model"]["num_classes"]
        ).to(device)
        
        
        checkpoint_dir = Path(config["output"]["checkpoint_dir"])
        model_path = checkpoint_dir / config["output"]["model_name"]
        
        if model_path.exists():
            
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"Warning: No checkpoint found at {model_path}. Please run train.py first.")
            model = None
            
    except Exception as e:
        print(f"Error during startup configuration: {e}")
        model = None

@app.get("/health")
def health_check():
    """Returns 200 OK only if the model loaded successfully."""
    if model is not None:
        return {"status": "ok", "message": "Model loaded and ready for predictions."}
    raise HTTPException(status_code=503, detail="Service unavailable: Model not loaded.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accepts an image upload and returns class probabilities."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        
        image = image.resize((32, 32))
        
        input_tensor = eval_transform(image).unsqueeze(0).to(device)
        
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
    
        probs_dict = {class_names[i]: round(float(probabilities[0][i]), 4) for i in range(len(class_names))}
        top_prob, top_class = torch.max(probabilities, 1)
        
        return {
            "predicted_class": class_names[top_class.item()],
            "confidence": round(float(top_prob.item()), 4),
            "probabilities": probs_dict
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prediction: {str(e)}")