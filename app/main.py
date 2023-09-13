from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse
from typing import List
import numpy as np
import onnxruntime
import cv2
from pydantic_settings import BaseSettings
import torchvision.transforms as transforms
from PIL import Image
import io

class AppConfig(BaseSettings):
    app_name: str = "Cat vs Dog Prediction"
    version: str = "0.1"

appconfig = AppConfig()

# Initialize your FastAPI app
app = FastAPI(title = appconfig.app_name, 
              version = appconfig.version,
              description="A demo backend app for Efficientnetv2b2 model serving with FastApi.")

ort_session_pretrained = onnxruntime.InferenceSession("./ONNX/Efficientnetv2b2Pre.onnx")
ort_session_scratch = onnxruntime.InferenceSession("./ONNX/Efficientnetv2b2Scr.onnx")

classes = {0: 'Cat', 1: 'Dog'}

# Define mean and std values for ImageNet dataset
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224,224)),    # Resize the image to 224x224
    transforms.ToTensor(),           # Convert image to PyTorch tensor
    transforms.Normalize(mean, std)  # Normalize the tensor (ImageNet statistics)
])


@app.get("/")
def index():
    return appconfig.app_name

@app.post("/pretrained-model-predict/")
async def pretrained_model_predict(file: UploadFile):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        input_data = transform(image)
        ort_inputs = {ort_session_pretrained.get_inputs()[0].name: np.array(input_data).reshape(1, 3, 224, 224).astype(np.float32)}
        ort_outputs = ort_session_pretrained.get_outputs()[0].name
        ort_outs = ort_session_pretrained.run([ort_outputs], ort_inputs)

        predictions = ort_outs[0]
        predicted_class = np.argmax(predictions)
        return classes[int(predicted_class)]

    except Exception as ex:
        return JSONResponse(content={"error": str(ex)}, status_code=500)
    
@app.post("/scratch-model-predict/")
async def scratch_model_predict(file: UploadFile):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        input_data = transform(image)
        ort_inputs = {ort_session_scratch.get_inputs()[0].name: np.array(input_data).reshape(1, 3, 224, 224).astype(np.float32)}
        ort_outputs = ort_session_scratch.get_outputs()[0].name
        ort_outs = ort_session_scratch.run([ort_outputs], ort_inputs)

        predictions = ort_outs[0]
        predicted_class = np.argmax(predictions)
        return classes[int(predicted_class)]

    except Exception as ex:
        return JSONResponse(content={"error": str(ex)}, status_code=500)
    

## docker run -d -p 8000:80 fastapi-app
## docker-compose up -d
## docker-compose down
