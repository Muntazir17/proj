from fastapi import FastAPI, Request, UploadFile, File
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import tensorflow as tf
import disease

IMAGE_SIZE = 256
# CLASS_NAMES = ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Septoria Leaf Spot',
#                'Spider Mites (Two-spotted_spider_mite)', 'Target Spot', 'Yellow Leaf Curl Virus',
#                'Mosaic Virus', 'Healthy']

# CLASS_NAMES = ['Grape___.Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#                'Grape___healthy']

CLASS_NAMES = ['Cotton___bacterial_blight', 'Cotton___curl_virus', 'Cotton___healthy', 'Tomato___Late_blight', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy']

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
MODEL = tf.keras.models.load_model("../saved_models/model_v2.h5")


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/home", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("About.html", {"request": request})


@app.get("/contact", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image1 = Image.open(BytesIO(data))
    image1.save("static/predict.jpg")
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    converted_image = np.array(img).astype('float32')
    return converted_image


# noinspection PyCompatibility
@app.post('/predict', response_class=HTMLResponse)
async def predict(request: Request, files: UploadFile = File(...)):
    image = read_file_as_image(await files.read())
    image_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    index = np.argmax(predictions[0])
    desc = disease.disease[index]
    prev = disease.prevention[index]
    confidence_ = np.max(predictions[0])
    confidence = round(confidence_ * 100, 2)
    return templates.TemplateResponse("predict.html",
                                      {"request": request, "label": predicted_class, "confidence": confidence,
                                       "desc": desc, "prev": prev})


if __name__ == '__main__':
    # noinspection PyTypeChecker
    uvicorn.run(app, host='localhost', port=8000)
