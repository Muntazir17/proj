from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse
import uvicorn
import shutil

app = FastAPI()


@app.post("/fastapi", response_class=RedirectResponse)
async def redirect_fastapi():
    return "/image"


@app.post("/image")
async def image(image: UploadFile = File(...)):
    with open("api/files/destination.png", "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    return {"filename": image.filename}


if __name__ == '__main__':
    uvicorn.run(app)
