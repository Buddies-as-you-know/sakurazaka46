import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import prediction as pd

app = FastAPI()
origins = [
    "http://localhost:3000",
    "https://rug6ws.csb.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def hello_world():
    return f"Hello,World"

@app.post('/api/predict')
def predict_image(file: UploadFile = File(...)):
    print(file)
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png","gif")
    if not extension:
        return "Image must be jpg,png and gif format!"
    image = pd.read_image(save_upload_file_tmp(file))
    image = pd.preprocess(image)
    pred = pd.predict(image)
    print(pred)
    return pred

def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
        tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')
