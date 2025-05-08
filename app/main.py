from fastapi import FastAPI, UploadFile, Form
import shutil
import uuid
import os
from .processor import evaluate_cv

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/submit_cv/")
async def submit_cv(file: UploadFile, email: str = Form(...)):
    filename = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = evaluate_cv(file_path, email)
    return result
