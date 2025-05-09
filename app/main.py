from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from .processor import evaluate_cv

app = FastAPI()

@app.post("/submit_cv/")
async def submit_cv(file: UploadFile = Form(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=422, detail="File must be a PDF.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        tmp_path = tmp.name

    try:
        result = evaluate_cv(tmp_path, "email")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "failed", "reason": str(e)}
        )
    finally:
        os.remove(tmp_path)

    return JSONResponse(status_code=200, content=result)
