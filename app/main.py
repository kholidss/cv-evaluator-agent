from fastapi import FastAPI, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from .processor import evaluate_cv, fine_tuning, ParamEvaluateCV

app = FastAPI()

@app.post("/submit_cv/")
async def submit_cv(file: UploadFile = Form(...), train_prompt: str = Form(None), evaluation_schema: str = Form(None), background_tasks: BackgroundTasks = BackgroundTasks()):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=422, detail="File must be a PDF.")

    try:
        result = await evaluate_cv(ParamEvaluateCV(pdf_file=file, train_prompt=train_prompt, evaluation_schema=evaluation_schema), background_tasks)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "failed", "reason": str(e)}
        )

    return JSONResponse(status_code=200, content=result)

@app.post("/tuning/")
async def submit_cv():
    try:
        result = await fine_tuning()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "failed", "reason": str(e)}
        )

    return JSONResponse(status_code=200, content=result)
