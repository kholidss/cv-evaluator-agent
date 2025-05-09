import os
import pdfplumber
from .llm_agent import CVEvaluator, ParamAgentCVEvaluatorEvaluate, ParamAgentCVEvaluatorTrain
from .emailer import send_email
from dataclasses import dataclass
import tempfile
from fastapi import UploadFile, BackgroundTasks
import asyncio
from typing import Optional

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


@dataclass
class ParamEvaluateCV:
    pdf_file: UploadFile
    user_email: Optional[str] = None
    train_prompt: Optional[str] = None

async def evaluate_cv(payload: ParamEvaluateCV, background_tasks: BackgroundTasks) -> dict:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(await payload.pdf_file.read())
            tmp.flush()
            tmp_path = tmp.name

        extracted_pdf = extract_text_from_pdf(tmp_path)
        if not extracted_pdf:
            return {"status": "failed", "reason": "Could not extract text from PDF"}

        evaluator = CVEvaluator()
        if payload.train_prompt:
            evaluator.set_prompt("train")
            result = evaluator.train(ParamAgentCVEvaluatorTrain(train_prompt=payload.train_prompt))
        else:
            evaluator.set_prompt("")
            result = evaluator.evaluate(ParamAgentCVEvaluatorEvaluate(cv_text=extracted_pdf))

        background_tasks.add_task(background_delay)

        if "YES" in result.upper():
            return {"status": "passed", "result": result}
        else:
            return {"status": "rejected", "result": result}

    except Exception as e:
        return {"status": "error", "reason": str(e)}

    finally:
        os.remove(tmp_path)


async def background_delay():
    print("⏳ Start background task...")
    await asyncio.sleep(1)
    print("✅ Finished after 1 second.")
