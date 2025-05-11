import os
import pdfplumber
from .llm_agent import CVEvaluator, ParamAgentCVEvaluatorEvaluate, ParamAgentCVEvaluatorTrain
from .emailer import send_email
from dataclasses import dataclass
import tempfile
from fastapi import UploadFile, BackgroundTasks
import asyncio
from typing import Optional
from .llm_train import FineTuningTrainer
import re

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
    evaluation_schema: Optional[str] = None

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
            evaluator.set_prompt("train", payload.evaluation_schema)
            result = evaluator.train(ParamAgentCVEvaluatorTrain(train_prompt=payload.train_prompt))
        else:
            evaluator.set_prompt("", payload.evaluation_schema)
            result = evaluator.evaluate(ParamAgentCVEvaluatorEvaluate(cv_text=extracted_pdf))

        background_tasks.add_task(background_delay)

        print(result)

        if "YES" in result.upper() or ("SCORE:" in result.upper() and  to_score_result(result) > 75) :
            return {"status": "passed", "result": result}
        else:
            return {"status": "rejected", "result": result}

    except Exception as e:
        return {"status": "error", "reason": str(e)}

    finally:
        os.remove(tmp_path)

async def fine_tuning() -> dict:
    try:
        trainer = FineTuningTrainer()
        trainer.fine_tune(dataset)
        return {"status": "success", "result": "success train with fine tuning"}

    except Exception as e:
        return {"status": "error", "reason": str(e)}



async def background_delay():
    print("⏳ Start background task...")
    await asyncio.sleep(1)
    print("✅ Finished after 1 second.")


def to_score_result(result: str) -> int:
    upper_result = result.upper()
    score_match = re.search(r"SCORE:\s*(\d+)", upper_result)
    score = int(score_match.group(1)) if score_match else 0
    return score


dataset = [
    {
        "cv_text": "John has 2 years experience with Golang and Node JS, and attended education in Surabaya.",
        "skills_required": "Golang, Node JS",
        "education_location": "Surabaya",
        "is_suitable": "YES",
        "reason": "Candidate has the required skills and at least 1 year of experience in backend development, plus attended education in Surabaya."
    },
    {
        "cv_text": "Jane has 3 years of frontend experience with React and JavaScript, and attended education in Jakarta.",
        "skills_required": "Golang, Node JS",
        "education_location": "Jakarta",
        "is_suitable": "NO",
        "reason": "Candidate does not have the required skills (Golang, Node JS) and did not attend education in Surabaya."
    },
    {
        "cv_text": "Alice has 1 year of experience working with Node JS but lacks experience with Golang, and attended education in Surabaya.",
        "skills_required": "Golang, Node JS",
        "education_location": "Surabaya",
        "is_suitable": "NO",
        "reason": "Candidate has experience with Node JS, but Golang is required, and attended education in Surabaya."
    },
    {
        "cv_text": "Bob has 5 years of experience working with Golang and Node JS, and attended education in Surabaya.",
        "skills_required": "Golang, Node JS",
        "education_location": "Surabaya",
        "is_suitable": "YES",
        "reason": "Candidate has 5 years of experience with the required skills and attended education in Surabaya."
    }
]