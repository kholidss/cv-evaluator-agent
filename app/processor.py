import pdfplumber
from .llm_agent import CVEvaluator, ParamCVEvaluatorEvaluate
from .emailer import send_email

def extract_text_from_pdf(file_path: str) -> str:
    print("file_path ===>> ", file_path)
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def evaluate_cv(file_path: str, user_email: str) -> dict:
    extracted_pdf = extract_text_from_pdf(file_path)
    if not extracted_pdf:
        return {"status": "failed", "reason": "Could not extract text from PDF"}

    evaluator = CVEvaluator()
    result = evaluator.evaluate(ParamCVEvaluatorEvaluate(cv_text=extracted_pdf))

    if "YES" in result.upper():
        # send_email(
        #     user_email,
        #     "Congratulations!",
        #     "Your CV meets our criteria. Welcome aboard!"
        # )
        return {"status": "passed", "result": result}
    else:
        return {"status": "rejected", "result": result}
