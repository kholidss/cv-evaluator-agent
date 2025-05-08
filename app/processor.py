import pdfplumber
from .llm_agent import chain
from .emailer import send_email

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def evaluate_cv(file_path: str, user_email: str) -> dict:
    cv_text = extract_text_from_pdf(file_path)
    if not cv_text:
        return {"status": "failed", "reason": "Could not extract text from PDF"}

    # FIXED: input to chain must be a dict
    result = chain.invoke({"cv_text": cv_text})

    if "YES" in result.upper():
        # send_email(
        #     user_email,
        #     "Congratulations!",
        #     "Your CV meets our criteria. Welcome aboard!"
        # )
        return {"status": "passed", "result": result}
    else:
        print("GA LOLOS BOS!")
        return {"status": "rejected", "result": result}
