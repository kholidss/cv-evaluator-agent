from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from dataclasses import dataclass

@dataclass
class ParamCVEvaluatorEvaluate:
    cv_text: str

class CVEvaluator:
    def __init__(self, model_name: str = "gemma3:1b"):
        self.llm = OllamaLLM(model=model_name)
        self.prompt = PromptTemplate(
            input_variables=["cv_text"],
            template="""
The following is a candidate's CV:

{cv_text}

Please evaluate whether this candidate is suitable for the 'Software Engineer' or 'Backend Engineer' position.
Requirements:
- Minimum 1 year of experience (if more, consider as pass at this point)
- Must have Golang skills (if more, consider as pass at this point)
- Must have attended education in Indonesia

Respond with YES or NO and give a brief reason.
"""
        )
        self.chain: RunnableSequence = self.prompt | self.llm

    def evaluate(self, param: ParamCVEvaluatorEvaluate) -> str:
        return self.chain.invoke({
            "cv_text": param.cv_text
        })