from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableSequence
from dataclasses import dataclass

@dataclass
class ParamAgentCVEvaluatorEvaluate:
    cv_text: str

@dataclass
class ParamAgentCVEvaluatorTrain:
    train_prompt: str

class CVEvaluator:
    def __init__(self, model_name: str = "gemma3:1b"):
        self.llm = OllamaLLM(model=model_name)
        self.prompt = PromptTemplate
        self.evaluation_prompt = """Respond with "YES" or "NO" and give a brief reason."""

    def evaluate(self, param: ParamAgentCVEvaluatorEvaluate) -> str:
        chain: RunnableSequence = self.prompt | self.llm
        return chain.invoke({
            "cv_text": param.cv_text,
            "evaluation_prompt": self.evaluation_prompt
        })
    
    def train(self, param: ParamAgentCVEvaluatorTrain) -> str:
        chain: RunnableSequence = self.prompt | self.llm
        return chain.invoke({
            "train_prompt": param.train_prompt
    })

    def set_prompt(self, type: str, evaluation_schema: str = ""):
        if type == "train":
            self.prompt = ChatPromptTemplate.from_messages([
                HumanMessagePromptTemplate.from_template("{train_prompt}")
            ])
            return

        if evaluation_schema == "score":
            self.evaluation_prompt = """Respond begin with "SCORE:0-100" based on the Requirement Points and give a brief reason."""

        self.prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You are a CV evaluation assistant."),
                HumanMessagePromptTemplate.from_template("""
                   The following is a candidate's CV:

                    {cv_text}

                    Please evaluate whether this candidate is suitable or relate for the Software Engineer, Backend Engineer job position.
                    "Requirements":
                    1. Candidate must have a minimum of 2 years of total working experience or more. If the candidate has less than 2 years, they should be automatically considered NOT suitable, regardless of other criteria.
                    2. Must have included all of this skills (Golang, Node JS)
                    3. Must have attended education in Manado
                    4. Don't check the education level major or years graduation

                    {evaluation_prompt}
                    """)
                    ])
            