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

    def evaluate(self, param: ParamAgentCVEvaluatorEvaluate) -> str:
        chain: RunnableSequence = self.prompt | self.llm
        return chain.invoke({
            "cv_text": param.cv_text
        })
    
    def train(self, param: ParamAgentCVEvaluatorTrain) -> str:
        chain: RunnableSequence = self.prompt | self.llm
        return chain.invoke({
            "train_prompt": param.train_prompt
    })

    def set_prompt(self, type: str):
        if type == "train":
            self.prompt = ChatPromptTemplate.from_messages([
                HumanMessagePromptTemplate.from_template("{train_prompt}")
            ])
            return

        self.prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You are a CV evaluation assistant."),
                HumanMessagePromptTemplate.from_template("""
                   The following is a candidate's CV:

                    {cv_text}

                    Please evaluate whether this candidate is suitable for the Software Engineer, Backend Engineer job position.
                    "Requirements":
                    1. Minimum 1 year of total working experience (only check year working experience, not year education experience)
                    2. Must have included all of this skills (Golang, Node JS)
                    3. Must have attended education in Surabaya
                    4. Don't check the education level major or years graduation

                    Respond with YES or NO and give a brief reason.
                    """)
                    ])
            