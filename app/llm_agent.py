from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

llm = OllamaLLM(model="gemma")

# Template prompt
prompt = PromptTemplate(
    input_variables=["cv_text"],
    template="""
The following is a candidate's CV:

{cv_text}

Please evaluate whether this candidate is suitable for the 'Virtual Assistant' position.
Requirements:
- At least a Bachelor's degree
- Minimum 1 year of experience
- Strong communication and attention to detail

Respond with YES or NO and give a brief reason.
"""
)

chain: RunnableSequence = prompt | llm

