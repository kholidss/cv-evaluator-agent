from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# Inisialisasi LLM dari langchain_ollama
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

# Contoh pemanggilan (opsional)
# result = chain.invoke({"cv_text": "John Doe, Bachelor of Arts, 2 years as Customer Service..."})
# print(result)
