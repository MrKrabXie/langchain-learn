from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM



if __name__ == '__main__':
    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = ChatPromptTemplate.from_template(template)

    model = OllamaLLM(model="llama3.1")

    chain = prompt | model

    raw_result = chain.invoke({"question": "What is LangChain?"})
    print(raw_result)