import os
# 设置代理环境变量（确保在导入其他依赖之前设置）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # 更新后的导入
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM  # 新版 Ollama 模型类

# 定义一个函数，根据问题从向量库中检索相关文档
def retrieve_relevant_documents(vectorstore, question):
    """
    根据问题从向量库中检索相关文档。
    """
    relevant_docs = vectorstore.similarity_search(question)
    relevant_doc_text = "\n".join([doc.page_content for doc in relevant_docs])
    return {"question": question, "relevant_document": relevant_doc_text}

if __name__ == '__main__':
    try:
        # 步骤1：加载单个 Markdown 文件
        file_loader = TextLoader(
            file_path="/Users/xiejiawei/Documents/交接文档/香港/香港环境部署及其配置的账号密码.md",
            encoding="utf-8"
        )
        documents = file_loader.load()
        print(f"加载文档数量: {len(documents)}")

        # 步骤2：拆分文档
        text_splitter = CharacterTextSplitter()
        split_documents = text_splitter.split_documents(documents)

        # 步骤3：创建嵌入（例如使用 all-MiniLM-L6-v2 模型）
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 步骤4：使用 FAISS 构建向量库
        vectorstore = FAISS.from_documents(split_documents, embedding_model)

        # 定义提示模板
        template = """{question}
相关文档: {relevant_document}
回答: 让我们一步一步思考。"""
        prompt = ChatPromptTemplate.from_template(template)

        # 加载新版 Ollama 模型（例如 llama3.1）
        model = OllamaLLM(model="llama3.1")

        # 使用可组合的链构建：将提示模板与 LLM 通过管道符“|”连接
        chain = prompt | model

        # 检索相关文档，并提出问题
        user_question = "告诉我数据库的 IP 地址"
        retrieval_data = retrieve_relevant_documents(vectorstore, user_question)
        result = chain.invoke(retrieval_data)
        print(result)

    except FileNotFoundError as e:
        print(f"指定的文档不存在: {str(e)}")
    except Exception as e:
        print(f"发生其他错误: {str(e)}")
