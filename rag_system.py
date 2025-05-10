import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.schema import Document

# 加载环境变量
load_dotenv()

class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vector_store = None
        self.qa_chain = None
        self.book_metadata = {}  # 存储书籍元数据

    def load_and_process_document(self, pdf_path: str, book_name: str = None):
        """加载PDF文档并进行处理"""
        if not book_name:
            book_name = os.path.basename(pdf_path)
        
        print(f"正在加载PDF文档: {book_name}...")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # 为每个文档添加元数据
        for page in pages:
            page.metadata["book_name"] = book_name
        
        print("正在分割文档...")
        chunks = self.text_splitter.split_documents(pages)
        
        # 更新书籍元数据
        self.book_metadata[book_name] = {
            "chunk_count": len(chunks),
            "total_pages": len(pages)
        }
        
        print("正在创建/更新向量存储...")
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
        else:
            self.vector_store.add_documents(chunks)
        
        print("正在初始化QA链...")
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3})
        )
        
        print(f"成功加载书籍: {book_name}")
        print(f"总页数: {len(pages)}")
        print(f"文本块数量: {len(chunks)}")

    def list_books(self) -> Dict:
        """列出所有已加载的书籍信息"""
        return self.book_metadata

    def query(self, question: str, book_name: str = None) -> str:
        """查询系统"""
        if not self.qa_chain:
            raise ValueError("请先调用load_and_process_document()方法加载至少一本书")
        
        # 如果指定了特定书籍，添加过滤条件
        if book_name:
            if book_name not in self.book_metadata:
                raise ValueError(f"未找到书籍: {book_name}")
            
            retriever = self.vector_store.as_retriever(
                search_kwargs={
                    "k": 3,
                    "filter": {"book_name": book_name}
                }
            )
            self.qa_chain.retriever = retriever
        
        response = self.qa_chain.invoke({"query": question})
        return response["result"]

def main():
    # 初始化RAG系统
    rag = RAGSystem()
    
    # 加载PDF文档
    pdf_path = "[The Morgan Kaufmann Series in Computer Architecture and Design] John L. Hennessy, David A. Patterson - Computer Architecture, Sixth Edition_ A Quantitative Approach (2017, Morgan Kaufmann).pdf"
    rag.load_and_process_document(pdf_path)
    
    print("欢迎使用RAG问答系统！")
    print("您可以询问任何关于计算机架构的问题。")
    print("输入'quit'退出系统。")
    
    while True:
        question = input("\n请输入您的问题: ")
        if question.lower() == 'quit':
            break
            
        answer = rag.query(question)
        print("\n回答:", answer)

if __name__ == "__main__":
    main() 