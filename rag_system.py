import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain.schema import Document

# 加载环境变量
load_dotenv()

class RAGSystem:
    def __init__(self):
        # 使用本地嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256,
            chunk_overlap=32,
            length_function=len
        )
        self.vector_store = None
        self.qa_chain = None
        self.book_metadata = {}  # 存储书籍元数据

    def load_and_process_document(self, pdf_path: str, book_name: str = None):
        """加载PDF文档并进行处理"""
        if not book_name:
            book_name = os.path.basename(pdf_path)
        
        # 检查向量数据库是否已存在
        if os.path.exists("./chroma_db"):
            print("检测到已存在的向量数据库，正在加载...")
            self.vector_store = Chroma(
                persist_directory="./chroma_db",
                embedding_function=self.embeddings
            )
            # 从向量存储中获取所有文档的元数据
            all_docs = self.vector_store.get()
            if all_docs and all_docs['metadatas']:
                # 统计每本书的块数
                for metadata in all_docs['metadatas']:
                    book_name = metadata.get('book_name')
                    if book_name:
                        if book_name not in self.book_metadata:
                            self.book_metadata[book_name] = {"chunk_count": 0}
                        self.book_metadata[book_name]["chunk_count"] += 1
            print("向量数据库加载完成")
        else:
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
            
            print("正在创建向量存储...")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
        
        print("正在初始化QA链...")
        # 使用本地LLM
        llm = LlamaCpp(
            model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
            temperature=0.7,
            max_tokens=256,
            n_ctx=512,
            top_p=1,
            verbose=True,
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 1})
        )
        
        print(f"成功加载书籍: {book_name}")
        if book_name in self.book_metadata:
            print(f"文本块数量: {self.book_metadata[book_name]['chunk_count']}")

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
                    "k": 1,
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