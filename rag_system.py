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
    
    # 示例：加载多本书籍
    books = [
        {
            "path": "[The Morgan Kaufmann Series in Computer Architecture and Design] John L. Hennessy, David A. Patterson - Computer Architecture, Sixth Edition_ A Quantitative Approach (2017, Morgan Kaufmann).pdf",
            "name": "计算机架构：量化研究方法"
        }
        # 可以在这里添加更多书籍
    ]
    
    # 加载所有书籍
    for book in books:
        try:
            rag.load_and_process_document(book["path"], book["name"])
        except Exception as e:
            print(f"加载书籍 {book['name']} 时发生错误: {str(e)}")
    
    # 显示已加载的书籍
    print("\n已加载的书籍:")
    for book_name, metadata in rag.list_books().items():
        print(f"- {book_name}: {metadata['total_pages']}页, {metadata['chunk_count']}个文本块")
    
    # 交互式问答
    print("\n欢迎使用RAG问答系统！")
    print("可用命令:")
    print("- 'list': 列出所有已加载的书籍")
    print("- 'quit': 退出系统")
    print("- 输入问题开始查询")
    
    while True:
        user_input = input("\n请输入您的问题或命令: ")
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'list':
            print("\n已加载的书籍:")
            for book_name, metadata in rag.list_books().items():
                print(f"- {book_name}: {metadata['total_pages']}页, {metadata['chunk_count']}个文本块")
            continue
            
        try:
            # 检查是否指定了特定书籍
            if " in " in user_input.lower():
                question, book_name = user_input.split(" in ", 1)
                answer = rag.query(question.strip(), book_name.strip())
            else:
                answer = rag.query(user_input)
            
            print("\n回答:", answer)
        except Exception as e:
            print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main() 