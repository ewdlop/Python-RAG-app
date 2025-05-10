# RAG问答系统

这是一个基于RAG（检索增强生成）技术的问答系统，专门用于处理计算机架构教材的内容。

## 功能特点

- PDF文档加载和处理
- 文本分块和向量化
- 基于相似度的文档检索
- 使用GPT-3.5进行问答生成

## 安装步骤

1. 克隆仓库
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

运行主程序：
```bash
python rag_system.py
```

系统启动后，您可以输入任何关于计算机架构的问题，系统会基于教材内容给出回答。

## 系统要求

- Python 3.8+
- Llama 2模型
- 足够的磁盘空间用于存储向量数据库

## 注意事项

- 首次运行时会需要一些时间来处理PDF文档
- 确保有稳定的网络连接以访问OpenAI API
- 向量数据库将保存在`./chroma_db`目录中 

## 下载Llama 2模型

```shell
mkdir -p models && cd models && curl -L -o llama-2-7b-chat.Q4_K_M.gguf https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```
