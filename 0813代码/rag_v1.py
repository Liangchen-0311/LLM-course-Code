"""
多跳问答RAG框架 - 详细实现文档

项目概述：
本框架解决了传统RAG在多跳问答场景中的痛点，包括检索结果冗余率高、信息丢失、
检索效率低下等问题。通过层次化检索、子问题改写、网络检索补充等技术，
显著提升了复杂问题的处理能力。

核心特性：
1. 智能子问题分解与改写
2. 层次化检索策略（BM25检索 + Contriever精排）
3. 网络检索补充（Google Serper API）
4. COT-SC提示词优化
5. 多代理架构支持
6. 长短期记忆管理
7. 文件系统工具集成

作者：[您的姓名]
版本：v1.1
日期：2024
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import pickle
from pathlib import Path

# 核心依赖
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModel
import torch
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.schema import BaseMemory
import openai
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI

# 新增BM25和Contriever相关导入
from rank_bm25 import BM25Okapi
import jieba

# 添加新的导入
from intent_recognition_feature_extraction import ZhipuElementExtractor
from db_field_recall import BGE_FieldMatcher, create_intent_elements_from_extracted_elements

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SubQuestion:
    """子问题数据结构"""
    id: str
    original_question: str
    rewritten_question: str
    context: str
    answer: str
    confidence: float
    retrieval_quality: float
    sources: List[str]

@dataclass
class MemoryItem:
    """记忆项数据结构"""
    id: str
    content: str
    timestamp: datetime
    importance: float
    context: str
    memory_type: str  # 'short' or 'long'

@dataclass
class Document:
    """文档数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class Config:
    """配置管理类"""
    
    def __init__(self):
        self.openai_api_key = "sk-"
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        self.model_name = "THUDM/glm-4-9b-chat"
        self.max_tokens = 4000
        self.temperature = 0.1
        self.retrieval_top_k = 20  # 增加BM25检索数量
        self.rerank_top_k = 5      # Contriever重排序数量
        self.memory_window_size = 10
        self.max_memory_items = 1000
        
        # BM25配置
        self.bm25_k1 = 1.5         # BM25参数k1
        self.bm25_b = 0.75         # BM25参数b
        
        # Contriever配置
        self.contriever_model_name = "facebook/contriever-msmarco"
        self.max_sequence_length = 512
        
        # FAISS配置（保留用于兼容性）
        self.embedding_model_name = "BAAI/bge-small-zh-v1.5"
        self.embedding_cache_folder = "D:/developer/vscode_workspace/llm/models"
        self.faiss_index_path = "./faiss_index"
        self.vector_dimension = 384
        
        # 文件系统配置
        self.workspace_dir = Path("./workspace")
        self.memory_dir = Path("./memory")
        self.cache_dir = Path("./cache")
        
        # 创建必要目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录结构"""
        for dir_path in [self.workspace_dir, self.memory_dir, self.cache_dir]:
            dir_path.mkdir(exist_ok=True)

class FAISSIndexManager:
    """FAISS索引管理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.embedding_model = self._init_embedding_model()
        self.index = self._init_faiss_index()
        self.documents = self._load_documents()
        self.document_ids = []
    
    def _init_embedding_model(self) -> HuggingFaceBgeEmbeddings:
        """初始化嵌入模型"""
        try:
            model = HuggingFaceBgeEmbeddings(
                model_name=self.config.embedding_model_name,
                cache_folder=self.config.embedding_cache_folder
            )
            logger.info(f"嵌入模型加载成功: {self.config.embedding_model_name}")
            return model
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise
    
    def _init_faiss_index(self) -> faiss.Index:
        """初始化FAISS索引"""
        try:
            index_path = Path(self.config.faiss_index_path)
            if index_path.exists():
                # 加载现有索引
                index = faiss.read_index(str(index_path))
                logger.info("FAISS索引加载成功")
                return index
            else:
                # 创建新索引
                dimension = self.config.vector_dimension
                index = faiss.IndexFlatIP(dimension)  # 内积索引，用于余弦相似度
                logger.info(f"创建新的FAISS索引，维度: {dimension}")
                return index
        except Exception as e:
            logger.error(f"FAISS索引初始化失败: {e}")
            # 降级到简单的Flat索引
            dimension = self.config.vector_dimension
            return faiss.IndexFlatIP(dimension)
    
    def _load_documents(self) -> List[Document]:
        """加载文档"""
        documents_file = Path(self.config.faiss_index_path).parent / "documents.pkl"
        if documents_file.exists():
            try:
                with open(documents_file, 'rb') as f:
                    documents = pickle.load(f)
                    self.document_ids = [doc.id for doc in documents]
                    logger.info(f"加载了 {len(documents)} 个文档")
                    return documents
            except Exception as e:
                logger.error(f"加载文档失败: {e}")
        
        # 如果没有现有文档，创建一些示例文档
        sample_documents = self._create_sample_documents()
        self.document_ids = [doc.id for doc in sample_documents]
        return sample_documents
    
    def _create_sample_documents(self) -> List[Document]:
        """创建示例文档"""
        sample_texts = [
            "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
            "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的学习过程。",
            "自然语言处理是人工智能的一个领域，专注于计算机理解和生成人类语言的能力。",
            "计算机视觉是人工智能的一个分支，使计算机能够从图像和视频中获取信息。",
            "强化学习是机器学习的一种方法，通过与环境交互来学习最优策略。",
            "神经网络是深度学习的基础，模拟生物神经元的连接结构。",
            "大数据分析是处理和分析大量数据以发现模式和趋势的过程。",
            "云计算提供了按需访问计算资源的能力，支持AI系统的部署和扩展。",
            "边缘计算将计算能力带到数据源附近，减少延迟并提高隐私性。"
        ]
        
        documents = []
        for i, text in enumerate(sample_texts):
            doc = Document(
                id=f"doc_{i+1}",
                content=text,
                metadata={"source": "sample", "category": "AI"}
            )
            documents.append(doc)
        
        return documents
    
    def add_documents(self, documents: List[Document]):
        """添加文档到索引"""
        if not documents:
            return
        
        # 生成嵌入向量
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)
        embeddings = np.array(embeddings)
        
        # 添加到FAISS索引
        self.index.add(embeddings.astype('float32'))
        
        # 更新文档列表
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
            self.documents.append(doc)
            self.document_ids.append(doc.id)
        
        logger.info(f"添加了 {len(documents)} 个文档到索引")
        self._save_index()
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """搜索文档"""
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.embed_query(query)
            query_embedding = np.array([query_embedding])
            
            # 搜索
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # 返回结果
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append((doc, float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS搜索失败: {e}")
            return self._mock_search(query, top_k)
    
    def _mock_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """模拟搜索"""
        results = []
        for i, doc in enumerate(self.documents[:top_k]):
            # 简单的文本相似度
            similarity = self._calculate_text_similarity(query, doc.content)
            results.append((doc, similarity))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0
    
    def _save_index(self):
        """保存索引"""
        try:
            index_path = Path(self.config.faiss_index_path)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存FAISS索引
            faiss.write_index(self.index, str(index_path))
            
            # 保存文档
            documents_file = index_path.parent / "documents.pkl"
            with open(documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
            
            logger.info("索引保存成功")
        except Exception as e:
            logger.error(f"保存索引失败: {e}")

class BM25IndexManager:
    """BM25索引管理器 - 第一阶段检索"""
    
    def __init__(self, config: Config):
        self.config = config
        self.documents = self._load_documents()
        self.bm25_index = self._init_bm25_index()
    
    def _load_documents(self) -> List[Document]:
        """加载文档"""
        documents_file = Path(self.config.faiss_index_path).parent / "documents.pkl"
        if documents_file.exists():
            try:
                with open(documents_file, 'rb') as f:
                    documents = pickle.load(f)
                    logger.info(f"BM25加载了 {len(documents)} 个文档")
                    return documents
            except Exception as e:
                logger.error(f"BM25加载文档失败: {e}")
        
        # 如果没有现有文档，创建示例文档
        sample_documents = self._create_sample_documents()
        return sample_documents
    
    def _create_sample_documents(self) -> List[Document]:
        """创建示例文档"""
        sample_texts = [
            "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
            "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的学习过程。",
            "自然语言处理是人工智能的一个领域，专注于计算机理解和生成人类语言的能力。",
            "计算机视觉是人工智能的一个分支，使计算机能够从图像和视频中获取信息。",
            "强化学习是机器学习的一种方法，通过与环境交互来学习最优策略。",
            "神经网络是深度学习的基础，模拟生物神经元的连接结构。",
            "大数据分析是处理和分析大量数据以发现模式和趋势的过程。",
            "云计算提供了按需访问计算资源的能力，支持AI系统的部署和扩展。",
            "边缘计算将计算能力带到数据源附近，减少延迟并提高隐私性。"
        ]
        
        documents = []
        for i, text in enumerate(sample_texts):
            doc = Document(
                id=f"doc_{i+1}",
                content=text,
                metadata={"source": "sample", "category": "AI"}
            )
            documents.append(doc)
        
        return documents
    
    def _init_bm25_index(self) -> BM25Okapi:
        """初始化BM25索引"""
        try:
            # 对文档进行分词
            tokenized_docs = []
            for doc in self.documents:
                # 使用jieba进行中文分词
                tokens = list(jieba.cut(doc.content))
                tokenized_docs.append(tokens)
            
            # 创建BM25索引
            bm25 = BM25Okapi(tokenized_docs, k1=self.config.bm25_k1, b=self.config.bm25_b)
            logger.info("BM25索引初始化成功")
            return bm25
            
        except Exception as e:
            logger.error(f"BM25索引初始化失败: {e}")
            # 降级处理：使用简单的空格分词
            tokenized_docs = []
            for doc in self.documents:
                tokens = doc.content.split()
                tokenized_docs.append(tokens)
            return BM25Okapi(tokenized_docs)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """BM25搜索"""
        try:
            # 对查询进行分词
            query_tokens = list(jieba.cut(query))
            
            # 计算BM25分数
            scores = self.bm25_index.get_scores(query_tokens)
            
            # 获取top_k结果
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            # 返回结果
            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    score = float(scores[idx])
                    results.append((doc, score))
            
            return results
            
        except Exception as e:
            logger.error(f"BM25搜索失败: {e}")
            return self._mock_search(query, top_k)
    
    def _mock_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """模拟搜索"""
        results = []
        for i, doc in enumerate(self.documents[:top_k]):
            # 简单的文本相似度
            similarity = self._calculate_text_similarity(query, doc.content)
            results.append((doc, similarity))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0

class ContrieverReranker:
    """Contriever重排序器 - 第二阶段精排"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer, self.model = self._init_contriever()
    
    def _init_contriever(self):
        """初始化Contriever模型"""
        try:
            # 从本地路径加载contriever-msmarco模型
            local_model_path = r"D:/developer/vscode_workspace/llm/models/contriever-msmarco"
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModel.from_pretrained(local_model_path)
            logger.info(f"Contriever模型加载成功: {local_model_path}")
            return tokenizer, model
        except Exception as e:
            logger.warning(f"Contriever模型初始化失败: {e}")
            return None, None
    
    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """使用Contriever进行重排序"""
        if not self.tokenizer or not self.model or not candidates:
            return candidates
        
        try:
            # 准备查询-文档对
            scored_candidates = []
            for candidate in candidates:
                content = candidate.get("content", "")
                # 限制文本长度以避免超出模型限制
                if len(content) > 400:
                    content = content[:400]
                
                # 使用Contriever计算相似度分数
                score = self._calculate_contriever_score(query, content)
                
                scored_candidates.append({
                    **candidate,
                    "contriever_score": score
                })
            
            # 按Contriever分数排序
            scored_candidates.sort(key=lambda x: x["contriever_score"], reverse=True)
            
            return scored_candidates
            
        except Exception as e:
            logger.error(f"Contriever重排序失败: {e}")
            return candidates
    
    def _calculate_contriever_score(self, query: str, content: str) -> float:
        """计算Contriever相似度分数"""
        try:
            # 编码查询和文档
            query_encoding = self.tokenizer(
                query, 
                padding=True, 
                truncation=True, 
                max_length=self.config.max_sequence_length,
                return_tensors='pt'
            )
            
            content_encoding = self.tokenizer(
                content, 
                padding=True, 
                truncation=True, 
                max_length=self.config.max_sequence_length,
                return_tensors='pt'
            )
            
            # 获取嵌入向量
            with torch.no_grad():
                query_embeddings = self.model(**query_encoding).last_hidden_state.mean(dim=1)
                content_embeddings = self.model(**content_encoding).last_hidden_state.mean(dim=1)
                
                # 计算余弦相似度
                similarity = torch.cosine_similarity(query_embeddings, content_embeddings, dim=1)
                
                return float(similarity.item())
                
        except Exception as e:
            logger.error(f"计算Contriever分数失败: {e}")
            # 降级到简单的文本相似度
            return self._calculate_text_similarity(query, content)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（降级方法）"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0

class QuestionDecomposer:
    """问题分解器 - 将复杂问题分解为子问题"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = self._init_llm()
        
    def _init_llm(self):
        """初始化语言模型"""
        return ChatOpenAI(
            temperature=0,
            model="THUDM/glm-4-9b-chat",  
            openai_api_key="sk-",
            openai_api_base="https://api.siliconflow.cn/v1",
            max_retries=0,
        )
    
    def decompose_question(self, question: str) -> List[str]:
        """将复杂问题分解为子问题"""
        
        prompt = f"""
        请将以下复杂问题分解为2-4个相关的子问题。每个子问题应该：
        1. 能够独立回答
        2. 信息完整，不会丢失关键信息
        3. 逻辑清晰，便于检索
        
        复杂问题：{question}
        
        请以JSON格式返回子问题列表：
        {{
            "sub_questions": [
                "子问题1",
                "子问题2",
                "子问题3"
            ]
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response.content)
            return result.get("sub_questions", [])
            
        except Exception as e:
            logger.error(f"问题分解失败: {e}")
            return [question]  # 降级处理

class QuestionRewriter:
    """问题改写器 - 改善子问题的检索效果"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = self._init_llm()
    
    def _init_llm(self):
        """初始化语言模型"""
        return ChatOpenAI(
            temperature=0,
            model="THUDM/glm-4-9b-chat",  
            openai_api_key="sk-",
            openai_api_base="https://api.siliconflow.cn/v1",
            max_retries=0,
        )
    
    def rewrite_question(self, sub_question: str, context: str = "") -> str:
        """改写子问题以改善检索效果"""
        
        prompt = f"""
        请改写以下子问题，使其更适合检索：
        
        原始子问题：{sub_question}
        上下文信息：{context}
        
        改写要求：
        1. 保持原问题的核心含义
        2. 添加更多检索关键词
        3. 明确化模糊概念
        4. 结构化查询意图
        
        改写后的子问题：
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"问题改写失败: {e}")
            return sub_question  # 降级处理

class HierarchicalRetriever:
    """层次化检索器 - 结合BM25检索和Contriever精排"""
    
    def __init__(self, config: Config):
        self.config = config
        self.bm25_manager = BM25IndexManager(config)  # 第一阶段：BM25
        self.contriever_reranker = ContrieverReranker(config)  # 第二阶段：Contriever
        self.serper_client = SerperClient(config)
    
    def bm25_retrieval(self, question: str) -> List[Dict]:
        """BM25检索 - 第一阶段"""
        try:
            # 使用BM25搜索
            search_results = self.bm25_manager.search(question, self.config.retrieval_top_k)
            
            # 转换为标准格式
            results = []
            for doc, score in search_results:
                results.append({
                    "content": doc.content,
                    "source": doc.metadata.get("source", "bm25"),
                    "score": score,
                    "metadata": doc.metadata
                })
            
            return results
            
        except Exception as e:
            logger.error(f"BM25检索失败: {e}")
            return self._mock_retrieval(question)
    
    def contriever_rerank(self, question: str, candidates: List[Dict]) -> List[Dict]:
        """Contriever重排序 - 第二阶段"""
        if not candidates:
            return candidates
        
        try:
            # 使用Contriever进行重排序
            reranked_results = self.contriever_reranker.rerank(question, candidates)
            
            # 返回top_k结果
            return reranked_results[:self.config.rerank_top_k]
            
        except Exception as e:
            logger.error(f"Contriever重排序失败: {e}")
            return candidates
    
    def _mock_retrieval(self, question: str) -> List[Dict]:
        """模拟检索结果"""
        return [
            {
                "content": f"关于'{question}'的相关信息",
                "source": "mock_source_1",
                "score": 0.8
            },
            {
                "content": f"'{question}'的详细说明",
                "source": "mock_source_2", 
                "score": 0.7
            }
        ]
    
    def retrieve(self, question: str) -> Tuple[List[Dict], float]:
        """执行层次化检索"""
        # 第一阶段：BM25检索
        bm25_results = self.bm25_retrieval(question)
        
        # 第二阶段：Contriever重排序
        reranked_results = self.contriever_rerank(question, bm25_results)
        
        # 评估检索质量
        retrieval_quality = self._assess_retrieval_quality(question, reranked_results)
        
        return reranked_results, retrieval_quality
    
    def _assess_retrieval_quality(self, question: str, results: List[Dict]) -> float:
        """评估检索质量"""
        if not results:
            return 0.0
        
        # 计算平均分数
        scores = []
        for result in results:
            # 优先使用Contriever分数，如果没有则使用BM25分数
            if "contriever_score" in result:
                scores.append(result["contriever_score"])
            else:
                scores.append(result.get("score", 0.5))
        
        avg_score = sum(scores) / len(scores)
        coverage_score = min(len(results) / self.config.retrieval_top_k, 1.0)
        
        return (avg_score + coverage_score) / 2

class SerperClient:
    """Google Serper API客户端 - 网络检索补充"""
    
    def __init__(self, config: Config):
        self.config = config
        self.api_key = config.serper_api_key
        self.base_url = "https://google.serper.dev/search"
    
    def search(self, query: str) -> List[Dict]:
        """执行网络搜索"""
        if not self.api_key:
            return self._mock_web_search(query)
        
        try:
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "q": query,
                "num": 5
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            return self._parse_search_results(data)
            
        except Exception as e:
            logger.error(f"网络搜索失败: {e}")
            return self._mock_web_search(query)
    
    def _parse_search_results(self, data: Dict) -> List[Dict]:
        """解析搜索结果"""
        results = []
        
        if "organic" in data:
            for item in data["organic"][:5]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "source": "web_search"
                })
        
        return results
    
    def _mock_web_search(self, query: str) -> List[Dict]:
        """模拟网络搜索结果"""
        return [
            {
                "title": f"关于'{query}'的网络搜索结果",
                "snippet": f"这是关于'{query}'的相关网络信息",
                "link": "https://example.com",
                "source": "web_search"
            }
        ]

class MemoryManager:
    """记忆管理器 - 长短期记忆管理"""
    
    def __init__(self, config: Config):
        self.config = config
        self.short_term_memory = ConversationBufferWindowMemory(
            k=self.config.memory_window_size,
            return_messages=True
        )
        self.long_term_memory = self._load_long_term_memory()
        self.memory_file = self.config.memory_dir / "long_term_memory.pkl"
    
    def _load_long_term_memory(self) -> List[MemoryItem]:
        """加载长期记忆"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'rb') as f:
                    return pickle.load(f)
            return []
        except Exception as e:
            logger.error(f"加载长期记忆失败: {e}")
            return []
    
    def _save_long_term_memory(self):
        """保存长期记忆"""
        try:
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.long_term_memory, f)
        except Exception as e:
            logger.error(f"保存长期记忆失败: {e}")
    
    def add_short_term_memory(self, content: str, context: str = ""):
        """添加短期记忆"""
        self.short_term_memory.save_context(
            {"input": content},
            {"output": context}
        )
    
    def add_long_term_memory(self, content: str, importance: float = 0.5, context: str = ""):
        """添加长期记忆"""
        memory_item = MemoryItem(
            id=self._generate_memory_id(content),
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            context=context,
            memory_type="long"
        )
        
        self.long_term_memory.append(memory_item)
        
        # 限制长期记忆数量
        if len(self.long_term_memory) > self.config.max_memory_items:
            self._prune_long_term_memory()
        
        self._save_long_term_memory()
    
    def _generate_memory_id(self, content: str) -> str:
        """生成记忆ID"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _prune_long_term_memory(self):
        """裁剪长期记忆"""
        # 按重要性排序，保留最重要的记忆
        self.long_term_memory.sort(key=lambda x: x.importance, reverse=True)
        self.long_term_memory = self.long_term_memory[:self.config.max_memory_items]
    
    def get_relevant_memories(self, query: str, limit: int = 5) -> List[MemoryItem]:
        """获取相关记忆"""
        relevant_memories = []
        
        for memory in self.long_term_memory:
            # 简单的相关性计算
            relevance = self._calculate_relevance(query, memory.content)
            if relevance > 0.3:  # 相关性阈值
                relevant_memories.append((memory, relevance))
        
        # 按相关性排序
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in relevant_memories[:limit]]
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """计算相关性"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        intersection = query_words.intersection(content_words)
        return len(intersection) / len(query_words) if query_words else 0.0

class FileSystemTool:
    """文件系统工具 - 统一文件操作接口"""
    
    def __init__(self, config: Config):
        self.config = config
        self.workspace = config.workspace_dir
        self.cache = config.cache_dir
    
    def read_file(self, file_path: str) -> str:
        """读取文件"""
        try:
            full_path = self.workspace / file_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logger.warning(f"文件不存在: {file_path}")
                return ""
        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            return ""
    
    def write_file(self, file_path: str, content: str) -> bool:
        """写入文件"""
        try:
            full_path = self.workspace / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"写入文件失败: {e}")
            return False
    
    def list_files(self, directory: str = "") -> List[str]:
        """列出文件"""
        try:
            dir_path = self.workspace / directory
            if dir_path.exists():
                return [str(f.relative_to(self.workspace)) for f in dir_path.rglob("*") if f.is_file()]
            return []
        except Exception as e:
            logger.error(f"列出文件失败: {e}")
            return []
    
    def delete_file(self, file_path: str) -> bool:
        """删除文件"""
        try:
            full_path = self.workspace / file_path
            if full_path.exists():
                full_path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"删除文件失败: {e}")
            return False

class PlanTool:
    """计划工具 - 优化Agent执行效果"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = self._init_llm()
    
    def _init_llm(self):
        """初始化语言模型"""
        return ChatOpenAI(
            temperature=0,
            model="THUDM/glm-4-9b-chat",  
            openai_api_key="sk-",
            openai_api_base="https://api.siliconflow.cn/v1",
            max_retries=0,
        )
    
    def create_execution_plan(self, task: str, context: str = "") -> Dict:
        """创建执行计划"""
        
        prompt = f"""
        请为以下任务创建一个详细的执行计划：
        
        任务：{task}
        上下文：{context}
        
        请创建包含以下内容的执行计划：
        1. 任务分解
        2. 执行步骤
        3. 所需资源
        4. 预期结果
        5. 风险评估
        
        以JSON格式返回：
        {{
            "task_decomposition": ["步骤1", "步骤2", ...],
            "execution_steps": ["具体操作1", "具体操作2", ...],
            "required_resources": ["资源1", "资源2", ...],
            "expected_results": ["结果1", "结果2", ...],
            "risk_assessment": ["风险1", "风险2", ...]
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            return json.loads(response.content)
            
        except Exception as e:
            logger.error(f"创建执行计划失败: {e}")
            return {
                "task_decomposition": [task],
                "execution_steps": ["执行任务"],
                "required_resources": [],
                "expected_results": ["完成任务"],
                "risk_assessment": []
            }
    
    def optimize_plan(self, plan: Dict, feedback: str) -> Dict:
        """优化执行计划"""
        
        prompt = f"""
        请根据以下反馈优化执行计划：
        
        原始计划：{json.dumps(plan, ensure_ascii=False)}
        反馈：{feedback}
        
        请返回优化后的计划：
        """
        
        try:
            response = self.llm.invoke(prompt)
            return json.loads(response.content)
            
        except Exception as e:
            logger.error(f"优化计划失败: {e}")
            return plan

class ReportGenerationAgent:
    """报告生成代理"""
    
    def __init__(self, config: Config, memory_manager: MemoryManager, file_tool: FileSystemTool):
        self.config = config
        self.memory_manager = memory_manager
        self.file_tool = file_tool
        self.llm = self._init_llm()
    
    def _init_llm(self):
        """初始化语言模型"""
        return ChatOpenAI(
            temperature=0,
            model="THUDM/glm-4-9b-chat",  
            openai_api_key="sk-",
            openai_api_base="https://api.siliconflow.cn/v1",
            max_retries=0,
        )
    
    def generate_report(self, topic: str, requirements: str = "") -> str:
        """生成报告"""
        
        # 获取相关记忆
        relevant_memories = self.memory_manager.get_relevant_memories(topic)
        memory_context = "\n".join([m.content for m in relevant_memories])
        
        prompt = f"""
        请生成一份关于"{topic}"的专业报告。
        
        要求：{requirements}
        相关背景信息：{memory_context}
        
        报告结构：
        1. 执行摘要
        2. 背景介绍
        3. 主要内容
        4. 分析结果
        5. 结论和建议
        
        请生成结构清晰、内容专业的报告。
        """
        
        try:
            response = self.llm.invoke(prompt)
            report = response.content
            
            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{topic}_{timestamp}.md"
            self.file_tool.write_file(filename, report)
            
            # 添加到长期记忆
            self.memory_manager.add_long_term_memory(
                f"生成了关于{topic}的报告",
                importance=0.8,
                context=topic
            )
            
            return report
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            return f"报告生成失败: {e}"

class FileProcessingAgent:
    """文件处理代理"""
    
    def __init__(self, config: Config, file_tool: FileSystemTool):
        self.config = config
        self.file_tool = file_tool
        self.llm = self._init_llm()
    
    def _init_llm(self):
        """初始化语言模型"""
        return ChatOpenAI(
            temperature=0,
            model="THUDM/glm-4-9b-chat",  
            openai_api_key="sk-",
            openai_api_base="https://api.siliconflow.cn/v1",
            max_retries=0,
        )
    
    def process_file(self, file_path: str, operation: str) -> str:
        """处理文件"""
        
        content = self.file_tool.read_file(file_path)
        if not content:
            return f"无法读取文件: {file_path}"
        
        prompt = f"""
        请对以下文件内容执行"{operation}"操作：
        
        文件路径：{file_path}
        操作：{operation}
        文件内容：
        {content}
        
        请返回处理结果。
        """
        
        try:
            response = self.llm.invoke(prompt)
            result = response.content
            
            # 保存处理结果
            output_path = f"processed_{file_path}"
            self.file_tool.write_file(output_path, result)
            
            return result
            
        except Exception as e:
            logger.error(f"文件处理失败: {e}")
            return f"文件处理失败: {e}"

class SQLGenerationAgent:
    """SQL生成代理 - 基于要素提取和字段召回的SQL生成"""
    
    def __init__(self, config: Config):
        self.config = config
        self.extractor = ZhipuElementExtractor(api_key=config.openai_api_key)
        self.matcher = BGE_FieldMatcher()
        self.sql_templates = self._generate_sql_templates()
    
    def _generate_sql_templates(self) -> Dict:
        """生成SQL模板"""
        prompt = """
你是一个SQL专家，请为以下六种常见的SQL查询类型，分别给出标准的SQL模板（用大括号占位符），每种类型只给出模板，不要举例，不要多余解释，返回JSON格式：

{
  "query": "SELECT {select_fields} FROM {main_table} {where_clause}",
  "calculation": "SELECT {aggregation}({select_fields}) FROM {main_table} {where_clause}",
  "statistics": "SELECT {group_by_fields}, {aggregation}({select_fields}) FROM {main_table} {where_clause} GROUP BY {group_by_fields}",
  "group_by": "SELECT {group_by_fields}, {aggregation}({select_fields}) FROM {main_table} {where_clause} GROUP BY {group_by_fields}",
  "subquery": "SELECT {select_fields} FROM {main_table} WHERE {field} {operator} (SELECT {sub_select_fields} FROM {sub_table} {sub_where_clause})",
  "join": "SELECT {select_fields} FROM {main_table} JOIN {join_table} ON {join_condition} {where_clause}"
}
"""
        try:
            # 创建临时的LLM实例
            llm = ChatOpenAI(
                temperature=0,
                model="THUDM/glm-4-9b-chat",  
                openai_api_key=self.config.openai_api_key,
                openai_api_base="https://api.siliconflow.cn/v1",
                max_retries=0,
            )
            response = llm.invoke(prompt)
            
            # 解析JSON
            start = response.content.find('{')
            end = response.content.rfind('}') + 1
            json_str = response.content[start:end]
            sql_templates = json.loads(json_str)
            return sql_templates
            
        except Exception as e:
            logger.error(f"生成SQL模板失败: {e}")
            # 返回默认模板
            return {
                "query": "SELECT {select_fields} FROM {main_table} {where_clause}",
                "calculation": "SELECT {aggregation}({select_fields}) FROM {main_table} {where_clause}",
                "statistics": "SELECT {group_by_fields}, {aggregation}({select_fields}) FROM {main_table} {where_clause} GROUP BY {group_by_fields}",
                "group_by": "SELECT {group_by_fields}, {aggregation}({select_fields}) FROM {main_table} {where_clause} GROUP BY {group_by_fields}",
                "subquery": "SELECT {select_fields} FROM {main_table} WHERE {field} {operator} (SELECT {sub_select_fields} FROM {sub_table} {sub_where_clause})",
                "join": "SELECT {select_fields} FROM {main_table} JOIN {join_table} ON {join_condition} {where_clause}"
            }
    
    def get_best_field_name(self, field: str, field_matches: Dict) -> str:
        """获取最佳字段名"""
        for t in field_matches:
            for f, _ in field_matches[t]:
                if field in f.description or field in f.field_name:
                    return f"{f.table_name}.{f.field_name}"
        return field
    
    def get_best_fields(self, types: List[str], main_table: str, field_matches: Dict, 
                       top_k: int = 2, allow_multi_table: bool = False) -> str:
        """获取最佳字段列表"""
        fields = []
        seen = set()
        for t in types:
            if t in field_matches:
                for field, _ in field_matches[t][:top_k]:
                    field_str = f"{field.table_name}.{field.field_name}"
                    if allow_multi_table or field.table_name == main_table:
                        if field_str not in seen:
                            fields.append(field_str)
                            seen.add(field_str)
        return ", ".join(fields) if fields else "*"
    
    def fill_sql_template(self, elements, field_matches: Dict, sql_templates: Dict) -> str:
        """填充SQL模板"""
        qtype = elements.question_type.value
        template = sql_templates.get(qtype, sql_templates.get("query"))
        
        # 选字段
        main_table = elements.join_tables[0] if elements.join_tables else "emp"
        allow_multi_table = elements.question_type.value == "join"
        select_fields = self.get_best_fields(
            ["attributes", "target_metrics", "entities"], 
            main_table, 
            field_matches, 
            top_k=2, 
            allow_multi_table=allow_multi_table
        )
        
        # 构建WHERE子句
        where_clause = ""
        if elements.conditions:
            conds = []
            for cond in elements.conditions:
                field = cond.get("field", "")
                op = cond.get("operator", "=")
                value = cond.get("value", "")
                real_field = self.get_best_field_name(field, field_matches)
                
                if op.upper() == "LIKE":
                    conds.append(f"{real_field} LIKE '{value}'")
                elif op.upper() == "IS NULL":
                    conds.append(f"{real_field} IS NULL")
                elif op.upper() == "IN":
                    conds.append(f"{real_field} IN ({value})")
                elif op.upper() == "BETWEEN" and isinstance(value, list) and len(value) == 2:
                    conds.append(f"{real_field} BETWEEN {value[0]} AND {value[1]}")
                else:
                    conds.append(f"{real_field} {op} {repr(value)}")
            where_clause = "WHERE " + " AND ".join(conds)
        
        # 聚合函数
        aggregation = elements.aggregation or "SUM"
        
        # 分组字段
        group_by_fields = self.get_best_fields(
            ["group_by_fields"], 
            main_table, 
            field_matches, 
            allow_multi_table=allow_multi_table
        )
        
        # JOIN相关
        join_table = elements.join_tables[1] if len(elements.join_tables) > 1 else ""
        join_condition = ""
        if elements.conditions:
            for cond in elements.conditions:
                field = cond.get("field", "")
                value = cond.get("value", "")
                if isinstance(field, str) and isinstance(value, str) and "." in field and "." in value:
                    join_condition = f"{field} = {value}"
                    break
        if not join_condition and join_table:
            join_condition = f"{main_table}.deptno = {join_table}.deptno"
        
        # 子查询相关
        sub_select_fields = select_fields
        sub_table = main_table
        sub_where_clause = where_clause
        field = select_fields.split(",")[0] if select_fields else "*"
        operator = "="
        
        # 填充模板
        sql = template.format(
            select_fields=select_fields,
            main_table=main_table,
            where_clause=where_clause,
            aggregation=aggregation,
            group_by_fields=group_by_fields,
            join_table=join_table,
            join_condition=join_condition,
            sub_select_fields=sub_select_fields,
            sub_table=sub_table,
            sub_where_clause=sub_where_clause,
            field=field,
            operator=operator
        )
        
        # 清理多余空格
        sql = " ".join(sql.split())
        return sql
    
    def generate_sql(self, question: str) -> Dict:
        """生成SQL查询"""
        try:
            # 1. 要素提取
            elements = self.extractor.extract_elements(question)
            
            # 2. 字段召回
            intent_elements = create_intent_elements_from_extracted_elements(elements)
            field_matches = self.matcher.match_intent_elements(
                intent_elements, 
                top_k=2, 
                similarity_threshold=0.2
            )
            
            # 3. SQL生成
            sql = self.fill_sql_template(elements, field_matches, self.sql_templates)
            
            return {
                "question": question,
                "sql": sql,
                "elements": elements.__dict__,
                "field_matches": {k: [(f.table_name, f.field_name, v) for f, v in v] for k, v in field_matches.items()},
                "success": True
            }
            
        except Exception as e:
            logger.error(f"SQL生成失败: {e}")
            return {
                "question": question,
                "sql": f"-- SQL生成失败: {e}",
                "elements": {},
                "field_matches": {},
                "success": False,
                "error": str(e)
            }

class MultiHopRAGSystem:
    """多跳问答RAG系统 - 核心框架"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # 初始化组件
        self.question_decomposer = QuestionDecomposer(config)
        self.question_rewriter = QuestionRewriter(config)
        self.retriever = HierarchicalRetriever(config)
        self.memory_manager = MemoryManager(config)
        self.file_tool = FileSystemTool(config)
        self.plan_tool = PlanTool(config)
        
        # 新增SQL生成代理
        self.sql_agent = SQLGenerationAgent(config)
        
        # 初始化代理
        self.report_agent = ReportGenerationAgent(config, self.memory_manager, self.file_tool)
        self.file_agent = FileProcessingAgent(config, self.file_tool)
        
        # 初始化工具
        self.tools = self._init_tools()
    
    def _init_tools(self) -> List[Tool]:
        """初始化工具列表"""
        tools = [
            Tool(
                name="file_reader",
                func=self.file_tool.read_file,
                description="读取文件内容"
            ),
            Tool(
                name="file_writer", 
                func=self.file_tool.write_file,
                description="写入文件内容"
            ),
            Tool(
                name="file_lister",
                func=self.file_tool.list_files,
                description="列出目录中的文件"
            ),
            Tool(
                name="report_generator",
                func=self.report_agent.generate_report,
                description="生成专业报告"
            ),
            Tool(
                name="file_processor",
                func=self.file_agent.process_file,
                description="处理文件内容"
            ),
            Tool(
                name="plan_creator",
                func=self.plan_tool.create_execution_plan,
                description="创建执行计划"
            ),
            # 新增SQL生成工具
            Tool(
                name="sql_generator",
                func=self.sql_agent.generate_sql,
                description="将自然语言问题转换为SQL查询"
            )
        ]
        return tools
    
    def answer_question(self, question: str) -> Dict:
        """回答多跳问题"""
        
        logger.info(f"开始处理问题: {question}")
        
        # 检查是否是SQL相关问题
        if self._is_sql_question(question):
            return self._handle_sql_question(question)
        
        # 原有的RAG处理逻辑
        # 1. 问题分解
        sub_questions = self.question_decomposer.decompose_question(question)
        logger.info(f"分解得到 {len(sub_questions)} 个子问题")
        
        # 2. 处理每个子问题
        sub_answers = []
        for i, sub_q in enumerate(sub_questions):
            logger.info(f"处理子问题 {i+1}: {sub_q}")
            
            # 2.1 问题改写
            rewritten_q = self.question_rewriter.rewrite_question(sub_q)
            
            # 2.2 层次化检索
            retrieval_results, retrieval_quality = self.retriever.retrieve(rewritten_q)
            
            # 2.3 网络检索补充（如果检索质量差）
            if retrieval_quality < 0.5:
                logger.info(f"子问题 {i+1} 检索质量低，补充网络搜索结果")
            
            # 2.4 生成答案
            answer = self._generate_answer(rewritten_q, retrieval_results)
            
            # 2.5 创建子问题对象
            sub_answer = SubQuestion(
                id=f"sub_{i+1}",
                original_question=sub_q,
                rewritten_question=rewritten_q,
                context="\n".join([r.get("content", "") for r in retrieval_results]),
                answer=answer,
                confidence=self._calculate_confidence(answer, retrieval_results),
                retrieval_quality=retrieval_quality,
                sources=[r.get("source", "") for r in retrieval_results]
            )
            
            sub_answers.append(sub_answer)
            
            # 添加到记忆
            self.memory_manager.add_short_term_memory(sub_q, answer)
        
        # 3. 综合答案生成
        final_answer = self._synthesize_final_answer(question, sub_answers)
        
        # 4. 添加到长期记忆
        self.memory_manager.add_long_term_memory(
            f"问题: {question}\n答案: {final_answer}",
            importance=0.7,
            context=question
        )
        
        return {
            "question": question,
            "sub_questions": [sq.__dict__ for sq in sub_answers],
            "final_answer": final_answer,
            "confidence": sum(sq.confidence for sq in sub_answers) / len(sub_answers),
            "timestamp": datetime.now().isoformat()
        }
    
    def _is_sql_question(self, question: str) -> bool:
        """判断是否是SQL相关问题"""
        sql_keywords = [
            "显示", "查找", "查询", "统计", "计算", "平均", "总和", "最高", "最低",
            "员工", "雇员", "部门", "工资", "入职", "姓名", "编号", "职位",
            "emp", "dept", "sal", "ename", "empno", "deptno", "job", "hiredate"
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in sql_keywords)
    
    def _handle_sql_question(self, question: str) -> Dict:
        """处理SQL相关问题"""
        logger.info(f"检测到SQL相关问题: {question}")
        
        # 使用SQL生成代理
        sql_result = self.sql_agent.generate_sql(question)
        
        # 添加到记忆
        self.memory_manager.add_short_term_memory(question, sql_result["sql"])
        
        return {
            "question": question,
            "sql_query": sql_result["sql"],
            "elements": sql_result.get("elements", {}),
            "field_matches": sql_result.get("field_matches", {}),
            "success": sql_result.get("success", False),
            "timestamp": datetime.now().isoformat(),
            "type": "sql_generation"
        }
    
    def _generate_answer(self, question: str, retrieval_results: List[Dict]) -> str:
        """生成答案"""
        
        context = "\n".join([r.get("content", "") for r in retrieval_results])
        
        prompt = f"""
        基于以下信息回答问题：
        
        问题：{question}
        相关信息：{context}
        
        请使用COT-SC（Chain of Thought with Self-Consistency）方法：
        1. 仔细分析问题
        2. 逐步推理
        3. 基于提供的信息给出准确答案
        4. 如果不确定，请说明
        
        答案：
        """
        
        try:
            # 创建临时的LLM实例
            llm = ChatOpenAI(
                temperature=0,
                model="THUDM/glm-4-9b-chat",  
                openai_api_key="sk-",
                openai_api_base="https://api.siliconflow.cn/v1",
                max_retries=0,
            )
            response = llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return f"无法生成答案: {e}"
    
    def _calculate_confidence(self, answer: str, retrieval_results: List[Dict]) -> float:
        """计算答案置信度"""
        if not retrieval_results:
            return 0.0
        
        # 简化的置信度计算
        avg_score = sum(r.get("score", 0.5) for r in retrieval_results) / len(retrieval_results)
        answer_length = len(answer)
        
        # 基于检索质量和答案长度计算置信度
        confidence = (avg_score * 0.7 + min(answer_length / 100, 1.0) * 0.3)
        return min(confidence, 1.0)
    
    def _synthesize_final_answer(self, original_question: str, sub_answers: List[SubQuestion]) -> str:
        """综合最终答案"""
        
        sub_answers_text = "\n".join([
            f"子问题 {i+1}: {sq.original_question}\n答案: {sq.answer}"
            for i, sq in enumerate(sub_answers)
        ])
        
        prompt = f"""
        请基于以下子问题的答案，综合回答原始问题：
        
        原始问题：{original_question}
        
        子问题及答案：
        {sub_answers_text}
        
        请提供一个综合、准确、完整的答案：
        """
        
        try:
            # 创建临时的LLM实例
            llm = ChatOpenAI(
                temperature=0,
                model="THUDM/glm-4-9b-chat",  
                openai_api_key="sk-",
                openai_api_base="https://api.siliconflow.cn/v1",
                max_retries=0,
            )
            response = llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"综合答案失败: {e}")
            return f"综合答案失败: {e}"
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            "memory_items": len(self.memory_manager.long_term_memory),
            "short_term_memory_size": self.config.memory_window_size,
            "workspace_files": len(self.file_tool.list_files()),
            "bm25_documents": len(self.retriever.bm25_manager.documents),  # 更新为BM25文档数
            "system_ready": bool(self.config.openai_api_key),
            "retrieval_pipeline": "BM25 + Contriever"  # 显示检索管道信息
        }

# 使用示例
def main():
    """主函数 - 演示系统使用"""
    
    # 初始化配置
    config = Config()
    
    # 创建RAG系统
    rag_system = MultiHopRAGSystem(config)
    
    # 示例问题
    test_questions = [
        "显示工资高于 3000 的员工",
        "查找 1982 年 1 月 1 号以后入职的员工",
        "显示工资在 2000-2500 之间的员工的情况",
        "显示首字母为 s 员工姓名",
        "如何显示第三个字符为大写 o 的所有员工的姓名和工资",
        "如何显示 empno 为 123、345、800 的雇员的情况",
        "显示没有上级的雇员的情况",
        "查询工资高于 500 岗位是 manager 的雇员同时他们的姓名首字母为 J",
        "如何按照工资从高到底来显示",
        "如何按照入职的先后顺序排列",
        "按照部门升序而员工的工资降序排列",
        "统计每个人的年薪并按照从低到高的顺序排列",
        "如何显示工资最高的和工资最低的",
        "如何显示最低工资和该雇员的名字",
        "显示所有员工的品军工资和工资总和",
        "把高于平均工资的雇员的名字和他的工资显示出来",
        "统计共有多少员工",
        "显示每个部门的平均工资和最高工资，并显示部门名称",
        "显示每个部门的每种岗位的平均工资和最低工资",
        "显示平均工资低于 2000 的部门号和它的平均工资",
        "显示雇员的名字，雇员的工资以及所在部门的名称",
        "显示部门名称号为 10 的部门名称、员工和工资",
        "显示雇员名、雇员工资以及所在部门的名字并按部门排序"
    ]
    
    print("=== 多跳问答RAG系统演示 ===\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"问题 {i}: {question}")
        print("-" * 50)
        
        # 获取答案
        result = rag_system.answer_question(question)
        
        if result.get("type") == "sql_generation":
            print(f"生成的SQL: {result['sql_query']}")
            print(f"成功: {result['success']}")
            if not result['success']:
                print(f"错误: {result.get('error', '未知错误')}")
        else:
            print(f"最终答案: {result['final_answer']}")
            print(f"置信度: {result['confidence']:.2f}")
            print(f"子问题数量: {len(result['sub_questions'])}")
        
        print("\n" + "="*50 + "\n")
    
    # 系统状态
    status = rag_system.get_system_status()
    print(f"系统状态: {status}")

if __name__ == "__main__":
    main()
