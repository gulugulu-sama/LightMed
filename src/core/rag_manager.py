"""
RAG管理器 - 基于medical_rag_kb_official.py最终优化版本
核心RAG功能管理，集成LightRAG和智能文档处理
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from datetime import datetime

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

logger = logging.getLogger(__name__)

class RAGManager:
    """RAG管理器 - 基于最终优化版本"""
    
    def __init__(self, config_path: str = "configs/rag_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.rag = None
        self.working_dir = self.config['lightrag']['working_dir']
        self.is_initialized = False
        self._semaphore = asyncio.Semaphore(1)  # 基于您的成功版本：并发控制
        
        # 确保工作目录存在
        os.makedirs(self.working_dir, exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载RAG配置失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置 - 基于您的最终优化版本"""
        return {
            'lightrag': {'working_dir': './lightrag_storage'},
            'llm_model': {
                'max_token_size': 8192,
                'timeout': 1200,
                'request_delay': 1.0
            },
            'embedding': {
                'dimensions': 1024,
                'max_token_size': 8192,
                'timeout': 1200
            },
            'performance': {
                'max_concurrent_requests': 1,
                'insert_delay': 1.0,
                'smart_chunking': True,
                'chunk_size': 1000,
                'large_doc_threshold': 3000,
                'sentence_split_ratio': 0.7
            },
            'error_handling': {
                'max_retries': 3,
                'retry_delay': 2,
                'recovery_delay': 30
            }
        }
    
    async def initialize(self, model_config: Dict[str, Any]) -> bool:
        """初始化RAG系统 - 基于您的最终优化版本"""
        try:
            logger.info("初始化RAG系统...")
            
            # 创建LightRAG实例 - 基于您的最终版本参数
            self.rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=ollama_model_complete,
                llm_model_name=model_config['llm']['model'],
                llm_model_max_token_size=model_config['llm']['max_tokens'],  # 8192
                llm_model_kwargs={
                    "host": model_config['ollama']['base_url'],
                    "options": {"num_ctx": model_config['llm']['context_window']},  # 8192
                    "timeout": model_config['llm']['timeout'],  # 1200秒
                },
                embedding_func=EmbeddingFunc(
                    embedding_dim=model_config['embedding']['dimension'],  # 1024
                    max_token_size=model_config['embedding']['max_tokens'],  # 8192
                    func=lambda texts: ollama_embed(
                        texts,
                        embed_model=model_config['embedding']['model'],
                        host=model_config['ollama']['base_url'],
                        timeout=model_config['embedding']['timeout'],  # 1200秒 - 关键修复
                    ),
                ),
            )
            
            # 官方标准初始化流程 - 基于您的最终版本
            await self.rag.initialize_storages()
            await initialize_pipeline_status()
            
            self.is_initialized = True
            logger.info("RAG系统初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"RAG系统初始化失败: {e}")
            return False
    
    async def test_basic_functions(self) -> bool:
        """测试基础函数 - 基于您的最终优化版本"""
        if not self.is_initialized:
            logger.error("RAG系统未初始化")
            return False
            
        try:
            logger.info("测试Embedding函数...")
            test_text = ["这是一个测试句子。"]
            embedding = await self.rag.embedding_func(test_text)
            embedding_dim = embedding.shape[1]
            logger.info(f"Embedding测试结果: {test_text}")
            logger.info(f"检测到的嵌入维度: {embedding_dim}")
            
            logger.info("测试LLM函数...")
            result = await self.rag.llm_model_func("你好，请简单回复")
            logger.info(f"LLM测试结果: {result[:100]}..." if len(result) > 100 else result)
            
            return True
            
        except Exception as e:
            logger.error(f"基础功能测试失败: {e}")
            return False
    
    def load_knowledge_base_files(self, kb_dir: str) -> List[Dict[str, Any]]:
        """加载知识库文件 - 基于您的智能分块策略"""
        kb_path = Path(kb_dir)
        if not kb_path.exists():
            logger.warning(f"知识库目录不存在: {kb_dir}")
            return []
        
        files = []
        txt_files = list(kb_path.rglob("*.txt"))
        
        logger.info(f"发现 {len(txt_files)} 个知识库文件:")
        
        # 性能配置
        perf_config = self.config.get('performance', {})
        chunk_size = perf_config.get('chunk_size', 1000)
        large_doc_threshold = perf_config.get('large_doc_threshold', 3000)
        sentence_split_ratio = perf_config.get('sentence_split_ratio', 0.7)
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                filename = file_path.name
                category = file_path.parent.name
                
                # 智能分块处理 - 基于您的最终版本
                if len(content) > large_doc_threshold:
                    logger.info(f"  {category}/{filename} ({len(content):,} 字符) - 需要分块")
                    
                    chunks = []
                    for i in range(0, len(content), chunk_size):
                        chunk = content[i:i + chunk_size]
                        
                        # 智能句号分割 - 基于您的版本
                        if i + chunk_size < len(content) and '。' in chunk:
                            last_period = chunk.rfind('。')
                            if last_period > chunk_size * sentence_split_ratio:
                                chunk = chunk[:last_period + 1]
                        
                        chunk_num = len(chunks) + 1
                        formatted_chunk = f"""文档名称: {filename} (第{chunk_num}部分)
文档分类: {category}
原文档大小: {len(content):,} 字符

{chunk}"""
                        
                        chunks.append({
                            'path': str(file_path),
                            'filename': f"{filename}_part{chunk_num}",
                            'category': category,
                            'content': formatted_chunk,
                            'size': len(chunk),
                            'is_chunk': True,
                            'chunk_num': chunk_num,
                            'total_chunks': 0
                        })
                    
                    # 更新总块数
                    for chunk in chunks:
                        chunk['total_chunks'] = len(chunks)
                    
                    files.extend(chunks)
                    logger.info(f"    分割为 {len(chunks)} 个块")
                    
                else:
                    # 小文档直接处理
                    formatted_content = f"""文档名称: {filename}
文档分类: {category}
文档大小: {len(content):,} 字符

{content}"""
                    
                    files.append({
                        'path': str(file_path),
                        'filename': filename,
                        'category': category,
                        'content': formatted_content,
                        'size': len(content),
                        'is_chunk': False
                    })
                    
                    logger.info(f"  {category}/{filename} ({len(content):,} 字符)")
                
            except Exception as e:
                logger.error(f"读取文件失败 {file_path}: {e}")
        
        return files
    
    async def load_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """加载文档到RAG系统 - 基于您的最终优化版本"""
        if not self.is_initialized:
            logger.error("RAG系统未初始化")
            return False
        
        logger.info(f"开始加载 {len(documents)} 个文档块到RAG系统...")
        
        total_chars = sum(doc['size'] for doc in documents)
        logger.info(f"总文档大小: {total_chars:,} 字符")
        
        # 统计信息
        chunk_docs = [doc for doc in documents if doc.get('is_chunk', False)]
        normal_docs = [doc for doc in documents if not doc.get('is_chunk', False)]
        
        if chunk_docs:
            logger.info(f"包含 {len(normal_docs)} 个完整文档 + {len(chunk_docs)} 个分块")
        else:
            logger.info(f"包含 {len(normal_docs)} 个完整文档，无需分块")
        
        success_count = 0
        failed_docs = []
        
        # 获取配置
        perf_config = self.config.get('performance', {})
        insert_delay = perf_config.get('insert_delay', 1.0)
        error_config = self.config.get('error_handling', {})
        recovery_delay = error_config.get('recovery_delay', 30)
        
        async with self._semaphore:  # 并发控制 - 基于您的成功版本
            for i, doc in enumerate(documents, 1):
                try:
                    doc_name = doc['filename']
                    if doc.get('is_chunk', False):
                        doc_name = f"{doc['filename']} ({doc['chunk_num']}/{doc['total_chunks']})"
                    
                    logger.info(f"[{i}/{len(documents)}] 正在插入: {doc['category']}/{doc_name} ({doc['size']:,} 字符)")
                    
                    await self.rag.ainsert(doc['content'])
                    success_count += 1
                    
                    logger.info("   插入完成")
                    
                    # 文档间延迟 - 基于您的成功版本
                    if i < len(documents):
                        await asyncio.sleep(insert_delay)
                    
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    logger.error(f"   插入失败: {error_msg}")
                    
                    failed_docs.append({
                        'doc': doc,
                        'error': error_msg
                    })
                    
                    # 超时错误处理 - 基于您的版本
                    if 'timeout' in error_msg.lower() or 'ReadTimeout' in error_msg:
                        logger.info(f"   超时错误，等待{recovery_delay}秒后继续...")
                        await asyncio.sleep(recovery_delay)
                    
                    continue
        
        logger.info(f"加载结果统计:")
        logger.info(f"成功: {success_count}/{len(documents)} 个文档")
        logger.info(f"失败: {len(failed_docs)} 个文档")
        
        if failed_docs and success_count > 0:
            logger.info(f"部分文档加载失败，但已有 {success_count} 个文档成功加载，可以继续使用系统")
            return True
        elif success_count > 0:
            logger.info("所有文档加载成功！")
            return True
        else:
            logger.error("所有文档都加载失败")
            return False
    
    def check_existing_knowledge_base(self) -> bool:
        """检查是否存在完整的知识库 - 基于您的最终版本"""
        working_dir = Path(self.working_dir)
        if not working_dir.exists():
            return False
        
        # 检查关键文件
        key_files = [
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json", 
            "vdb_entities.json",
            "vdb_relationships.json"
        ]
        
        existing_files = []
        for file in key_files:
            file_path = working_dir / file
            if file_path.exists() and file_path.stat().st_size > 100:
                existing_files.append(file)
        
        if len(existing_files) >= 3:
            logger.info(f"发现现有知识库: {len(existing_files)}/{len(key_files)} 个关键文件存在")
            return True
        else:
            logger.info(f"知识库不完整: 只有 {len(existing_files)}/{len(key_files)} 个文件存在")
            return False
    
    def clear_old_data(self):
        """清理旧数据文件 - 基于您的最终版本"""
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_llm_response_cache.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
        ]
        
        logger.info("清理旧数据文件...")
        for file in files_to_delete:
            file_path = os.path.join(self.working_dir, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"  删除旧文件: {file}")
    
    async def query(self, question: str, mode: str = "hybrid") -> str:
        """查询RAG系统"""
        if not self.is_initialized:
            return "ERROR: RAG系统未初始化"
        
        try:
            # 医疗安全检查
            safety_warning = self._medical_safety_check(question)
            if safety_warning:
                return safety_warning
            
            logger.info(f"{mode.upper()} 模式查询: {question}")
            
            response = await self.rag.aquery(
                question,
                param=QueryParam(mode=mode)
            )
            
            # 添加医疗免责声明
            medical_disclaimer = "\n\n WARNING: 医疗免责声明：本系统提供的信息仅供参考，不能替代专业医疗建议。如有健康问题，请咨询专业医生。"
            
            return response + medical_disclaimer
            
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return f"ERROR: 查询失败: {e}"
    
    def _medical_safety_check(self, question: str) -> str:
        """医疗安全检查 - 基于您的最终版本"""
        diagnosis_keywords = ['诊断', '确诊', '是不是', '什么病', '得了', '患了']
        if any(keyword in question for keyword in diagnosis_keywords):
            return 'WARNING: 医疗安全提醒：本系统不提供医疗诊断，请咨询专业医生。'
        
        prescription_keywords = ['开药', '用药量', '吃多少', '处方', '药物剂量', '怎么吃药']
        if any(keyword in question for keyword in prescription_keywords):
            return 'WARNING: 医疗安全提醒：本系统不提供用药指导，请咨询专业医生。'
        
        emergency_keywords = ['急救', '紧急', '突发', '抢救', '危险']
        if any(keyword in question for keyword in emergency_keywords):
            return 'EMERGENCY: 紧急情况请立即拨打120或前往最近的医院急诊科！'
        
        return ""
    
    async def cleanup(self):
        """清理资源 - 基于您的最终版本"""
        if self.rag:
            try:
                await self.rag.llm_response_cache.index_done_callback()
                await self.rag.finalize_storages()
                logger.info("RAG资源清理完成")
            except Exception as e:
                logger.error(f"资源清理时出错: {e}")
    
    def reload_config(self):
        """重新加载配置"""
        self.config = self._load_config()
        logger.info("RAG配置已重新加载")


# 全局RAG管理器实例
rag_manager = RAGManager()


def get_rag_manager() -> RAGManager:
    """获取RAG管理器实例"""
    return rag_manager 