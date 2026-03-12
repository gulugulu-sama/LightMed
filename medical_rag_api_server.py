#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗RAG系统API服务器
基于FastAPI提供RESTful API接口，支持外界调用医疗RAG查询功能
"""

import os
import sys
import asyncio
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.model_manager import ModelManager
from src.core.rag_manager import RAGManager

# 配置日志
def configure_logging():
    """配置日志系统"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "medical_rag_api.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

logger = configure_logging()

# API请求和响应模型
class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str = Field(..., min_length=1, max_length=1000, description="医疗相关问题")
    mode: str = Field(default="hybrid", description="查询模式: naive, local, global, hybrid")
    include_sources: bool = Field(default=True, description="是否包含引用信息")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "糖尿病有什么症状？",
                "mode": "hybrid",
                "include_sources": True
            }
        }

class QueryResponse(BaseModel):
    """查询响应模型"""
    success: bool = Field(description="请求是否成功")
    answer: str = Field(description="AI回答")
    mode: str = Field(description="使用的查询模式")
    timestamp: str = Field(description="响应时间戳")
    processing_time: float = Field(description="处理时间(秒)")
    sources: Optional[Dict[str, Any]] = Field(None, description="引用源信息")
    error: Optional[str] = Field(None, description="错误信息")

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(description="服务状态")
    version: str = Field(description="系统版本")
    uptime: str = Field(description="运行时间")
    models_status: Dict[str, str] = Field(description="模型状态")
    knowledge_base_status: str = Field(description="知识库状态")

class SystemStatsResponse(BaseModel):
    """系统统计响应模型"""
    total_queries: int = Field(description="总查询数")
    successful_queries: int = Field(description="成功查询数")
    failed_queries: int = Field(description="失败查询数")
    avg_processing_time: float = Field(description="平均处理时间")
    knowledge_base_info: Dict[str, Any] = Field(description="知识库信息")

# 全局变量
rag_system = None
system_stats = {
    "total_queries": 0,
    "successful_queries": 0,
    "failed_queries": 0,
    "processing_times": [],
    "start_time": datetime.now()
}

class MedicalRAGAPIService:
    """医疗RAG API服务类"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.rag_manager = RAGManager("configs/rag_config.yaml")
        self.is_initialized = False
        self.knowledge_base_dir = "./knowledge_bases"
        # 使用现有的small storage
        self.rag_manager.working_dir = "./medical_json_small_storage"
    
    async def initialize(self) -> bool:
        """初始化系统"""
        try:
            logger.info("医疗RAG API服务初始化开始...")
            
            # 1. 检查现有知识图谱
            if not self.check_existing_graph():
                logger.error("知识图谱文件不完整")
                return False
            
            # 2. 初始化模型管理器
            logger.info("初始化模型管理器...")
            if not await self.model_manager.initialize_models():
                logger.error("模型初始化失败")
                return False
            
            # 3. 获取模型配置
            model_config = self.model_manager.get_model_config()
            
            # 4. 初始化RAG管理器
            logger.info("初始化RAG管理器...")
            if not await self.rag_manager.initialize(model_config):
                logger.error("RAG管理器初始化失败")
                return False
            
            # 5. 测试基础功能
            if not await self.rag_manager.test_basic_functions():
                logger.error("基础功能测试失败")
                return False
            
            self.is_initialized = True
            logger.info("医疗RAG API服务初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            return False
    
    def check_existing_graph(self) -> bool:
        """检查现有知识图谱文件"""
        working_dir = Path(self.rag_manager.working_dir)
        
        if not working_dir.exists():
            logger.error(f"工作目录不存在: {working_dir}")
            return False
        
        required_files = [
            'graph_chunk_entity_relation.graphml',
            'kv_store_doc_status.json',
            'kv_store_full_docs.json',
            'kv_store_text_chunks.json',
            'kv_store_llm_response_cache.json',
            'vdb_chunks.json',
            'vdb_entities.json',
            'vdb_relationships.json'
        ]
        
        missing_files = []
        for file in required_files:
            file_path = working_dir / file
            if not file_path.exists():
                missing_files.append(file)
        
        if missing_files:
            logger.warning(f"缺少文件: {missing_files}")
            return False
        else:
            logger.info(f"所有 {len(required_files)} 个知识图谱文件都存在")
            return True
    
    async def query(self, question: str, mode: str = "hybrid") -> Dict[str, Any]:
        """执行查询"""
        if not self.is_initialized:
            raise RuntimeError("系统未初始化")
        
        start_time = datetime.now()
        try:
            # 执行查询
            response = await self.rag_manager.query(question, mode)
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # 获取引用信息
            sources = await self.get_source_info(question, mode)
            
            return {
                "success": True,
                "answer": response,
                "mode": mode,
                "timestamp": end_time.isoformat(),
                "processing_time": processing_time,
                "sources": sources,
                "error": None
            }
            
        except Exception as e:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            logger.error(f"查询失败: {e}")
            return {
                "success": False,
                "answer": "",
                "mode": mode,
                "timestamp": end_time.isoformat(),
                "processing_time": processing_time,
                "sources": None,
                "error": str(e)
            }
    
    async def get_source_info(self, question: str, mode: str) -> Dict[str, Any]:
        """获取引用源信息 - 改进版本，动态阈值和更准确的计数"""
        try:
            sources = {
                "relevant_chunks": 0,
                "relevant_entities": 0,
                "relevant_relations": 0,
                "chunk_details": [],
                "entity_details": [],
                "relation_details": []
            }
            
            # 动态阈值：根据问题长度和复杂度调整
            base_chunk_threshold = 0.4  # 提高基础阈值
            base_entity_threshold = 0.5
            base_relation_threshold = 0.4
            
            # 搜索更多结果，然后动态过滤
            search_top_k = 15
            
            # 搜索相关的文档片段
            if hasattr(self.rag_manager.rag, 'chunks_vdb') and self.rag_manager.rag.chunks_vdb:
                chunk_results = await self.rag_manager.rag.chunks_vdb.query(question, top_k=search_top_k)
                if chunk_results:
                    # 动态阈值：取前几个结果的平均相似度作为参考
                    similarities = [r.get('distance', 0) for r in chunk_results[:5]]
                    avg_sim = sum(similarities) / len(similarities) if similarities else 0
                    dynamic_threshold = max(base_chunk_threshold, avg_sim * 0.7)  # 动态调整
                    
                    relevant_chunks = [r for r in chunk_results if r.get('distance', 0) >= dynamic_threshold]
                    sources["relevant_chunks"] = len(relevant_chunks)
                    sources["chunk_details"] = [
                        {
                            "content": chunk.get('content', '')[:200] + "...",
                            "similarity": chunk.get('distance', 0)
                        }
                        for chunk in relevant_chunks[:3]
                    ]
            
            # 搜索相关的医疗实体
            if hasattr(self.rag_manager.rag, 'entities_vdb') and self.rag_manager.rag.entities_vdb:
                entity_results = await self.rag_manager.rag.entities_vdb.query(question, top_k=search_top_k)
                if entity_results:
                    # 实体的动态阈值
                    similarities = [r.get('distance', 0) for r in entity_results[:5]]
                    avg_sim = sum(similarities) / len(similarities) if similarities else 0
                    dynamic_threshold = max(base_entity_threshold, avg_sim * 0.8)
                    
                    relevant_entities = [r for r in entity_results if r.get('distance', 0) >= dynamic_threshold]
                    sources["relevant_entities"] = len(relevant_entities)
                    sources["entity_details"] = [
                        {
                            "name": entity.get('entity_name', entity.get('id', '未知')),
                            "similarity": entity.get('distance', 0)
                        }
                        for entity in relevant_entities[:5]
                    ]
            
            # 搜索相关的知识关系
            if hasattr(self.rag_manager.rag, 'relationships_vdb') and self.rag_manager.rag.relationships_vdb:
                rel_results = await self.rag_manager.rag.relationships_vdb.query(question, top_k=search_top_k)
                if rel_results:
                    # 关系的动态阈值
                    similarities = [r.get('distance', 0) for r in rel_results[:5]]
                    avg_sim = sum(similarities) / len(similarities) if similarities else 0
                    dynamic_threshold = max(base_relation_threshold, avg_sim * 0.75)
                    
                    relevant_relations = [r for r in rel_results if r.get('distance', 0) >= dynamic_threshold]
                    sources["relevant_relations"] = len(relevant_relations)
                    sources["relation_details"] = [
                        {
                            "source": rel.get('src_id', rel.get('source_id', '?')),
                            "target": rel.get('tgt_id', rel.get('target_id', '?')),
                            "similarity": rel.get('distance', 0)
                        }
                        for rel in relevant_relations[:3]
                    ]
            
            # 添加调试信息（仅在开发环境）
            logger.debug(f"引用统计 - 问题: {question[:50]}...")
            logger.debug(f"  文档: {sources['relevant_chunks']} 个")
            logger.debug(f"  实体: {sources['relevant_entities']} 个")
            logger.debug(f"  关系: {sources['relevant_relations']} 个")
            
            return sources
            
        except Exception as e:
            logger.warning(f"获取引用信息失败: {e}")
            return {
                "relevant_chunks": 0,
                "relevant_entities": 0,
                "relevant_relations": 0,
                "error": str(e)
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        uptime = datetime.now() - system_stats["start_time"]
        
        return {
            "status": "healthy" if self.is_initialized else "initializing",
            "version": "1.0.0",
            "uptime": str(uptime),
            "models_status": {
                "ollama": "online" if self.model_manager.check_ollama_status() else "offline",
                "embedding": "ready" if self.is_initialized else "not_ready",
                "llm": "ready" if self.is_initialized else "not_ready"
            },
            "knowledge_base_status": "loaded" if self.is_initialized else "not_loaded"
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        avg_time = 0
        if system_stats["processing_times"]:
            avg_time = sum(system_stats["processing_times"]) / len(system_stats["processing_times"])
        
        # 获取知识库信息
        kb_info = {}
        try:
            import json
            storage_dir = self.rag_manager.working_dir
            
            files_info = {}
            for filename in ['vdb_chunks.json', 'vdb_entities.json', 'vdb_relationships.json']:
                file_path = os.path.join(storage_dir, filename)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'data' in data:
                            files_info[filename.replace('.json', '')] = len(data['data'])
                        else:
                            files_info[filename.replace('.json', '')] = 0
            
            kb_info = {
                "documents": files_info.get('vdb_chunks', 0),
                "entities": files_info.get('vdb_entities', 0), 
                "relationships": files_info.get('vdb_relationships', 0)
            }
            
        except Exception as e:
            logger.warning(f"获取知识库信息失败: {e}")
            kb_info = {"error": str(e)}
        
        return {
            "total_queries": system_stats["total_queries"],
            "successful_queries": system_stats["successful_queries"], 
            "failed_queries": system_stats["failed_queries"],
            "avg_processing_time": round(avg_time, 3),
            "knowledge_base_info": kb_info
        }
    
    async def cleanup(self):
        """清理资源"""
        try:
            if self.rag_manager:
                await self.rag_manager.cleanup()
            logger.info("系统资源清理完成")
        except Exception as e:
            logger.error(f"资源清理时出错: {e}")

# FastAPI应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global rag_system
    
    # 启动时初始化
    logger.info("启动医疗RAG API服务...")
    rag_system = MedicalRAGAPIService()
    
    if await rag_system.initialize():
        logger.info("医疗RAG API服务启动成功")
        yield
    else:
        logger.error("医疗RAG API服务启动失败")
        raise RuntimeError("服务初始化失败")
    
    # 关闭时清理
    logger.info("正在关闭医疗RAG API服务...")
    if rag_system:
        await rag_system.cleanup()
    logger.info("医疗RAG API服务已关闭")

# 创建FastAPI应用
app = FastAPI(
    title="医疗RAG系统API",
    description="基于LightRAG的医疗知识图谱问答系统API服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API路由
@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径"""
    return {
        "message": "医疗RAG系统API服务",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    health_data = rag_system.get_health_status()
    return HealthResponse(**health_data)

@app.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """获取系统统计信息"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    stats_data = rag_system.get_system_stats()
    return SystemStatsResponse(**stats_data)

@app.post("/query", response_model=QueryResponse)
async def query_medical_rag(request: QueryRequest):
    """医疗RAG查询接口"""
    global system_stats
    
    if not rag_system or not rag_system.is_initialized:
        system_stats["failed_queries"] += 1
        raise HTTPException(status_code=503, detail="RAG系统未初始化")
    
    # 验证查询模式
    valid_modes = ["naive", "local", "global", "hybrid"]
    if request.mode not in valid_modes:
        system_stats["failed_queries"] += 1
        raise HTTPException(
            status_code=400, 
            detail=f"无效的查询模式: {request.mode}。有效模式: {valid_modes}"
        )
    
    try:
        # 记录统计
        system_stats["total_queries"] += 1
        
        # 执行查询
        result = await rag_system.query(request.question, request.mode)
        
        # 更新统计
        if result["success"]:
            system_stats["successful_queries"] += 1
        else:
            system_stats["failed_queries"] += 1
        
        system_stats["processing_times"].append(result["processing_time"])
        
        # 如果不需要引用信息，移除sources字段
        if not request.include_sources:
            result["sources"] = None
        
        return QueryResponse(**result)
        
    except Exception as e:
        system_stats["failed_queries"] += 1
        logger.error(f"查询处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")

@app.get("/modes")
async def get_available_modes():
    """获取可用的查询模式"""
    return {
        "modes": [
            {
                "name": "naive",
                "description": "基础检索模式，快速返回最相关的文档片段"
            },
            {
                "name": "local", 
                "description": "局部推理模式，基于相关实体和关系进行推理"
            },
            {
                "name": "global",
                "description": "全局推理模式，基于全局知识图谱提供广泛背景信息"
            },
            {
                "name": "hybrid",
                "description": "混合推理模式，结合多种方式提供最全面答案（推荐）"
            }
        ],
        "default": "hybrid"
    }

# 异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "内部服务器错误", "error": str(exc)}
    )

def start_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """启动API服务器"""
    logger.info(f"启动医疗RAG API服务器...")
    logger.info(f"服务地址: http://{host}:{port}")
    logger.info(f"API文档: http://{host}:{port}/docs")
    logger.info(f"健康检查: http://{host}:{port}/health")
    
    uvicorn.run(
        "medical_rag_api_server:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="医疗RAG API服务器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    
    args = parser.parse_args()
    
    start_server(host=args.host, port=args.port, workers=args.workers)