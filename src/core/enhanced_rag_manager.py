"""
增强的RAG管理器 - 集成查询优化器
在原有RAGManager基础上添加查询优化功能
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.core.rag_manager import RAGManager
from src.core.query_optimizer import AdaptiveQueryOptimizer, get_query_optimizer

logger = logging.getLogger(__name__)


class EnhancedRAGManager(RAGManager):
    """增强的RAG管理器 - 继承原有功能，添加查询优化"""
    
    def __init__(self, config_path: str = "configs/rag_config.yaml", 
                 enable_optimization: bool = True):
        super().__init__(config_path)
        self.query_optimizer = get_query_optimizer()
        self.enable_optimization = enable_optimization
        
        # 性能追踪
        self.query_stats = {
            'total_queries': 0,
            'optimized_queries': 0,
            'mode_changes': 0
        }
    
    async def query(self, question: str, mode: Optional[str] = None, 
                   use_optimization: Optional[bool] = None) -> str:
        """增强的查询方法 - 支持自动优化"""
        use_opt = use_optimization if use_optimization is not None else self.enable_optimization
        
        if use_opt:
            # 使用查询优化器
            optimization_result = self.query_optimizer.optimize_query(question)
            
            # 如果用户没有指定模式，使用优化器推荐的模式
            if mode is None:
                mode = optimization_result['recommended_mode']
                logger.info(f"自动选择查询模式: {mode} (意图: {optimization_result.get('intent', {}).get('type')})")
            
            # 使用增强后的查询
            enhanced_query = optimization_result['enhanced_query']
            
            # 记录统计
            self.query_stats['optimized_queries'] += 1
            if mode != 'hybrid':  # 如果改变了默认模式
                self.query_stats['mode_changes'] += 1
            
            # 记录优化信息
            logger.info(f"查询优化:")
            logger.info(f"  原始查询: {question}")
            logger.info(f"  增强查询: {enhanced_query}")
            logger.info(f"  提取实体: {len(optimization_result.get('entities', []))} 个")
            logger.info(f"  推荐模式: {mode}")
            
            # 执行查询
            start_time = datetime.now()
            try:
                response = await super().query(enhanced_query, mode)
                response_time = (datetime.now() - start_time).total_seconds()
                
                # 更新性能反馈（简化版，实际应该评估响应质量）
                self.query_optimizer.update_feedback(
                    mode=mode,
                    success=True,
                    response_time=response_time,
                    quality_score=0.8  # 简化评分，实际应该用质量评估模型
                )
                
                return response
            except Exception as e:
                response_time = (datetime.now() - start_time).total_seconds()
                self.query_optimizer.update_feedback(
                    mode=mode,
                    success=False,
                    response_time=response_time,
                    quality_score=0.0
                )
                raise
        else:
            # 不使用优化，直接查询
            return await super().query(question, mode)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        return {
            'query_stats': self.query_stats.copy(),
            'optimizer_performance': self.query_optimizer.get_performance_stats(),
            'optimization_enabled': self.enable_optimization
        }
    
    def enable_query_optimization(self, enable: bool = True):
        """启用/禁用查询优化"""
        self.enable_optimization = enable
        logger.info(f"查询优化已{'启用' if enable else '禁用'}")
    
    def configure_optimization(self, enable_intent: bool = True,
                              enable_rewriting: bool = True,
                              enable_adaptive: bool = True):
        """配置优化器功能"""
        self.query_optimizer.enable_intent_classification = enable_intent
        self.query_optimizer.enable_query_rewriting = enable_rewriting
        self.query_optimizer.enable_adaptive_selection = enable_adaptive
        
        logger.info(f"优化器配置更新:")
        logger.info(f"  意图分类: {enable_intent}")
        logger.info(f"  查询重写: {enable_rewriting}")
        logger.info(f"  自适应选择: {enable_adaptive}")

