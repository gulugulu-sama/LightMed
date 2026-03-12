"""
查询优化器 - 毕业设计创新点实现
包含：查询意图分类、医疗实体增强、自适应模式选择
"""

import re
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QueryIntent:
    """查询意图"""
    intent_type: str  # factual, relational, diagnostic, treatment
    confidence: float
    keywords: List[str]


@dataclass
class EnhancedQuery:
    """增强后的查询"""
    original_query: str
    enhanced_query: str
    extracted_entities: List[Dict[str, Any]]
    synonyms_added: List[str]
    related_entities: List[str]


class QueryIntentClassifier:
    """查询意图分类器 - 创新点1的核心组件"""
    
    def __init__(self):
        # 意图关键词模式
        self.intent_patterns = {
            'factual': {
                'keywords': ['是什么', '定义', '概念', '含义', '介绍', '什么是'],
                'weight': 1.0
            },
            'relational': {
                'keywords': ['关系', '关联', '影响', '导致', '引起', '相关', '联系'],
                'weight': 1.0
            },
            'diagnostic': {
                'keywords': ['症状', '可能', '是不是', '是否', '诊断', '得了', '患了'],
                'weight': 1.2  # 诊断类问题权重更高
            },
            'treatment': {
                'keywords': ['治疗', '方法', '如何', '怎么办', '用药', '处方', '疗法'],
                'weight': 1.0
            }
        }
        
        # 历史性能数据（用于自适应选择）
        self.mode_performance = defaultdict(lambda: {
            'total_queries': 0,
            'successful_queries': 0,
            'avg_response_time': 0.0,
            'avg_quality_score': 0.0
        })
    
    def classify_intent(self, query: str) -> QueryIntent:
        """分类查询意图"""
        query_lower = query.lower()
        intent_scores = {}
        
        for intent_type, pattern in self.intent_patterns.items():
            score = 0.0
            matched_keywords = []
            
            for keyword in pattern['keywords']:
                if keyword in query_lower:
                    score += pattern['weight']
                    matched_keywords.append(keyword)
            
            if score > 0:
                intent_scores[intent_type] = {
                    'score': score,
                    'keywords': matched_keywords
                }
        
        # 选择得分最高的意图
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1]['score'])
            intent_type = best_intent[0]
            confidence = min(best_intent[1]['score'] / 3.0, 1.0)  # 归一化到0-1
            keywords = best_intent[1]['keywords']
        else:
            # 默认意图：综合查询
            intent_type = 'factual'
            confidence = 0.5
            keywords = []
        
        return QueryIntent(
            intent_type=intent_type,
            confidence=confidence,
            keywords=keywords
        )
    
    def recommend_mode(self, intent: QueryIntent) -> str:
        """基于意图推荐查询模式"""
        # 基础映射规则
        intent_to_mode = {
            'factual': 'naive',      # 事实查询用naive更快
            'relational': 'local',   # 关系查询用local
            'diagnostic': 'hybrid',   # 诊断需要综合信息
            'treatment': 'global'     # 治疗方案需要全局知识
        }
        
        recommended_mode = intent_to_mode.get(intent.intent_type, 'hybrid')
        
        # 考虑历史性能数据（如果有）
        if self.mode_performance[recommended_mode]['total_queries'] > 0:
            # 如果推荐模式历史表现不佳，考虑降级
            success_rate = (self.mode_performance[recommended_mode]['successful_queries'] / 
                          self.mode_performance[recommended_mode]['total_queries'])
            
            if success_rate < 0.5:  # 成功率低于50%，考虑使用hybrid
                recommended_mode = 'hybrid'
        
        return recommended_mode
    
    def update_performance(self, mode: str, success: bool, response_time: float, quality_score: float = 0.0):
        """更新模式性能数据"""
        perf = self.mode_performance[mode]
        perf['total_queries'] += 1
        
        if success:
            perf['successful_queries'] += 1
        
        # 更新平均响应时间（移动平均）
        if perf['total_queries'] == 1:
            perf['avg_response_time'] = response_time
            perf['avg_quality_score'] = quality_score
        else:
            alpha = 0.3  # 平滑系数
            perf['avg_response_time'] = (alpha * response_time + 
                                       (1 - alpha) * perf['avg_response_time'])
            perf['avg_quality_score'] = (alpha * quality_score + 
                                        (1 - alpha) * perf['avg_quality_score'])


class MedicalEntityExtractor:
    """医疗实体提取器 - 创新点2的核心组件"""
    
    def __init__(self):
        # 医疗实体词典（简化版，实际应该从知识图谱中提取）
        self.medical_entities = {
            'disease': [
                '糖尿病', '高血压', '心脏病', '冠心病', '肺炎', '感冒',
                '心肌梗死', '脑卒中', '哮喘', '胃炎', '肝炎', '肾炎'
            ],
            'symptom': [
                '胸痛', '发热', '咳嗽', '头痛', '呼吸困难', '腹痛',
                '恶心', '呕吐', '腹泻', '乏力', '心悸', '头晕'
            ],
            'drug': [
                '胰岛素', '阿司匹林', '青霉素', '硝苯地平', '氨氯地平',
                '二甲双胍', '格列美脲', '阿卡波糖'
            ],
            'treatment': [
                '手术治疗', '药物治疗', '物理治疗', '化疗', '放疗',
                '介入治疗', '保守治疗'
            ]
        }
        
        # 实体同义词库
        self.entity_synonyms = {
            '糖尿病': ['DM', 'diabetes', '高血糖'],
            '高血压': ['HTN', 'hypertension', '血压高'],
            '心脏病': ['心脏病', '心血管疾病'],
            '胸痛': ['胸部疼痛', '心前区疼痛']
        }
        
        # 实体关系库（简化版）
        self.entity_relations = {
            '糖尿病': ['高血糖', '胰岛素', '血糖', '并发症'],
            '高血压': ['心脏病', '脑卒中', '血管', '血压'],
            '胸痛': ['心脏病', '心肌梗死', '心绞痛', '冠心病']
        }
    
    def extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """提取查询中的医疗实体"""
        entities = []
        query_lower = query.lower()
        
        for entity_type, entity_list in self.medical_entities.items():
            for entity in entity_list:
                if entity in query:
                    entities.append({
                        'text': entity,
                        'type': entity_type,
                        'position': query.find(entity)
                    })
        
        return entities
    
    def get_synonyms(self, entity: str) -> List[str]:
        """获取实体同义词"""
        return self.entity_synonyms.get(entity, [])
    
    def get_related_entities(self, entity: str) -> List[str]:
        """获取相关实体"""
        return self.entity_relations.get(entity, [])


class QueryRewriter:
    """查询重写器 - 创新点2的核心组件"""
    
    def __init__(self, entity_extractor: MedicalEntityExtractor):
        self.entity_extractor = entity_extractor
    
    def rewrite_query(self, query: str, use_synonyms: bool = True, 
                     use_relations: bool = True) -> EnhancedQuery:
        """重写查询，增强医疗实体"""
        # 提取实体
        entities = self.entity_extractor.extract_entities(query)
        
        enhanced_parts = [query]
        synonyms_added = []
        related_entities = []
        
        # 添加同义词
        if use_synonyms:
            for entity in entities:
                synonyms = self.entity_extractor.get_synonyms(entity['text'])
                if synonyms:
                    synonyms_added.extend(synonyms)
                    # 添加同义词到查询中
                    enhanced_parts.append(' '.join(synonyms))
        
        # 添加相关实体
        if use_relations:
            for entity in entities:
                related = self.entity_extractor.get_related_entities(entity['text'])
                if related:
                    related_entities.extend(related)
                    # 添加相关实体到查询中
                    enhanced_parts.append(' '.join(related[:3]))  # 限制数量
        
        # 构建增强查询
        enhanced_query = ' '.join(enhanced_parts)
        
        return EnhancedQuery(
            original_query=query,
            enhanced_query=enhanced_query,
            extracted_entities=entities,
            synonyms_added=synonyms_added,
            related_entities=related_entities
        )


class AdaptiveQueryOptimizer:
    """自适应查询优化器 - 整合所有创新点"""
    
    def __init__(self):
        self.intent_classifier = QueryIntentClassifier()
        self.entity_extractor = MedicalEntityExtractor()
        self.query_rewriter = QueryRewriter(self.entity_extractor)
        
        # 配置：是否启用各个优化功能
        self.enable_intent_classification = True
        self.enable_query_rewriting = True
        self.enable_adaptive_selection = True
    
    def optimize_query(self, query: str) -> Dict[str, Any]:
        """优化查询，返回优化后的查询和推荐模式"""
        result = {
            'original_query': query,
            'intent': None,
            'recommended_mode': 'hybrid',  # 默认模式
            'enhanced_query': query,
            'entities': [],
            'optimization_steps': []
        }
        
        # 步骤1: 意图分类
        if self.enable_intent_classification:
            intent = self.intent_classifier.classify_intent(query)
            result['intent'] = {
                'type': intent.intent_type,
                'confidence': intent.confidence,
                'keywords': intent.keywords
            }
            result['optimization_steps'].append('intent_classification')
            
            # 基于意图推荐模式
            if self.enable_adaptive_selection:
                recommended_mode = self.intent_classifier.recommend_mode(intent)
                result['recommended_mode'] = recommended_mode
                result['optimization_steps'].append('adaptive_mode_selection')
        
        # 步骤2: 查询重写
        if self.enable_query_rewriting:
            enhanced = self.query_rewriter.rewrite_query(
                query,
                use_synonyms=True,
                use_relations=True
            )
            result['enhanced_query'] = enhanced.enhanced_query
            result['entities'] = [
                {'text': e['text'], 'type': e['type']} 
                for e in enhanced.extracted_entities
            ]
            result['optimization_steps'].append('query_rewriting')
        
        return result
    
    def update_feedback(self, mode: str, success: bool, 
                       response_time: float, quality_score: float = 0.0):
        """更新性能反馈，用于自适应学习"""
        if self.enable_adaptive_selection:
            self.intent_classifier.update_performance(
                mode, success, response_time, quality_score
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return {
            'mode_performance': dict(self.intent_classifier.mode_performance),
            'total_queries': sum(
                p['total_queries'] 
                for p in self.intent_classifier.mode_performance.values()
            )
        }


# 全局优化器实例
query_optimizer = AdaptiveQueryOptimizer()


def get_query_optimizer() -> AdaptiveQueryOptimizer:
    """获取查询优化器实例"""
    return query_optimizer

