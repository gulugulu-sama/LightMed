#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime


sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.model_manager import ModelManager
from src.core.rag_manager import RAGManager

# 配置日志 
def configure_logging():
    """配置日志系统"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "medical_rag_json_small.log"
    
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

class MedicalRAGSystem:
    
    
    def __init__(self):
        # 使用现有知识图谱的配置 - 基于medical_rag_json_optimized_small.py
        self.model_manager = ModelManager()
        self.rag_manager = RAGManager("configs/rag_config.yaml")
        self.is_initialized = False
        self.knowledge_base_dir = "./knowledge_bases"
        # 覆盖工作目录为您的成功版本目录
        self.rag_manager.working_dir = "./medical_json_small_storage"
    
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
        
        logger.info(f"检查知识图谱文件: {working_dir}")
        
        missing_files = []
        existing_files = []
        
        for file in required_files:
            file_path = working_dir / file
            if file_path.exists():
                size = file_path.stat().st_size
                logger.info(f"  SUCCESS: {file} ({size:,} bytes)")
                existing_files.append(file)
            else:
                logger.info(f"  MISSING: {file}")
                missing_files.append(file)
        
        if missing_files:
            logger.warning(f"缺少 {len(missing_files)} 个文件: {missing_files}")
            return False
        else:
            logger.info(f"所有 {len(required_files)} 个知识图谱文件都存在")
            return True
    
    async def initialize(self) -> bool:
       
        try:
            logger.info("医疗RAG系统 - 基于最终优化版本")
            logger.info("=" * 50)
            logger.info("优化策略:")
            logger.info("  1. 使用现有知识图谱，无需重新构建")
            logger.info("  2. 超时时间优化为查询模式")
            logger.info("  3. 智能检查：有完整知识库则跳过数据加载")
            logger.info("  4. 支持多种查询模式")
            logger.info("=" * 50)
            
            # 1. 检查现有知识图谱
            if not self.check_existing_graph():
                logger.error("知识图谱文件不完整，请先构建知识图谱")
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
            logger.info("系统初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            return False
    
    async def load_knowledge_base(self) -> bool:
        """加载知识库 - 基于您的智能策略"""
        try:
            # 1. 检查现有知识库
            has_existing_kb = self.rag_manager.check_existing_knowledge_base()
            
            if has_existing_kb:
                logger.info("发现完整的现有知识库，跳过数据加载阶段")
                return True
            else:
                logger.info("需要重新构建知识库")
                
                # 清理旧文件
                self.rag_manager.clear_old_data()
            
            # 2. 加载知识库文件
            logger.info("加载知识库文件...")
            kb_documents = self.rag_manager.load_knowledge_base_files(self.knowledge_base_dir)
            
            if not kb_documents:
                logger.error("没有找到知识库文件")
                return False
            
            # 3. 将文档插入RAG系统
            return await self.rag_manager.load_documents(kb_documents)
            
        except Exception as e:
            logger.error(f"知识库加载失败: {e}")
            return False
    
    async def query_with_mode(self, question: str, mode: str = "hybrid"):
        """使用指定模式查询并显示详细信息"""
        if not self.is_initialized:
            logger.error("系统未初始化")
            return
        
        logger.info(f"使用{mode.upper()}模式查询: {question}")
        print(f"\n[查询] 使用 {mode.upper()} 模式查询...")
        print("-" * 50)
        
        try:
            response = await self.rag_manager.query(question, mode)
            print(f"\n[回答] {mode.upper()}模式回答:")
            print("=" * 50)
            print(response)
            print("=" * 50)
            
            # 尝试显示引用信息
            await self.show_source_info(question, mode)
            
        except Exception as e:
            logger.error(f"{mode}模式查询失败: {e}")
            print(f"[错误] ERROR: {mode}模式查询失败: {e}")
    
    async def show_source_info(self, question: str, mode: str):
        """显示针对当前问题的具体引用统计和详情"""
        try:
            print(f"\n[引用] {mode.upper()}模式引用信息:")
            print("-" * 40)
            
            # 获取针对当前问题的相关知识
            print(f"[统计] 针对问题「{question}」的引用统计:")
            
            relevant_chunks = 0
            relevant_entities = 0
            relevant_relations = 0
            
            try:
                # 正确的方式：直接用字符串调用VectorDB的query方法
                # 这样VectorDB会内部处理embedding，避免类型错误
                
                # 1. 搜索相关的文档片段
                if hasattr(self.rag_manager.rag, 'chunks_vdb') and self.rag_manager.rag.chunks_vdb:
                    chunk_results = await self.rag_manager.rag.chunks_vdb.query(question, top_k=10)
                    if chunk_results:
                        # 统计超过阈值的结果
                        relevant_chunks = len([r for r in chunk_results if r.get('distance', 0) > 0.3])
                        print(f"  [文档] 相关文档片段: {relevant_chunks} 个")
                        
                        # 显示最相关的几个片段
                        if relevant_chunks > 0:
                            print("    [详情] 主要引用片段:")
                            for i, chunk_data in enumerate([r for r in chunk_results if r.get('distance', 0) > 0.3][:3], 1):
                                distance = chunk_data.get('distance', 0)
                                content = chunk_data.get('content', '')
                                content_preview = content[:100] + "..." if len(content) > 100 else content
                                print(f"      [{i}] 相似度: {distance:.3f} - {content_preview}")
                
                # 2. 搜索相关的医疗实体
                if hasattr(self.rag_manager.rag, 'entities_vdb') and self.rag_manager.rag.entities_vdb:
                    entity_results = await self.rag_manager.rag.entities_vdb.query(question, top_k=10)
                    if entity_results:
                        relevant_entities = len([r for r in entity_results if r.get('distance', 0) > 0.4])
                        print(f"  [实体] 相关医疗实体: {relevant_entities} 个")
                        
                        # 显示最相关的实体
                        if relevant_entities > 0:
                            print("    [详情] 主要相关实体:")
                            for i, entity_data in enumerate([r for r in entity_results if r.get('distance', 0) > 0.4][:5], 1):
                                distance = entity_data.get('distance', 0)
                                entity_name = entity_data.get('entity_name', entity_data.get('id', '未知'))
                                print(f"      [{i}] {entity_name} (相似度: {distance:.3f})")
                
                # 3. 搜索相关的知识关系
                if hasattr(self.rag_manager.rag, 'relationships_vdb') and self.rag_manager.rag.relationships_vdb:
                    rel_results = await self.rag_manager.rag.relationships_vdb.query(question, top_k=8)
                    if rel_results:
                        relevant_relations = len([r for r in rel_results if r.get('distance', 0) > 0.3])
                        print(f"  [关系] 相关知识关系: {relevant_relations} 个")
                        
                        # 显示最相关的关系
                        if relevant_relations > 0:
                            print("    [详情] 主要知识关系:")
                            for i, rel_data in enumerate([r for r in rel_results if r.get('distance', 0) > 0.3][:3], 1):
                                distance = rel_data.get('distance', 0)
                                src = rel_data.get('src_id', rel_data.get('source_id', '?'))
                                tgt = rel_data.get('tgt_id', rel_data.get('target_id', '?'))
                                print(f"      [{i}] {src} ↔ {tgt} (相似度: {distance:.3f})")
                
            except Exception as search_error:
                logger.warning(f"搜索过程出错: {search_error}")
                print(f"  [警告] 搜索过程出错，使用简化统计: {search_error}")
                
                
                print("\n[总览] 知识库总体规模参考:")
                try:
                    import json
                    import os
                    storage_dir = self.rag_manager.working_dir
                    
                    chunk_count = 0
                    entity_count = 0
                    rel_count = 0
                    
                    chunks_file = os.path.join(storage_dir, 'vdb_chunks.json')
                    if os.path.exists(chunks_file):
                        with open(chunks_file, 'r', encoding='utf-8') as f:
                            chunks_data = json.load(f)
                            if 'data' in chunks_data:
                                chunk_count = len(chunks_data['data'])
                    
                    entities_file = os.path.join(storage_dir, 'vdb_entities.json')
                    if os.path.exists(entities_file):
                        with open(entities_file, 'r', encoding='utf-8') as f:
                            entities_data = json.load(f)
                            if 'data' in entities_data:
                                entity_count = len(entities_data['data'])
                    
                    relationships_file = os.path.join(storage_dir, 'vdb_relationships.json')
                    if os.path.exists(relationships_file):
                        with open(relationships_file, 'r', encoding='utf-8') as f:
                            relationships_data = json.load(f)
                            if 'data' in relationships_data:
                                rel_count = len(relationships_data['data'])
                    
                    print(f"  [文档] 可用文档片段: {chunk_count} 个")
                    print(f"  [实体] 可用医疗实体: {entity_count} 个")
                    print(f"  [关系] 可用知识关系: {rel_count} 个")
                    print(f"  [信息] 系统从这些知识中筛选相关内容来回答您的问题")
                    
                except Exception as file_error:
                    print(f"  [警告] 无法获取统计信息: {file_error}")
            
            # 显示引用总计
            total_relevant = relevant_chunks + relevant_entities + relevant_relations
            if total_relevant > 0:
                print(f"  [合计] 本次查询引用总计: {total_relevant} 个知识单元")
            
            # 显示查询模式说明
            mode_descriptions = {
                'naive': '直接检索相关文档片段，快速获取原始信息',
                'local': '基于局部知识图谱推理，专注于相关实体和关系',
                'global': '基于全局知识图谱推理，提供更广泛的背景信息',
                'hybrid': '结合多种推理方式，提供最全面和准确的答案'
            }
            
            print(f"\n[模式] {mode.upper()}模式特点:")
            print(f"  {mode_descriptions.get(mode.lower(), '未知模式')}")
            
            if total_relevant > 0:
                print(f"\n[提示] 提示: {mode.upper()}模式基于以上{total_relevant}个相关知识单元为您生成了答案")
            else:
                print(f"\n[提示] 提示: {mode.upper()}模式基于知识库中的相关信息为您生成了答案")
            
        except Exception as e:
            logger.error(f"显示引用信息失败: {e}")
            print(f"[警告] 无法显示引用信息: {e}")
            print()
    
    async def run_interactive_mode(self):
        """智能交互式查询模式 - 支持模式切换指令"""
        logger.info("进入智能交互式查询模式")
        logger.info("=" * 50)
        
        # 当前查询模式
        current_mode = "hybrid"  # 默认使用HYBRID模式
        
        # 显示帮助信息
        self.show_help_info(current_mode)
        
        while True:
            try:
                print(f"\n[模式] 当前模式: {current_mode.upper()}")
                print("请输入您的医疗相关问题 (或输入指令):")
                user_input = input(">>> ").strip()
                
                # 处理退出指令
                if user_input.lower() in ['quit', 'exit', '退出', 'q', 'bye']:
                    logger.info("感谢使用医疗RAG系统！")
                    break
                
                # 处理帮助指令
                if user_input.lower() in ['help', 'h', '帮助', '?']:
                    self.show_help_info(current_mode)
                    continue
                
                # 处理模式切换指令
                if user_input.lower().startswith('/'):
                    new_mode = self.handle_mode_command(user_input, current_mode)
                    if new_mode != current_mode:
                        current_mode = new_mode
                        print(f"[成功] 已切换到 {current_mode.upper()} 模式")
                    continue
                
                # 处理空输入
                if not user_input:
                    continue
                
                # 执行查询
                await self.query_with_mode(user_input, current_mode)
                
            except KeyboardInterrupt:
                logger.info("用户中断，退出系统")
                break
            except Exception as e:
                logger.error(f"查询过程中出现错误: {e}")
                print(f"[错误] 查询错误: {e}")
                continue
    
    def show_help_info(self, current_mode: str):
        """显示帮助信息"""
        print(f"""
[系统] 医疗RAG智能问答系统 - 当前模式: {current_mode.upper()}
{'='*60}

[指令] 可用指令:
  /naive    - 切换到NAIVE模式 (基础检索)
  /local    - 切换到LOCAL模式 (局部推理)  
  /global   - 切换到GLOBAL模式 (全局推理)
  /hybrid   - 切换到HYBRID模式 (混合推理，推荐)
  /help 或 h - 显示此帮助信息
  /quit 或 q - 退出系统

[模式] 查询模式说明:
  • NAIVE:  快速文档检索，显示最相关的原始文档片段
  • LOCAL:  基于局部知识图谱进行推理，适合具体问题
  • GLOBAL: 基于全局知识图谱进行推理，适合概括性问题  
  • HYBRID: 结合多种推理方式，提供最全面的答案 (默认推荐)

[示例] 示例问题:
  • "糖尿病有什么症状？"
  • "高血压的治疗方法"
  • "心肌梗死的诊断标准"

[警告] 医疗免责声明: 本系统仅供参考，不能替代专业医疗建议
{'='*60}
        """)
    
    def handle_mode_command(self, command: str, current_mode: str) -> str:
        """处理模式切换指令"""
        command = command.lower().strip()
        
        mode_map = {
            '/naive': 'naive',
            '/local': 'local', 
            '/global': 'global',
            '/hybrid': 'hybrid',
            '/n': 'naive',
            '/l': 'local',
            '/g': 'global', 
            '/h': 'hybrid'
        }
        
        if command in mode_map:
            new_mode = mode_map[command]
            if new_mode == current_mode:
                print(f"[信息] 当前已经是 {current_mode.upper()} 模式")
            return new_mode
        elif command in ['/help', '/h', '/?']:
            self.show_help_info(current_mode)
            return current_mode
        elif command in ['/quit', '/q', '/exit']:
            return 'quit'
        else:
            print(f"[错误] 未知指令: {command}")
            print("[提示] 输入 /help 查看可用指令")
            return current_mode
    
    async def cleanup(self):
        """清理资源 - 基于您的最终版本"""
        try:
            if self.rag_manager:
                await self.rag_manager.cleanup()
            logger.info("系统资源清理完成")
        except Exception as e:
            logger.error(f"资源清理时出错: {e}")

async def main():
    """主程序 - 基于medical_rag_kb_official.py最终优化版本"""
    system = None
    try:
        # 1. 创建系统实例
        system = MedicalRAGSystem()
        
        # 2. 初始化系统
        if not await system.initialize():
            logger.error("系统初始化失败，退出程序")
            return
        
        # 3. 加载知识库（如果需要）
        if not await system.load_knowledge_base():
            logger.error("知识库加载失败，退出程序")
            return
        
        # 4. 启动智能交互模式
        print("\n[启动] 启动医疗RAG智能问答系统...")
        print("[模式] 默认使用HYBRID模式，支持指令切换模式")
        print("[帮助] 输入 /help 查看所有可用指令")
        
        await system.run_interactive_mode()
        
        logger.info("程序运行完成")
        
    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        if system:
            await system.cleanup()
        logger.info("程序执行完成！")

if __name__ == "__main__":
    asyncio.run(main()) 