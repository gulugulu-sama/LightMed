"""
医疗RAG系统 - 小规模优化版本
200条记录，10条一块，快速测试和验证
"""

import os
import asyncio
import json
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./medical_json_small_storage"
JSON_FILE_PATH = "./knowledgebase1/medical_subset_200.json"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    """初始化RAG系统 - 超时问题完全修复版"""
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5:7b",
        llm_model_max_token_size=8192,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 8192},
            "timeout": 1800,  # 增加到30分钟
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model="bge-m3:latest",
                host="http://localhost:11434",
                timeout=1800,  # 增加到30分钟
            ),
        ),
    )
    
    await rag.initialize_storages()
    await initialize_pipeline_status()
    
    return rag

def load_json_medical_data_small():
    """加载200条记录的小数据集"""
    try:
        print(f"📚 正在加载小规模JSON医疗数据库: {JSON_FILE_PATH}")
        
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            medical_records = json.load(f)
        
        print(f"✅ 成功加载 {len(medical_records)} 条医疗记录")
        
        # 按更小的块进行分组，减少单次处理压力
        chunk_size = 3  # 进一步减少到3条记录一块
        chunks = []
        
        for i in range(0, len(medical_records), chunk_size):
            chunk_records = medical_records[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (len(medical_records) + chunk_size - 1) // chunk_size
            
            # 构建块内容 - 更简洁的格式
            chunk_content = f"""医疗知识库数据块 {chunk_num}/{total_chunks}
包含疾病记录: {len(chunk_records)} 条

"""
            
            # 处理每条记录 - 简化格式减少token消耗
            for j, record in enumerate(chunk_records, 1):
                disease_name = record.get('name', '未知疾病')
                description = record.get('desc', '无描述')
                symptoms = record.get('symptom', [])
                cause = record.get('cause', '病因不明')
                treatment_dept = record.get('cure_department', [])
                treatment_method = record.get('cure_way', [])
                
                # 简化格式，只保留核心信息
                record_text = f"""
疾病{j}: {disease_name}
描述: {description}
症状: {', '.join(symptoms) if symptoms else '症状不详'}
病因: {cause}
科室: {', '.join(treatment_dept) if treatment_dept else '未知'}
治疗: {', '.join(treatment_method) if treatment_method else '治疗方法不详'}
"""
                chunk_content += record_text
            
            chunks.append({
                'chunk_id': chunk_num,
                'total_chunks': total_chunks,
                'record_count': len(chunk_records),
                'content': chunk_content,
                'size': len(chunk_content)
            })
        
        print(f"📊 数据分块结果:")
        print(f"  总块数: {len(chunks)}")
        print(f"  每块记录数: {chunk_size}")
        print(f"  平均块大小: {sum(chunk['size'] for chunk in chunks) // len(chunks):,} 字符")
        
        return chunks
        
    except FileNotFoundError:
        print(f"❌ 文件不存在: {JSON_FILE_PATH}")
        print("请先运行 fix_json_and_analyze_small.py 生成数据文件")
        return []
    except Exception as e:
        print(f"❌ 加载JSON数据失败: {e}")
        return []

def medical_safety_check(question: str) -> str:
    """医疗安全检查"""
    diagnosis_keywords = ['诊断', '确诊', '是不是', '什么病', '得了', '患了', '我是否']
    if any(keyword in question for keyword in diagnosis_keywords):
        return '⚠️ 医疗安全提醒：本系统不提供医疗诊断，请咨询专业医生。'
    
    prescription_keywords = ['开药', '用药量', '吃多少', '处方', '药物剂量', '怎么吃药', '服用方法']
    if any(keyword in question for keyword in prescription_keywords):
        return '⚠️ 医疗安全提醒：本系统不提供用药指导，请咨询专业医生。'
    
    emergency_keywords = ['急救', '紧急', '突发', '抢救', '危险', '生命危险']
    if any(keyword in question for keyword in emergency_keywords):
        return '🚨 紧急情况请立即拨打120或前往最近的医院急诊科！'
    
    return ""

async def test_basic_functions(rag):
    """测试基础函数"""
    print("🧪 测试Embedding函数...")
    try:
        test_text = ["糖尿病症状测试"]
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print(f"Embedding测试成功，维度: {embedding_dim}")
    except Exception as e:
        print(f"Embedding测试失败: {e}")
        return False
    
    print("\n🧪 测试LLM函数...")
    try:
        result = await rag.llm_model_func("请简单回复：你好")
        print(f"LLM测试成功: {result[:50]}...")
    except Exception as e:
        print(f"LLM测试失败: {e}")
        return False
    
    return True

async def load_chunks_to_rag(rag, chunks):
    """将数据块加载到RAG系统 - 小规模优化"""
    print(f"\n📚 开始加载 {len(chunks)} 个小数据块...")
    
    success_count = 0
    failed_count = 0
    
    for i, chunk in enumerate(chunks, 1):
        try:
            chunk_id = chunk['chunk_id']
            record_count = chunk['record_count']
            chunk_size = chunk['size']
            
            print(f"📄 [{i}/{len(chunks)}] 插入块{chunk_id}: {record_count}条记录 ({chunk_size:,}字符)")
            
            await rag.ainsert(chunk['content'])
            success_count += 1
            
            print(f"   ✅ 完成")
            
            # 增加延迟，给系统更多处理时间
            if i < len(chunks):
                await asyncio.sleep(1)  # 增加到3秒延迟
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"   ❌ 失败: {error_msg}")
            failed_count += 1
            
            # 超时处理和重试
            if 'timeout' in error_msg.lower() or 'ReadTimeout' in error_msg:
                print(f"   ⏳ 超时错误，等待30秒后继续...")
                await asyncio.sleep(30)
            elif 'connection' in error_msg.lower():
                print(f"   🔌 连接错误，等待10秒后继续...")
                await asyncio.sleep(10)
            
            continue
    
    print(f"\n📊 加载统计:")
    print(f"✅ 成功: {success_count}/{len(chunks)}")
    print(f"❌ 失败: {failed_count}")
    
    return success_count > 0

async def query_with_mode(rag, question, mode):
    """使用指定模式查询"""
    print(f"\n {mode.upper()} 模式查询:")
    print("-" * 30)
    try:
        response = await rag.aquery(
            question,
            param=QueryParam(mode=mode)
        )
        print(response)
        return True
    except Exception as e:
        print(f"❌ {mode}模式查询失败: {e}")
        return False

def check_existing_knowledge_base():
    """检查是否存在现有知识库"""
    working_dir = Path(WORKING_DIR)
    if not working_dir.exists():
        return False
    
    key_files = [
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json", 
        "vdb_entities.json",
        "vdb_relationships.json"
    ]
    
    existing_files = []
    for file in key_files:
        file_path = working_dir / file
        if file_path.exists() and file_path.stat().st_size > 50:
            existing_files.append(file)
    
    if len(existing_files) >= 3:
        print(f" 发现现有小规模知识库: {len(existing_files)}/{len(key_files)} 个文件")
        return True
    else:
        print(f" 知识库不完整: {len(existing_files)}/{len(key_files)} 个文件")
        return False

async def main():
    rag = None
    try:
        print(" 医疗RAG系统 - 小规模测试版本")
        print("=" * 50)
        print(" 超时问题完全修复配置:")
        print("  1. 200条医疗记录")
        print("  2. 3条记录一块（约67块）")
        print("  3. 1800秒超时（30分钟），8192 token")
        print("  4. 移除不支持的client_timeout参数")
        print("  5. 块间3秒延迟，减少服务器压力")
        print("=" * 50)
        
        # 1. 检查现有知识库
        has_existing_kb = check_existing_knowledge_base()
        
        if has_existing_kb:
            print(" 发现现有知识库，跳过数据加载")
            should_load_data = False
        else:
            print(" 需要构建新知识库")
            should_load_data = True
            
            # 清理旧文件
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
            
            print(" 清理旧文件...")
            for file in files_to_delete:
                file_path = os.path.join(WORKING_DIR, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"  删除: {file}")
        
        # 2. 初始化RAG系统
        print("\n 初始化RAG系统...")
        rag = await initialize_rag()
        print(" RAG系统初始化完成")
        
        # 3. 测试基础函数
        if not await test_basic_functions(rag):
            print(" 基础功能测试失败")
            return
        
        # 4. 加载数据
        if should_load_data:
            print("\n 加载小规模医疗数据...")
            chunks = load_json_medical_data_small()
            
            if not chunks:
                print(" 没有找到数据文件")
                return
            
            # 5. 插入数据
            if not await load_chunks_to_rag(rag, chunks):
                print(" 数据加载失败")
                return
        else:
            print(" 使用现有知识库")
        
        # 6. 测试查询
        print("\n" + "=" * 50)
        print(" 开始测试查询")
        print("=" * 50)
        
        test_questions = [
            "糖尿病有什么症状？",
            "高血压怎么治疗？"
        ]
        
        for i, question in enumerate(test_questions, 1):
            safety_warning = medical_safety_check(question)
            if safety_warning:
                print(f"\n{safety_warning}")
            
            print(f"\n{'='*60}")
            print(f"测试问题 {i}: {question}")
            
            # 只测试naive和local模式，减少时间
            for mode in ['naive', 'local']:
                await query_with_mode(rag, question, mode)
                await asyncio.sleep(1)
        
        # 7. 简单交互
        print("\n" + "=" * 50)
        print(" 简单交互模式 (输入 'quit' 退出)")
        print("=" * 50)
        
        interaction_count = 0
        max_interactions = 3  # 限制交互次数
        
        while interaction_count < max_interactions:
            try:
                question = input(f"\n [{interaction_count+1}/{max_interactions}] 请输入问题: ")
                
                if question.lower() in ['quit', 'exit', '退出', 'q']:
                    break
                
                if not question.strip():
                    continue
                
                safety_warning = medical_safety_check(question)
                if safety_warning:
                    print(f"\n{safety_warning}")
                
                # 只用naive模式快速回答
                await query_with_mode(rag, question, 'global')
                interaction_count += 1
                
            except KeyboardInterrupt:
                print("\n\n 用户中断")
                break
            except Exception as e:
                print(f"\n 查询错误: {e}")
                continue
        
        print("\n 小规模测试完成！")

    except Exception as e:
        print(f"\n 程序错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if rag:
            try:
                await rag.llm_response_cache.index_done_callback()
                await rag.finalize_storages()
                print(" 资源清理完成")
            except Exception as e:
                print(f" 清理错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())
    print("\n🎉 小规模测试结束！") 