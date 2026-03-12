"""
医疗RAG系统 - 完全按照官方示例标准的版本
解决超时问题的根本原因分析和修复
"""

import os
import asyncio
import glob
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./medical_kb_official_storage"
KNOWLEDGE_BASE_DIR = "./knowledge_bases"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5:7b",
        llm_model_max_token_size=8192,  # 使用8192
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 8192},  # 使用8192
            "timeout": 1200,  # 大幅增加LLM超时：医疗文档实体提取需要更长时间
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,  # 官方默认8192
            func=lambda texts: ollama_embed(
                texts,
                embed_model="bge-m3:latest",
                host="http://localhost:11434",
                timeout=1200,  # 关键修复：BGE-M3处理大文本时Ollama客户端需要更长超时
                # BGE-M3专为长文本设计(最多8192 tokens)，但处理时间较长
                # 没有这个参数会导致httpx.ReadTimeout错误
            ),
        ),
    )
    
    # 官方标准：先初始化存储，再初始化pipeline状态
    await rag.initialize_storages()
    await initialize_pipeline_status()
    
    return rag

def load_knowledge_base_files():
    """加载知识库文件，支持大文档分块"""
    kb_path = Path(KNOWLEDGE_BASE_DIR)
    if not kb_path.exists():
        print(f"⚠️ 知识库目录不存在: {KNOWLEDGE_BASE_DIR}")
        return []
    
    files = []
    txt_files = glob.glob(str(kb_path / "**" / "*.txt"), recursive=True)
    
    print(f"📚 发现 {len(txt_files)} 个知识库文件:")
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            filename = os.path.basename(file_path)
            category = os.path.basename(os.path.dirname(file_path))
            
            # 如果文档超过3000字符，进行分块处理
            if len(content) > 3000:
                print(f"  📄 {category}/{filename} ({len(content):,} 字符) - 需要分块")
                
                # 分块处理：每块1000字符，保持段落完整性
                chunks = []
                chunk_size = 1000
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i + chunk_size]
                    
                    # 尝试在句号处分割，保持内容完整性
                    if i + chunk_size < len(content) and '。' in chunk:
                        last_period = chunk.rfind('。')
                        if last_period > chunk_size * 0.7:  # 如果句号位置合理
                            chunk = chunk[:last_period + 1]
                    
                    chunk_num = len(chunks) + 1
                    formatted_chunk = f"""文档名称: {filename} (第{chunk_num}部分)
文档分类: {category}
原文档大小: {len(content):,} 字符

{chunk}"""
                    
                    chunks.append({
                        'path': file_path,
                        'filename': f"{filename}_part{chunk_num}",
                        'category': category,
                        'content': formatted_chunk,
                        'size': len(chunk),
                        'is_chunk': True,
                        'chunk_num': chunk_num,
                        'total_chunks': 0  # 会在后面更新
                    })
                
                # 更新总块数
                for chunk in chunks:
                    chunk['total_chunks'] = len(chunks)
                
                files.extend(chunks)
                print(f"    ✂️ 分割为 {len(chunks)} 个块")
            else:
                # 小文档直接处理
                formatted_content = f"""文档名称: {filename}
文档分类: {category}

{content}"""
                
                files.append({
                    'path': file_path,
                    'filename': filename,
                    'category': category,
                    'content': formatted_content,
                    'size': len(content),
                    'is_chunk': False
                })
                
                print(f"  📄 {category}/{filename} ({len(content):,} 字符)")
            
        except Exception as e:
            print(f"⚠️ 读取文件失败 {file_path}: {e}")
    
    return files

def medical_safety_check(question: str) -> str:
    """医疗安全检查"""
    diagnosis_keywords = ['诊断', '确诊', '是不是', '什么病', '得了', '患了']
    if any(keyword in question for keyword in diagnosis_keywords):
        return '⚠️ 医疗安全提醒：本系统不提供医疗诊断，请咨询专业医生。'
    
    prescription_keywords = ['开药', '用药量', '吃多少', '处方', '药物剂量', '怎么吃药']
    if any(keyword in question for keyword in prescription_keywords):
        return '⚠️ 医疗安全提醒：本系统不提供用药指导，请咨询专业医生。'
    
    emergency_keywords = ['急救', '紧急', '突发', '抢救', '危险']
    if any(keyword in question for keyword in emergency_keywords):
        return '🚨 紧急情况请立即拨打120或前往最近的医院急诊科！'
    
    return ""

async def test_basic_functions(rag):
    """测试基础函数 - 按照官方示例"""
    print("🧪 测试Embedding函数...")
    try:
        test_text = ["这是一个测试句子。"]
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print(f"Embedding测试结果: {test_text}")
        print(f"检测到的嵌入维度: {embedding_dim}")
    except Exception as e:
        print(f"Embedding测试失败: {e}")
        return False
    
    print("\n🧪 测试LLM函数...")
    try:
        # 使用rag实例的llm_model_func，就像官方示例
        result = await rag.llm_model_func("你好，请简单回复")
        print(f"LLM测试结果: {result[:100]}..." if len(result) > 100 else result)
    except Exception as e:
        print(f"LLM测试失败: {e}")
        return False
    
    return True

async def load_documents_to_rag(rag, documents):
    """将文档加载到RAG系统 - 支持分块策略和错误恢复"""
    print(f"\n📚 开始加载 {len(documents)} 个文档块到RAG系统...")
    
    total_chars = sum(doc['size'] for doc in documents)
    print(f"📊 总文档大小: {total_chars:,} 字符")
    
    # 统计信息
    chunk_docs = [doc for doc in documents if doc.get('is_chunk', False)]
    normal_docs = [doc for doc in documents if not doc.get('is_chunk', False)]
    
    if chunk_docs:
        print(f"📋 包含 {len(normal_docs)} 个完整文档 + {len(chunk_docs)} 个分块")
    else:
        print(f"📋 包含 {len(normal_docs)} 个完整文档，无需分块")
    
    success_count = 0
    failed_docs = []
    
    for i, doc in enumerate(documents, 1):
        try:
            doc_name = doc['filename']
            if doc.get('is_chunk', False):
                doc_name = f"{doc['filename']} ({doc['chunk_num']}/{doc['total_chunks']})"
            
            print(f"📄 [{i}/{len(documents)}] 正在插入: {doc['category']}/{doc_name} ({doc['size']:,} 字符)")
            
            # 使用更长的超时时间进行插入
            await rag.ainsert(doc['content'])
            success_count += 1
            
            print(f"   ✅ 插入完成")
            
            # 在文档间添加小延迟，减少服务器压力
            if i < len(documents):
                await asyncio.sleep(1)
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"   ❌ 插入失败: {error_msg}")
            
            failed_docs.append({
                'doc': doc,
                'error': error_msg
            })
            
            # 如果是超时错误，等待更长时间再继续
            if 'timeout' in error_msg.lower() or 'ReadTimeout' in error_msg:
                print(f"   ⏳ 超时错误，等待30秒后继续...")
                await asyncio.sleep(30)
            
            continue
    
    print(f"\n📊 加载结果统计:")
    print(f"✅ 成功: {success_count}/{len(documents)} 个文档")
    print(f"❌ 失败: {len(failed_docs)} 个文档")
    
    if failed_docs and success_count > 0:
        print(f"📝 部分文档加载失败，但已有 {success_count} 个文档成功加载，可以继续使用系统")
        return True
    elif success_count > 0:
        print(f"🎉 所有文档加载成功！")
        return True
    else:
        print("❌ 所有文档都加载失败")
        return False

async def query_with_all_modes(rag, question):
    """使用所有模式查询"""
    safety_warning = medical_safety_check(question)
    if safety_warning:
        print(f"\n{safety_warning}")
    
    print(f"\n🤔 正在查询: {question}")
    print("=" * 60)
    
    modes = ['naive', 'local', 'global', 'hybrid']
    
    for mode in modes:
        print(f"\n🔍 {mode.upper()} 模式查询:")
        print("-" * 30)
        try:
            response = await rag.aquery(
                question,
                param=QueryParam(mode=mode)
            )
            print(response)
        except Exception as e:
            print(f"❌ {mode}模式查询失败: {e}")

def check_existing_knowledge_base():
    """检查是否存在完整的知识库"""
    working_dir = Path(WORKING_DIR)
    if not working_dir.exists():
        return False
    
    # 检查关键文件是否存在
    key_files = [
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json", 
        "vdb_entities.json",
        "vdb_relationships.json"
    ]
    
    existing_files = []
    for file in key_files:
        file_path = working_dir / file
        if file_path.exists() and file_path.stat().st_size > 100:  # 文件存在且不为空
            existing_files.append(file)
    
    if len(existing_files) >= 3:  # 至少有3个关键文件
        print(f"🎯 发现现有知识库: {len(existing_files)}/{len(key_files)} 个关键文件存在")
        return True
    else:
        print(f"❌ 知识库不完整: 只有 {len(existing_files)}/{len(key_files)} 个文件存在")
        return False

async def main():
    rag = None
    try:
        print("🏥 医疗RAG系统 - 智能分块版本")
        print("=" * 50)
        print("🔧 优化策略:")
        print("  1. 超时时间增加到1200秒（20分钟）")
        print("  2. 大文档(>3000字符)自动分块处理")
        print("  3. 智能清理：有完整知识库则跳过数据加载")
        print("  4. 分块大小1000字符，保持段落完整性")
        print("=" * 50)
        
        # 1. 检查现有知识库
        has_existing_kb = check_existing_knowledge_base()
        
        if has_existing_kb:
            print("✅ 发现完整的现有知识库，跳过数据加载阶段")
            should_load_data = False
        else:
            print("🔄 需要重新构建知识库")
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
            
            print("🗑️ 清理旧数据文件...")
            for file in files_to_delete:
                file_path = os.path.join(WORKING_DIR, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"  删除旧文件: {file}")
        
        # 2. 初始化RAG系统
        print("\n🚀 初始化RAG系统...")
        rag = await initialize_rag()
        print("✅ RAG系统初始化完成")
        
        # 3. 测试基础函数
        if not await test_basic_functions(rag):
            print("❌ 基础功能测试失败")
            return
        
        # 4. 根据知识库状态决定是否加载数据
        if should_load_data:
            print("\n📖 加载知识库文件...")
            kb_documents = load_knowledge_base_files()
            
            if not kb_documents:
                print("❌ 没有找到知识库文件")
                return
            
            # 5. 将文档插入RAG系统（使用分块策略）
            if not await load_documents_to_rag(rag, kb_documents):
                print("❌ 文档加载失败，但程序可以继续运行")
        else:
            print("⚡ 使用现有知识库，直接进入查询阶段")
        
        # 6. 测试查询
        print("\n" + "=" * 50)
        print("🧪 开始测试查询功能")
        print("=" * 50)
        
        test_questions = [
            "糖尿病有哪些症状？",
            "高血压的治疗方法有哪些？",
            "急性心肌梗死的诊断标准是什么？"
        ]
        
        print("🤖 自动测试模式:")
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*60}")
            print(f"测试问题 {i}/{len(test_questions)}")
            await query_with_all_modes(rag, question)
            
            if i < len(test_questions):
                print(f"\n⏳ 等待 2 秒后进行下一个测试...")
                await asyncio.sleep(2)
        
        # 7. 交互式查询
        print("\n" + "=" * 50)
        print("💬 进入交互式查询模式 (输入 'quit' 退出)")
        print("=" * 50)
        
        while True:
            try:
                print("\n请输入您的医疗相关问题:")
                question = input("❓ ")
                
                if question.lower() in ['quit', 'exit', '退出', 'q']:
                    print("👋 感谢使用医疗RAG系统！")
                    break
                
                if not question.strip():
                    continue
                
                await query_with_all_modes(rag, question)
                
            except KeyboardInterrupt:
                print("\n\n👋 用户中断，退出系统")
                break
            except Exception as e:
                print(f"\n❌ 查询过程中出现错误: {e}")
                continue

    except Exception as e:
        print(f"\n❌ 程序执行错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 官方标准的清理方式
        if rag:
            try:
                await rag.llm_response_cache.index_done_callback()
                await rag.finalize_storages()
                print("✅ RAG资源清理完成")
            except Exception as e:
                print(f"⚠️ 资源清理时出错: {e}")

if __name__ == "__main__":
    asyncio.run(main())
    print("\n🎉 测试完成！") 