#!/usr/bin/env python3
"""
🏥 医疗RAG套件 - 快速启动验证
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

def print_banner():
    """打印启动横幅"""
    print("=" * 60)
    print("🏥 医疗RAG套件 - 快速启动验证")
    print("=" * 60)
    print()

def check_python_version():
    """检查Python版本"""
    print("1. 检查Python版本...")
    version = sys.version_info
    print(f"   当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("   ❌ Python版本过低，LightRAG需要Python 3.9+")
        print("   请创建新的conda环境：")
        print("   conda create -n medical_rag python=3.10 -y")
        print("   conda activate medical_rag")
        return False
    else:
        print("   ✅ Python版本符合要求")
        return True

def check_dependencies():
    """检查关键依赖"""
    print("\n2. 检查关键依赖...")
    
    dependencies = [
        ("lightrag", "LightRAG框架"),
        ("ollama", "Ollama客户端"),
        ("yaml", "YAML配置"),
        ("loguru", "日志库"),
        ("watchdog", "文件监听"),
    ]
    
    missing = []
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"   ✅ {description}: {module}")
        except ImportError:
            print(f"   ❌ {description}: {module} 未安装")
            missing.append(module)
    
    if missing:
        print(f"\n   缺失依赖: {', '.join(missing)}")
        print("   请运行: pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple")
        return False
    
    return True

def check_ollama_service():
    """检查Ollama服务"""
    print("\n3. 检查Ollama服务...")
    
    try:
        from src.core.model_manager import get_model_manager
        model_manager = get_model_manager()
        
        if model_manager.check_ollama_service():
            print("   ✅ Ollama服务运行正常")
            
            # 检查可用模型
            available_models = model_manager.list_available_models()
            print(f"   ✅ 可用模型: {len(available_models)} 个")
            
            # 检查推荐模型
            recommended = ["deepseek-r1:1.5b", "qwen2.5:7b", "nomic-embed-text"]
            for model in recommended:
                if model in available_models:
                    print(f"   ✅ 推荐模型 {model}: 已安装")
                else:
                    print(f"   ⚠️  推荐模型 {model}: 未安装")
                    print(f"      安装命令: ollama pull {model}")
            
            return True
        else:
            print("   ❌ Ollama服务未运行")
            print("   启动命令: ollama serve")
            return False
            
    except Exception as e:
        print(f"   ❌ 检查Ollama服务失败: {e}")
        return False

def check_config_files():
    """检查配置文件"""
    print("\n4. 检查配置文件...")
    
    config_files = [
        ("configs/model_config.yaml", "模型配置"),
        ("configs/rag_config.yaml", "RAG配置"), 
        ("configs/medical_config.yaml", "医疗配置")
    ]
    
    all_exists = True
    for file_path, description in config_files:
        if os.path.exists(file_path):
            print(f"   ✅ {description}: {file_path}")
        else:
            print(f"   ❌ {description}: {file_path} 不存在")
            all_exists = False
    
    return all_exists

def check_knowledge_base():
    """检查知识库"""
    print("\n5. 检查知识库结构...")
    
    kb_dirs = [
        ("knowledge_bases", "知识库根目录"),
        ("knowledge_bases/medical_textbooks", "医学教科书"),
        ("knowledge_bases/clinical_guidelines", "临床指南"),
        ("knowledge_bases/drug_database", "药物数据库")
    ]
    
    for dir_path, description in kb_dirs:
        if os.path.exists(dir_path):
            if os.path.isdir(dir_path):
                files = os.listdir(dir_path)
                print(f"   ✅ {description}: {len(files)} 个文件")
            else:
                print(f"   ⚠️  {description}: 不是目录")
        else:
            print(f"   ⚠️  {description}: 不存在")

async def test_basic_import():
    """测试基础导入"""
    print("\n6. 测试核心组件导入...")
    
    try:
        from src.core.model_manager import get_model_manager
        from src.core.kb_manager import get_kb_manager
        from src.core.rag_manager import get_rag_manager
        
        print("   ✅ 模型管理器导入成功")
        print("   ✅ 知识库管理器导入成功") 
        print("   ✅ RAG管理器导入成功")
        
        # 测试LightRAG导入
        from lightrag import LightRAG, QueryParam
        from lightrag.llm.ollama import ollama_model_complete, ollama_embed
        print("   ✅ LightRAG核心组件导入成功")
        
        return True
    except Exception as e:
        print(f"   ❌ 导入失败: {e}")
        return False

def print_summary(all_checks_passed):
    """打印总结"""
    print("\n" + "=" * 60)
    
    if all_checks_passed:
        print("🎉 恭喜！所有检查都通过了！")
        print("\n📋 系统就绪，您可以:")
        print("   1. 运行主程序: python main.py")
        print("   2. 添加医疗文档到 knowledge_bases/ 目录")
        print("   3. 开始使用医疗RAG问答系统")
        print("\n💡 快速上手:")
        print("   - 将医疗文档放入对应的知识库文件夹")
        print("   - 根据需要调整 configs/ 中的配置")
        print("   - 使用 /help 查看系统命令")
    else:
        print("⚠️  系统检查发现问题，请根据上述提示解决")
        print("\n🔧 常见解决方案:")
        print("   1. Python版本: conda create -n medical_rag python=3.10")
        print("   2. 依赖安装: pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple")
        print("   3. Ollama服务: ollama serve")
        print("   4. 下载模型: ollama pull deepseek-r1:1.5b")
    
    print("=" * 60)

async def main():
    """主函数"""
    print_banner()
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_ollama_service(),
        check_config_files(),
        True,  # check_knowledge_base() 总是返回True，只是检查
        await test_basic_import()
    ]
    
    # 知识库检查不影响整体结果
    check_knowledge_base()
    
    all_passed = all(checks)
    print_summary(all_passed)
    
    return all_passed

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1) 