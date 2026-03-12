"""
模型管理器 - 基于实践测试优化的最终版本
集成正确的超时配置、并发控制和错误处理
"""

import os
import yaml
import asyncio
import requests
from typing import Dict, Any, Optional
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

class ModelManager:
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        self.config_path = config_path
        self.current_config = None
        self.semaphore = asyncio.Semaphore(1)  # 控制并发
        self.ollama_host = "http://localhost:11434"
        
    def load_config(self) -> Dict[str, Any]:
        """加载模型配置"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        self.current_config = config
        return config
    
    def check_ollama_status(self) -> bool:
        """检查Ollama服务状态 - 优化超时设置"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"WARNING: Ollama服务检查失败: {e}")
            return False
    
    def get_available_models(self) -> list:
        """获取可用模型列表"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=15)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            print(f"WARNING: 获取模型列表失败: {e}")
        return []
    
    def check_model_exists(self, model_name: str) -> bool:
        """检查模型是否存在"""
        available_models = self.get_available_models()
        return model_name in available_models
    
    async def pull_model_if_needed(self, model_name: str) -> bool:
        """如果需要，拉取模型 - 优化超时和错误处理"""
        if self.check_model_exists(model_name):
            print(f"SUCCESS: 模型 {model_name} 已存在")
            return True
            
        print(f"INFO: 正在拉取模型 {model_name}...")
        print("INFO: 这可能需要几分钟时间，请耐心等待...")
        
        try:
            response = requests.post(
                f"{self.ollama_host}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=1800  # 30分钟超时，模型下载可能很久
            )
            
            if response.status_code == 200:
                # 显示下载进度
                for line in response.iter_lines():
                    if line:
                        try:
                            import json
                            data = json.loads(line)
                            if 'status' in data:
                                print(f"INFO: {data['status']}")
                        except:
                            pass
                
                print(f"SUCCESS: 模型 {model_name} 拉取完成")
                return True
            else:
                print(f"ERROR: 模型 {model_name} 拉取失败，状态码: {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"ERROR: 模型 {model_name} 拉取超时，请检查网络连接")
            return False
        except Exception as e:
            print(f"ERROR: 拉取模型时出错: {e}")
            return False
    
    def get_llm_function(self):
        """获取LLM函数 - 使用官方推荐方式"""
        # 直接使用LightRAG官方的ollama函数，已经过优化
        return ollama_model_complete
    
    def get_embedding_function(self):
        """获取Embedding函数 - 基于成功版本的配置"""
        config = self.current_config or self.load_config()
        embed_config = config['embedding']
        
        return EmbeddingFunc(
            embedding_dim=embed_config['dimension'],
            max_token_size=embed_config['max_tokens'],
            func=lambda texts: ollama_embed(
                texts,
                embed_model=embed_config['model'],
                host=self.ollama_host,
                timeout=embed_config.get('timeout', 600)  # 基于您的成功版本的600秒超时设置
            ),
        )
    
    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置参数 - 基于实践测试优化"""
        config = self.current_config or self.load_config()
        llm_config = config['llm']
        
        return {
            'model_name': llm_config['model'],
            'max_token_size': llm_config['max_tokens'],
            'model_kwargs': {
                'host': self.ollama_host,
                'options': {
                    'num_ctx': llm_config['context_window'],
                    'temperature': llm_config.get('temperature', 0.1)
                },
                'timeout': llm_config.get('timeout', 600)  # 基于您的成功版本的600秒超时
            }
        }
    
    async def initialize_models(self) -> bool:
        """初始化模型 - 改进的错误处理和验证"""
        config = self.load_config()
        
        print("INFO: 初始化模型管理器...")
        
        # 检查Ollama服务
        print("INFO: 检查Ollama服务状态...")
        if not self.check_ollama_status():
            print("ERROR: Ollama服务未运行")
            print("INFO: 请确保Ollama已安装并运行: ollama serve")
            return False
        
        print("SUCCESS: Ollama服务运行正常")
        
        # 检查并拉取LLM模型
        llm_model = config['llm']['model']
        print(f"INFO: 检查LLM模型: {llm_model}")
        if not await self.pull_model_if_needed(llm_model):
            print(f"ERROR: LLM模型 {llm_model} 准备失败")
            return False
        
        # 检查并拉取Embedding模型
        embed_model = config['embedding']['model']
        print(f"INFO: 检查Embedding模型: {embed_model}")
        if not await self.pull_model_if_needed(embed_model):
            print(f"ERROR: Embedding模型 {embed_model} 准备失败")
            return False
        
        print("SUCCESS: 所有模型准备就绪")
        
        # 验证模型配置
        try:
            llm_config = self.get_llm_config()
            embed_func = self.get_embedding_function()
            print("SUCCESS: 模型配置验证通过")
            return True
        except Exception as e:
            print(f"ERROR: 模型配置验证失败: {e}")
            return False
    
    def hot_reload_config(self) -> bool:
        """热重载配置"""
        try:
            old_config = self.current_config
            new_config = self.load_config()
            
            if old_config != new_config:
                print("INFO: 检测到模型配置变更...")
                print(f"   LLM模型: {new_config['llm']['model']}")
                print(f"   Embedding模型: {new_config['embedding']['model']}")
                return True
            return False
        except Exception as e:
            print(f"ERROR: 配置重载失败: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        config = self.current_config or self.load_config()
        
        return {
            'llm_model': config['llm']['model'],
            'embedding_model': config['embedding']['model'],
            'llm_context_window': config['llm']['context_window'],
            'embedding_dimension': config['embedding']['dimension'],
            'ollama_host': self.ollama_host,
            'available_models': self.get_available_models()
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取完整模型配置 - RAG管理器需要的格式"""
        config = self.current_config or self.load_config()
        
        return {
            'llm': {
                'model': config['llm']['model'],
                'max_tokens': config['llm']['max_tokens'],
                'context_window': config['llm']['context_window'],
                'timeout': config['llm'].get('timeout', 600)
            },
            'embedding': {
                'model': config['embedding']['model'],
                'dimension': config['embedding']['dimension'],
                'max_tokens': config['embedding']['max_tokens'],
                'timeout': config['embedding'].get('timeout', 600)
            },
            'ollama': {
                'base_url': self.ollama_host
            }
        }


# 全局模型管理器实例
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """获取模型管理器实例"""
    return model_manager 