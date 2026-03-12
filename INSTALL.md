# 🏥 医疗RAG套件 - 安装指南

基于LightRAG的热插拔医疗知识问答系统安装说明

## 📋 系统要求

### 必需组件
- **Python**: 3.8+ (推荐3.9+)
- **Ollama**: 最新版本
- **操作系统**: Windows 10+ / Linux / macOS
- **内存**: 最少8GB RAM (推荐16GB+)
- **磁盘**: 至少10GB可用空间

### 推荐硬件
- **GPU**: 支持CUDA的显卡 (可选，用于加速)
- **CPU**: 4核心以上
- **网络**: 稳定的互联网连接 (用于下载模型)

## 🚀 快速安装 (Windows)

### 1. 安装Ollama
```bash
# 访问 https://ollama.ai/download 下载Windows版本
# 或使用winget安装
winget install Ollama.Ollama
```

### 2. 启动Ollama服务
```bash
ollama serve
```

### 3. 下载推荐模型
```bash
# 推荐选项1：轻量级模型 (1.5B参数)
ollama pull deepseek-r1:1.5b

# 推荐选项2：更强性能 (7B参数，需要更多内存)  
ollama pull qwen2.5:7b

# 嵌入模型 (必需)
ollama pull nomic-embed-text
```

### 4. 克隆并运行项目
```bash
# 克隆项目
git clone <your-repo-url>
cd Medical-RAG-Kit

# 运行启动脚本
start.bat
```

## 📚 手动安装 (高级用户)

### 1. 创建Python虚拟环境
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS  
source venv/bin/activate
```

### 2. 安装Python依赖
```bash
pip install -r requirements.txt
```

### 3. 验证安装
```bash
python -c "import lightrag; print('LightRAG安装成功')"
```

### 4. 启动系统
```bash
python main.py
```

## ⚙️ 配置说明

### 模型配置 (`configs/model_config.yaml`)

```yaml
# 切换到不同模型只需修改model名称
llm_model:
  name: "deepseek-r1:1.5b"  # 或 "qwen2.5:7b"
  temperature: 0.1
  max_tokens: 2048

embedding_model:
  name: "nomic-embed-text"
```

### 知识库配置 (`configs/medical_config.yaml`)

```yaml
knowledge_categories:
  textbooks:
    path: "knowledge_bases/medical_textbooks"
    priority: 1
  guidelines:
    path: "knowledge_bases/clinical_guidelines"  
    priority: 2
  drugs:
    path: "knowledge_bases/drug_database"
    priority: 3
```

## 📖 使用说明

### 1. 添加医疗文档

将医疗文档放入对应文件夹：
- `knowledge_bases/medical_textbooks/` - 医学教科书
- `knowledge_bases/clinical_guidelines/` - 临床指南
- `knowledge_bases/drug_database/` - 药物数据库

支持格式：`.pdf`, `.txt`, `.md`, `.docx`

### 2. 启动系统

```bash
# Windows
start.bat

# 或直接运行
python main.py
```

### 3. 交互式使用

```
🩺 医疗助手 [hybrid] > 高血压的治疗方案是什么？

📋 回答:
根据医疗知识库，高血压的治疗方案包括...

🩺 医疗助手 [hybrid] > /help
💡 可用命令：...
```

### 4. 热插拔功能

**模型热插拔**：
1. 修改 `configs/model_config.yaml`
2. 在程序中运行 `/reload`

**知识库热插拔**：
1. 直接添加/删除 `knowledge_bases/` 中的文件
2. 系统会自动检测并更新

## 🔧 高级配置

### LightRAG参数调优 (`configs/rag_config.yaml`)

```yaml
knowledge_graph:
  chunk_token_size: 1200      # 文本块大小
  chunk_overlap_token_size: 100  # 重叠大小

default_query_mode: "hybrid"   # 查询模式
```

### 医疗安全设置 (`configs/medical_config.yaml`)

```yaml
medical_safety:
  disclaimer_enabled: true    # 免责声明
  no_diagnosis: true         # 禁止诊断
  no_prescription: true      # 禁止开药
  safety_check: true         # 安全检查
```

## 🔍 故障排除

### 常见问题

**Q: Ollama服务连接失败**
```bash
A: 确保Ollama服务正在运行
   ollama serve
   
   检查端口11434是否被占用
```

**Q: 模型下载失败**
```bash
A: 检查网络连接，可能需要代理
   export HTTP_PROXY=http://your-proxy:port
   export HTTPS_PROXY=http://your-proxy:port
```

**Q: 内存不足**
```bash
A: 尝试使用更小的模型
   修改configs/model_config.yaml中的模型为deepseek-r1:1.5b
```

**Q: LightRAG导入失败**
```bash
A: 确保安装了最新版本
   pip install --upgrade lightrag
```

### 日志查看

程序运行日志保存在：
- `logs/medical_rag.log` - 详细日志
- 控制台输出 - 实时状态

### 性能优化

1. **使用GPU加速** (如果有NVIDIA显卡)：
   ```bash
   # 安装CUDA版本的依赖
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **调整并发数**：
   ```yaml
   # configs/rag_config.yaml
   entity_extract:
     max_async: 2  # 降低并发数减少内存使用
   ```

3. **使用更小的chunk size**：
   ```yaml
   knowledge_graph:
     chunk_token_size: 800  # 减少内存使用
   ```

## 📝 更新日志

### v1.0
- 基于LightRAG的核心RAG系统
- 支持模型和知识库热插拔
- 医疗领域安全检查
- Windows友好的安装体验

## 🆘 获取帮助

1. **查看日志**: `logs/medical_rag.log`
2. **程序内帮助**: 运行程序后输入 `/help`
3. **配置检查**: 运行 `/stats` 查看系统状态

## ⚖️ 免责声明

本系统仅供医疗专业人员参考学习使用，不能替代专业医疗诊断和治疗建议。如有健康问题，请咨询专业医疗人员。 