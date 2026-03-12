<div align="center">


<img width="1566" height="459" alt="bf9caf707b13f146b3d5d7f2848dcd4c" src="https://github.com/user-attachments/assets/51f97ac6-6175-43a8-b855-c5eebfc15849" />



---

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![LightRAG](https://img.shields.io/badge/LightRAG-Latest-green?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-Local-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)
![For](https://img.shields.io/badge/For-Learners-red?style=flat-square)

</div>

---

# RAG : Zero to Hero

这是我的一个个人学习项目，也是我第一次尝试构建基于知识图谱的 RAG 系统。领域选择了医疗，使用的框架是 [LightRAG](https://github.com/HKUDS/LightRAG) 和 [Ollama](https://ollama.ai/)。

这个仓库记录了我从零开始跑通整个系统的过程，包括遇到的各种问题以及最终的解决方案。如果你也是 RAG 初学者，正在尝试用本地模型跑 LightRAG，希望这里的记录能帮你少走一些弯路。

---

## 这个项目是什么

这不是一个框架，也不是生产级工具。它是基于 LightRAG 已有 API 搭建的一个横向探索项目。知识图谱的构建和检索算法都来自 LightRAG，我没有自己实现这些核心部分。我做的事情是：

- 将 LightRAG 接入本地运行的 Ollama（LLM 用 qwen2.5:7b，Embedding 用 bge-m3）
- 加载中文医疗知识库，构建知识图谱
- 用 FastAPI 封装成 API 服务，配套写了一个简单的 HTML 网页客户端
- 花了大量时间调试超时和并发问题（见下文）

---

## 遇到的问题与解决方案

这一节是我保留这个仓库最主要的原因。用本地 7B 模型跑 LightRAG，在消费级硬件上远没有文档里描述的那么顺利。

### 问题一：插入知识图谱时频繁超时

这是最难缠的问题。LightRAG 在插入文档时，会调用 LLM 对每个文本块进行实体和关系提取。在本地 7B 模型上，这个过程比预期慢得多，默认的超时设置会导致整个插入过程中途失败。

**具体表现：**
- 处理几个文档后出现 `ReadTimeout` 或 `httpx.ReadTimeout` 错误
- 进程在处理 10~20 个 chunk 后无声无息地挂掉
- 存储目录里留下不完整的图谱文件

**解决方案：** 将 LLM 超时设置为 1200 秒，Embedding 超时设置为 1800 秒，并在每次超时错误后等待 30 秒再继续。

```python
llm_model_kwargs={
    "host": "http://localhost:11434",
    "options": {"num_ctx": 8192},
    "timeout": 1200,  # 20 分钟，是的，真的需要这么长
}
```

### 问题二：并发请求压垮本地模型

LightRAG 支持并发请求以加快插入速度。这在调用云端 API 时没有问题，但在本地 7B 模型上，多个请求同时占用 GPU 显存，会导致显存溢出或连锁超时。

**解决方案：** 用 `asyncio.Semaphore(1)` 强制所有插入请求串行执行。

```python
self._semaphore = asyncio.Semaphore(1)

async with self._semaphore:
    await self.rag.ainsert(doc_content)
    await asyncio.sleep(1.0)  # 每次插入后稍作等待
```

这样做插入速度变慢，但不会再出现崩溃。

### 问题三：长文档即使经过 LightRAG 内部分块后仍然超时

LightRAG 自身有分块机制，但对于一些很长的医疗文档（有些超过 5000 字），即使经过 LightRAG 的内部分块，单个 chunk 在实体提取阶段依然会超时。

**解决方案：** 在将文档传入 LightRAG 之前，先做一轮预处理分块，目标每块约 1000 字符，并尽量在句号处截断，保证语义完整性。

```python
if len(content) > 3000:
    for i in range(0, len(content), 1000):
        chunk = content[i:i + 1000]
        # 尝试在句号处截断，避免切断完整语句
        if '。' in chunk:
            last_period = chunk.rfind('。')
            if last_period > 700:
                chunk = chunk[:last_period + 1]
        await self.rag.ainsert(chunk)
        await asyncio.sleep(1.0)
```

也就是说，最终是在 LightRAG 自身分块机制之上，又加了一层预处理分块。这个双重分块策略让整个插入过程稳定下来。

---

## 系统结构

```
浏览器（HTML 网页客户端）
    |
    | HTTP
    v
FastAPI 服务端  (medical_rag_api_server.py)
    |
    | Python
    v
RAGManager + ModelManager  (src/core/)
    |
    | LightRAG API
    v
LightRAG  （知识图谱构建与检索，来自原始仓库）
    |
    | HTTP
    v
Ollama  （qwen2.5:7b 负责推理，bge-m3 负责向量化）
    |
    | 读写文件
    v
本地存储  (medical_json_small_storage/)
```

LightRAG 提供四种查询模式，均直接使用 LightRAG 的实现：

| 模式 | 说明 |
|------|------|
| `naive` | 纯向量相似度检索，速度最快 |
| `local` | 从最相关实体出发，在图谱中做局部遍历 |
| `global` | 利用图谱全局的社区摘要信息 |
| `hybrid` | 以上三种综合，效果最好，推荐使用 |

---

## 运行方法

### 环境要求

- Python 3.9+
- [Ollama](https://ollama.ai/) 本地运行中
- 建议 8GB 以上内存，6GB 以上显存的 GPU

### 安装

```bash
# 1. 克隆本项目
git clone https://github.com/gulugulu-sama/LightMed.git
cd LightMed

# 2. 克隆 LightRAG 并从源码安装（必须步骤，直接 pip 装的版本可能不兼容）
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
pip install -e .
cd ..

# 3. 安装其余依赖
pip install -r requirements.txt

# 4. 拉取本地模型
ollama pull qwen2.5:7b
ollama pull bge-m3:latest
```

### 第一步：构建知识图谱（只需执行一次）

将医疗数据的 `.json` 文件放入 `knowledgebase1/` 目录（格式参考 `medical_subset_200.json`），然后运行：

```bash
python medical_rag_json_optimized_small.py
```

脚本会将 JSON 中的每 3 条疾病记录合并为一个文本块再传入 LightRAG。这个过程需要较长时间。插入 200 条记录大约需要 1~2 小时。请耐心等待，中途不要中断。

<img width="809" height="569" alt="c4a5397b6e5fe1e107981f683f9477cd" src="https://github.com/user-attachments/assets/810c8e67-08bc-48f4-9ea3-de13e60774d0" />

### 第二步：启动 API 服务

```bash
python medical_rag_api_server.py
```

然后用浏览器直接打开 `medical_rag_web_client_optimized.html` 即可使用。

---

## 项目结构

```
├── src/core/
│   ├── rag_manager.py          # 封装 LightRAG 的初始化和查询
│   ├── model_manager.py        # 检查 Ollama 状态和模型是否存在
│   ├── kb_manager.py           # 知识库文件加载与预处理
│   └── query_optimizer.py      # 查询优化
├── configs/
│   ├── model_config.yaml       # LLM 和 Embedding 模型参数
│   └── rag_config.yaml         # LightRAG 各项参数和超时配置
├── medical_rag_api_server.py              # FastAPI 后端服务
├── medical_rag_json_optimized_small.py   # 知识图谱构建脚本（首次运行，读 JSON）
├── medical_rag_web_client_optimized.html  # 网页客户端
├── main.py                     # 命令行交互模式
└── requirements.txt
```

---

## 配置说明

`configs/model_config.yaml` 中的超时时间设置得非常高，这是有意为之。不要轻易降低这些值，除非你的硬件明显优于中端消费级 GPU。

```yaml
llm:
  model: "qwen2.5:7b"
  timeout: 1200         # 20 分钟，针对本地模型推理慢的问题

embedding:
  model: "bge-m3:latest"
  timeout: 1800         # 30 分钟
```

---

## 致谢

- [LightRAG](https://github.com/HKUDS/LightRAG) — 本项目所有核心能力的来源
- [Ollama](https://ollama.ai/) — 让本地运行大模型变得可行

---

## 医疗声明

本系统仅供个人学习和研究使用，不适用于任何临床诊断或医疗决策场景。如有健康问题，请咨询专业医生。
