"""
知识库管理器 - 基于实践测试优化的最终版本
集成预分块策略、文档预处理和增量更新
"""

import os
import glob
from typing import List, Dict, Any
from pathlib import Path
import asyncio
import hashlib

class KnowledgeBaseManager:
    def __init__(self, knowledge_base_path: str = "knowledge_bases"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.file_cache = {}
        self.last_modified = {}
        self.file_hashes = {}  # 文件哈希缓存，用于检测内容变更
        
    def get_all_files(self) -> List[str]:
        """获取所有知识库文件"""
        files = []
        if not self.knowledge_base_path.exists():
            print(f"⚠️ 知识库目录不存在: {self.knowledge_base_path}")
            return files
        
        # 支持的文件类型
        extensions = ['*.txt', '*.md', '*.pdf', '*.docx']
        
        for ext in extensions:
            pattern = str(self.knowledge_base_path / "**" / ext)
            found_files = glob.glob(pattern, recursive=True)
            files.extend(found_files)
        
        return files
    
    def _preprocess_text(self, content: str) -> str:
        """预处理文本 - 清理和格式化"""
        # 移除多余的空行
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # 非空行
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1]:  # 保留段落分隔
                cleaned_lines.append('')
        
        return '\n'.join(cleaned_lines)
    
    def _chunk_document(self, content: str, filename: str, category: str) -> List[str]:
        """文档分块策略 - 基于实践测试优化"""
        # 预处理内容
        content = self._preprocess_text(content)
        
        # 分块参数 - 基于测试优化
        max_chunk_size = 800  # 字符数，适中的chunk大小
        overlap_size = 100    # 重叠大小
        
        chunks = []
        
        # 按段落分割
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            # 如果当前段落加上现有chunk不超过限制，则合并
            if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # 保存当前chunk（如果不为空）
                if current_chunk:
                    chunk_header = f"文档: {filename}\n分类: {category}\n\n"
                    chunks.append(chunk_header + current_chunk)
                
                # 开始新chunk
                if len(paragraph) > max_chunk_size:
                    # 段落太长，需要强制分割
                    words = paragraph.split(' ')
                    temp_chunk = ""
                    
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 <= max_chunk_size:
                            if temp_chunk:
                                temp_chunk += " " + word
                            else:
                                temp_chunk = word
                        else:
                            if temp_chunk:
                                chunk_header = f"文档: {filename}\n分类: {category}\n\n"
                                chunks.append(chunk_header + temp_chunk)
                            temp_chunk = word
                    
                    current_chunk = temp_chunk
                else:
                    current_chunk = paragraph
        
        # 保存最后一个chunk
        if current_chunk:
            chunk_header = f"文档: {filename}\n分类: {category}\n\n"
            chunks.append(chunk_header + current_chunk)
        
        return chunks
    
    def read_text_file(self, file_path: str) -> str:
        """读取文本文件 - 改进编码处理"""
        try:
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print(f"⚠️ 无法读取文件 {file_path}：编码问题")
                return ""
            
            return content
            
        except Exception as e:
            print(f"⚠️ 读取文件失败 {file_path}: {e}")
            return ""
    
    def _get_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except:
            return ""
    
    async def load_all_documents(self) -> List[str]:
        """异步加载所有文档 - 集成预分块策略"""
        files = self.get_all_files()
        all_chunks = []
        
        print(f"📚 发现 {len(files)} 个知识库文件")
        
        if not files:
            print("⚠️ 未找到任何知识库文件")
            return []
        
        for file_path in files:
            try:
                filename = os.path.basename(file_path)
                category = os.path.basename(os.path.dirname(file_path))
                
                print(f"📄 处理文件: {filename}")
                
                # 检查文件是否已缓存且未变更
                file_hash = self._get_file_hash(file_path)
                if file_path in self.file_hashes and self.file_hashes[file_path] == file_hash:
                    if file_path in self.file_cache:
                        print(f"   📋 使用缓存 (共{len(self.file_cache[file_path])}块)")
                        all_chunks.extend(self.file_cache[file_path])
                        continue
                
                # 读取和处理文件
                content = self.read_text_file(file_path)
                if not content:
                    continue
                
                # 分块处理
                chunks = self._chunk_document(content, filename, category)
                print(f"   ✂️ 分割为 {len(chunks)} 个块")
                
                # 缓存结果
                self.file_cache[file_path] = chunks
                self.file_hashes[file_path] = file_hash
                
                all_chunks.extend(chunks)
                
                # 添加处理延迟，避免过快处理
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"⚠️ 处理文件失败 {file_path}: {e}")
                continue
        
        print(f"✅ 文档处理完成，共生成 {len(all_chunks)} 个文档块")
        return all_chunks
    
    def check_for_updates(self) -> bool:
        """检查文件是否有更新 - 改进的变更检测"""
        current_files = self.get_all_files()
        has_updates = False
        
        # 检查新文件或修改的文件
        for file_path in current_files:
            try:
                current_mtime = os.path.getmtime(file_path)
                current_hash = self._get_file_hash(file_path)
                
                # 检查修改时间或文件哈希
                if (file_path not in self.last_modified or 
                    current_mtime > self.last_modified[file_path] or
                    file_path not in self.file_hashes or
                    current_hash != self.file_hashes[file_path]):
                    
                    has_updates = True
                    self.last_modified[file_path] = current_mtime
                    self.file_hashes[file_path] = current_hash
                    
                    # 清除该文件的缓存
                    if file_path in self.file_cache:
                        del self.file_cache[file_path]
                    
                    print(f"🔄 检测到文件变更: {os.path.basename(file_path)}")
                    
            except Exception as e:
                print(f"⚠️ 检查文件修改时间失败 {file_path}: {e}")
        
        # 检查删除的文件
        cached_files = set(self.file_cache.keys())
        current_files_set = set(current_files)
        deleted_files = cached_files - current_files_set
        
        if deleted_files:
            has_updates = True
            for deleted_file in deleted_files:
                print(f"🗑️ 检测到文件删除: {os.path.basename(deleted_file)}")
                if deleted_file in self.file_cache:
                    del self.file_cache[deleted_file]
                if deleted_file in self.file_hashes:
                    del self.file_hashes[deleted_file]
                if deleted_file in self.last_modified:
                    del self.last_modified[deleted_file]
        
        return has_updates
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        files = self.get_all_files()
        categories = set()
        total_chunks = 0
        
        for file_path in files:
            category = os.path.basename(os.path.dirname(file_path))
            categories.add(category)
            
            if file_path in self.file_cache:
                total_chunks += len(self.file_cache[file_path])
        
        return {
            'total_files': len(files),
            'total_chunks': total_chunks,
            'categories': list(categories),
            'base_path': str(self.knowledge_base_path),
            'cached_files': len(self.file_cache)
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.file_cache.clear()
        self.file_hashes.clear()
        self.last_modified.clear()
        print("🧹 知识库缓存已清空")
    
    def get_category_files(self, category: str) -> List[str]:
        """获取特定分类的文件"""
        all_files = self.get_all_files()
        category_files = []
        
        for file_path in all_files:
            file_category = os.path.basename(os.path.dirname(file_path))
            if file_category == category:
                category_files.append(file_path)
        
        return category_files 