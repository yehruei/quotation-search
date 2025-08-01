#!/usr/bin/env python3
"""
缓存管理器 - 支持embedding向量缓存
避免重复计算，提升检索性能
"""

import os
import hashlib
import pickle
import numpy as np
from datetime import datetime, timedelta


class CacheManager:
    """缓存管理器 - 支持embedding和其他数据缓存"""
    
    def __init__(self, cache_dir='cache', expire_days=7):
        self.cache_dir = cache_dir
        self.expire_days = expire_days
        
        # 确保缓存目录存在
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_path(self, key):
        """获取缓存文件路径"""
        # 使用MD5哈希作为文件名，避免特殊字符问题
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    def _is_cache_valid(self, cache_path):
        """检查缓存是否有效（未过期）"""
        if not os.path.exists(cache_path):
            return False
        
        # 检查文件修改时间
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        expire_time = datetime.now() - timedelta(days=self.expire_days)
        
        return file_time > expire_time
    
    def get(self, key):
        """获取缓存数据"""
        cache_path = self._get_cache_path(key)
        
        if not self._is_cache_valid(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data
        except Exception as e:
            print(f"缓存读取失败: {str(e)}")
            return None
    
    def set(self, key, data):
        """设置缓存数据"""
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"缓存写入失败: {str(e)}")
            return False
    
    def clear_expired(self):
        """清理过期缓存"""
        if not os.path.exists(self.cache_dir):
            return
        
        cleared_count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.cache'):
                cache_path = os.path.join(self.cache_dir, filename)
                if not self._is_cache_valid(cache_path):
                    try:
                        os.remove(cache_path)
                        cleared_count += 1
                    except Exception as e:
                        print(f"删除缓存文件失败 {filename}: {str(e)}")
        
        if cleared_count > 0:
            print(f"✓ 清理了 {cleared_count} 个过期缓存文件")
    
    def clear_all(self):
        """清理所有缓存"""
        if not os.path.exists(self.cache_dir):
            return
        
        cleared_count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.cache'):
                cache_path = os.path.join(self.cache_dir, filename)
                try:
                    os.remove(cache_path)
                    cleared_count += 1
                except Exception as e:
                    print(f"删除缓存文件失败 {filename}: {str(e)}")
        
        print(f"✓ 清理了 {cleared_count} 个缓存文件")
    
    def get_cache_info(self):
        """获取缓存统计信息"""
        if not os.path.exists(self.cache_dir):
            return {'count': 0, 'size': 0}
        
        count = 0
        total_size = 0
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.cache'):
                cache_path = os.path.join(self.cache_dir, filename)
                if os.path.exists(cache_path):
                    count += 1
                    total_size += os.path.getsize(cache_path)
        
        return {
            'count': count,
            'size': total_size,
            'size_mb': total_size / (1024 * 1024)
        }


class EmbeddingCache(CacheManager):
    """专门用于embedding向量缓存的类"""
    
    def __init__(self, cache_dir='cache/embeddings', expire_days=30):
        super().__init__(cache_dir, expire_days)
        self.model_name = None
    
    def set_model(self, model_name):
        """设置当前使用的模型名称"""
        self.model_name = model_name
    
    def _get_embedding_key(self, text_list, model_name=None):
        """生成embedding缓存键"""
        if model_name is None:
            model_name = self.model_name or "default"
        
        # 组合文本内容和模型名称作为键
        combined_text = "\n".join(text_list) if isinstance(text_list, list) else str(text_list)
        cache_key = f"{model_name}::{combined_text}"
        return cache_key
    
    def get_embeddings(self, text_list, model_name=None):
        """获取文本列表的embedding缓存"""
        cache_key = self._get_embedding_key(text_list, model_name)
        cached_embeddings = self.get(cache_key)
        
        if cached_embeddings is not None:
            print(f"✓ 从缓存加载了 {len(text_list)} 个文本的embeddings")
            return cached_embeddings
        
        return None
    
    def set_embeddings(self, text_list, embeddings, model_name=None):
        """缓存文本列表的embeddings"""
        cache_key = self._get_embedding_key(text_list, model_name)
        
        # 确保embeddings是numpy数组
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        success = self.set(cache_key, embeddings)
        if success:
            print(f"✓ 缓存了 {len(text_list)} 个文本的embeddings")
        
        return success
    
    def get_single_embedding(self, text, model_name=None):
        """获取单个文本的embedding缓存"""
        return self.get_embeddings([text], model_name)
    
    def set_single_embedding(self, text, embedding, model_name=None):
        """缓存单个文本的embedding"""
        return self.set_embeddings([text], [embedding], model_name)


# 全局缓存实例
embedding_cache = EmbeddingCache()
general_cache = CacheManager()
