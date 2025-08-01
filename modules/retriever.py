from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import warnings
from rank_bm25 import BM25Okapi

def load_sentence_transformer_model(model_path):
    """
    安全加载SentenceTransformer模型，优先使用本地缓存
    """
    # 检查model_path是否为None或空
    if model_path is None or model_path == "":
        print("⚠️ 模型路径为空，使用默认模型")
        model_path = 'sentence-transformers/all-MiniLM-L6-v2'
    
    # 如果传入的是模型名称而不是路径，直接尝试加载
    if model_path.startswith('sentence-transformers/') or model_path.startswith('all-'):
        try:
            print(f"🔍 使用模型名称加载: {model_path}")
            model = SentenceTransformer(model_path, device='cpu', trust_remote_code=False)
            print(f"✅ 模型加载成功: {model_path}")
            return model
        except Exception as e:
            print(f"⚠️ 模型 {model_path} 加载失败: {e}")
            return None
    
    # 如果是本地路径，先检查是否存在
    import os
    if os.path.exists(model_path):
        try:
            print(f"📚 正在加载本地模型: {model_path}")
            model = SentenceTransformer(model_path)
            print(f"✅ 本地模型加载成功")
            return model
        except Exception as e:
            print(f"⚠️ 本地模型 {model_path} 加载失败: {e}")
    
    # 如果本地路径不存在或加载失败，尝试使用模型名称
    model_names = [
        'sentence-transformers/all-MiniLM-L6-v2',
        'all-MiniLM-L6-v2'
    ]
    
    for model_name in model_names:
        try:
            print(f"🔍 尝试通过模型名称加载: {model_name}")
            model = SentenceTransformer(model_name, device='cpu', trust_remote_code=False)
            print(f"✅ 模型加载成功: {model_name}")
            return model
        except Exception as e:
            print(f"⚠️ 模型 {model_name} 加载失败: {e}")
            
            # 尝试清理缓存后重新加载
            try:
                import shutil
                cache_dirs = [
                    os.path.expanduser("~/.cache/torch/sentence_transformers"),
                    os.path.expanduser("~/.cache/huggingface/transformers"),
                    os.path.expanduser("~/.cache/huggingface/hub")
                ]
                
                for cache_dir in cache_dirs:
                    if os.path.exists(cache_dir):
                        print(f"🧹 清理缓存目录: {cache_dir}")
                        try:
                            shutil.rmtree(cache_dir)
                        except:
                            pass
                
                print(f"🔄 清理缓存后重试加载: {model_name}")
                model = SentenceTransformer(model_name, device='cpu', trust_remote_code=False)
                print(f"✅ 清理缓存后模型加载成功: {model_name}")
                return model
                
            except Exception as e2:
                print(f"⚠️ 清理缓存后仍然失败: {e2}")
                continue
    
    print("🌐 本地缓存不可用，尝试在线下载...")
    
    # 备用方案：尝试下载在线模型（带SSL处理）
    try:
        # 禁用SSL验证作为临时解决方案
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        print(f"🌐 尝试在线下载: {model_name}")
        model = SentenceTransformer(model_name)
        print(f"✅ 在线模型下载和加载成功: {model_name}")
        return model
        
    except Exception as e:
        print(f"⚠️ 在线下载失败: {e}")
    
    # 最后的备用方案：使用简单词汇模型
    print("🚨 所有模型加载失败，将使用基于词汇的简单模型")
    return None

class SimpleVocabModel:
    """
    简单的基于词汇的语义模型，作为SentenceTransformer的备用方案
    """
    def __init__(self):
        self.vocab = {}
        
    def encode(self, texts):
        """简单的词汇编码"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            words = text.lower().split()
            # 创建简单的词频向量
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # 转换为固定长度的向量
            # 使用常见的1000个词汇维度
            vocab_size = 1000
            embedding = np.zeros(vocab_size)
            
            for word, count in word_counts.items():
                # 简单的哈希函数映射词汇到维度
                dim = hash(word) % vocab_size
                embedding[dim] = count
                
            # 归一化
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
                
            embeddings.append(embedding)
        
        return np.array(embeddings)

class EmbeddingRetriever:
    def __init__(self, model_path):
        self.model = load_sentence_transformer_model(model_path)
        self.use_simple_model = self.model is None
        if self.use_simple_model:
            self.model = SimpleVocabModel()
            print("🔄 使用简单词汇模型进行语义搜索")
        self.keywords = []

    def retrieve(self, text_chunks, k):
        query = " ".join(self.keywords)
        
        try:
            query_embedding = self.model.encode([query])
            
            texts = [chunk['content'] for chunk in text_chunks]
            text_embeddings = self.model.encode(texts)
            
            similarities = cosine_similarity(query_embedding, text_embeddings)[0]
            
            results = []
            for i, score in enumerate(similarities):
                results.append({
                    'chunk_id': text_chunks[i].get('id', f'chunk_{i}'),
                    'content': text_chunks[i]['content'],
                    'page_num': text_chunks[i].get('page_num', 1),
                    'word_count': len(text_chunks[i]['content'].split()),
                    'index': i,
                    'similarity_score': score,
                    'found_keywords': self.keywords,
                    'model_type': 'simple_vocab' if self.use_simple_model else 'sentence_transformer'
                })
                
            return sorted(results, key=lambda x: x['similarity_score'], reverse=True)[:k]
            
        except Exception as e:
            print(f"⚠️ 语义检索失败: {e}")
            # 降级到BM25搜索
            print("🔄 降级到BM25搜索")
            bm25_retriever = BM25Retriever()
            bm25_retriever.keywords = self.keywords
            return bm25_retriever.retrieve(text_chunks, k)

class BM25Retriever:
    def __init__(self):
        self.keywords = []
        self.bm25 = None

    def retrieve(self, text_chunks, k):
        tokenized_corpus = [chunk['content'].split(" ") for chunk in text_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        query = " ".join(self.keywords)
        tokenized_query = query.split(" ")
        
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        top_n_indices = np.argsort(doc_scores)[::-1][:k]
        
        results = []
        for i in top_n_indices:
            results.append({
                'chunk_id': text_chunks[i].get('id', f'chunk_{i}'),
                'content': text_chunks[i]['content'],
                'page_num': text_chunks[i].get('page_num', 1),
                'word_count': len(text_chunks[i]['content'].split()),
                'index': i,
                'bm25_score': doc_scores[i],
                'found_keywords': self.keywords
            })
            
        return results

def search(query, texts, model_path):
    """Performs a semantic search for a query within a list of texts."""
    model = load_sentence_transformer_model(model_path)
    
    if model is None:
        print("🔄 使用简单词汇模型进行搜索")
        model = SimpleVocabModel()
    
    try:
        query_embedding = model.encode([query])
        text_embeddings = model.encode(texts)

        similarities = cosine_similarity(query_embedding, text_embeddings)[0]

        results = []
        for i, score in enumerate(similarities):
            results.append({
                'score': score, 
                'text': texts[i],
                'page_num': i + 1,  # 简单的页码分配
                'index': i
            })

        return sorted(results, key=lambda x: x['score'], reverse=True)
        
    except Exception as e:
        print(f"⚠️ 语义搜索失败: {e}")
        # 降级到简单的文本匹配
        print("🔄 降级到简单文本匹配")
        results = []
        query_words = query.lower().split()
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            score = 0
            for word in query_words:
                score += text_lower.count(word)
            
            # 归一化分数
            max_possible_score = len(query_words) * max(1, len(text.split()) // 10)
            normalized_score = score / max_possible_score if max_possible_score > 0 else 0
            
            results.append({
                'score': normalized_score,
                'text': text,
                'page_num': i + 1,
                'index': i
            })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
