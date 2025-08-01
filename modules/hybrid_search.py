#!/usr/bin/env python3
"""
Enhanced Hybrid Search Module - Integrates BM25 with EmbeddingRetriever
Uses Reciprocal Rank Fusion (RRF) for optimal result combination
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import time
import concurrent.futures
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Represents a search result with unified scoring"""
    chunk_id: str
    content: str
    page_num: int
    word_count: int
    index: int
    
    # Original scores
    bm25_score: float = 0.0
    embedding_score: float = 0.0
    
    # Ranking information
    bm25_rank: int = float('inf')
    embedding_rank: int = float('inf')
    
    # Fusion scores
    rrf_score: float = 0.0
    weighted_score: float = 0.0
    
    # Additional metadata
    found_keywords: List[str] = None
    literary_analysis: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.found_keywords is None:
            self.found_keywords = []
        if self.literary_analysis is None:
            self.literary_analysis = {}

class EnhancedRRF:
    """Enhanced Reciprocal Rank Fusion implementation"""
    
    def __init__(self, k: int = 60, weights: Optional[Dict[str, float]] = None):
        """
        Initialize RRF with configurable parameters
        
        Args:
            k: RRF parameter (typically 60)
            weights: Optional weights for different retrieval methods
        """
        self.k = k
        self.weights = weights or {'bm25': 0.5, 'embedding': 0.5}
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
    
    def fuse_results(self, bm25_results: List[SearchResult], 
                    embedding_results: List[SearchResult]) -> List[SearchResult]:
        """
        Fuse results from BM25 and embedding retrieval using RRF
        
        Args:
            bm25_results: Results from BM25 retrieval
            embedding_results: Results from embedding retrieval
            
        Returns:
            Fused results ranked by RRF score
        """
        # Create mappings for efficient lookup
        bm25_map = {r.index: r for r in bm25_results}
        embedding_map = {r.index: r for r in embedding_results}
        
        # Create rank mappings
        bm25_ranks = {r.index: i + 1 for i, r in enumerate(bm25_results)}
        embedding_ranks = {r.index: i + 1 for i, r in enumerate(embedding_results)}
        
        # Get all unique indices
        all_indices = set(bm25_map.keys()) | set(embedding_map.keys())
        
        fused_results = []
        
        for idx in all_indices:
            # Get the result (prioritize embedding result if available)
            if idx in embedding_map:
                result = embedding_map[idx]
                result.bm25_score = bm25_map.get(idx, SearchResult("", "", 0, 0, idx)).bm25_score
            else:
                result = bm25_map[idx]
                result.embedding_score = 0.0
            
            # Calculate RRF score
            rrf_score = 0.0
            
            if idx in bm25_ranks:
                rrf_score += self.weights['bm25'] / (self.k + bm25_ranks[idx])
                result.bm25_rank = bm25_ranks[idx]
            
            if idx in embedding_ranks:
                rrf_score += self.weights['embedding'] / (self.k + embedding_ranks[idx])
                result.embedding_rank = embedding_ranks[idx]
            
            result.rrf_score = rrf_score
            fused_results.append(result)
        
        # Sort by RRF score
        fused_results.sort(key=lambda x: x.rrf_score, reverse=True)
        
        return fused_results
    
    def fuse_with_score_normalization(self, bm25_results: List[SearchResult], 
                                    embedding_results: List[SearchResult]) -> List[SearchResult]:
        """
        Alternative fusion method using score normalization
        
        Args:
            bm25_results: Results from BM25 retrieval
            embedding_results: Results from embedding retrieval
            
        Returns:
            Fused results ranked by normalized weighted score
        """
        # Normalize BM25 scores
        if bm25_results:
            bm25_scores = [r.bm25_score for r in bm25_results]
            bm25_max = max(bm25_scores)
            bm25_min = min(bm25_scores)
            if bm25_max > bm25_min:
                for result in bm25_results:
                    result.bm25_score = (result.bm25_score - bm25_min) / (bm25_max - bm25_min)
        
        # Normalize embedding scores
        if embedding_results:
            embedding_scores = [r.embedding_score for r in embedding_results]
            embedding_max = max(embedding_scores)
            embedding_min = min(embedding_scores)
            if embedding_max > embedding_min:
                for result in embedding_results:
                    result.embedding_score = (result.embedding_score - embedding_min) / (embedding_max - embedding_min)
        
        # Create mappings
        bm25_map = {r.index: r for r in bm25_results}
        embedding_map = {r.index: r for r in embedding_results}
        
        # Get all unique indices
        all_indices = set(bm25_map.keys()) | set(embedding_map.keys())
        
        fused_results = []
        
        for idx in all_indices:
            # Get the result (prioritize embedding result if available)
            if idx in embedding_map:
                result = embedding_map[idx]
                bm25_score = bm25_map.get(idx, SearchResult("", "", 0, 0, idx)).bm25_score
            else:
                result = bm25_map[idx]
                result.embedding_score = 0.0
                bm25_score = result.bm25_score
            
            # Calculate weighted score
            weighted_score = (self.weights['bm25'] * bm25_score + 
                            self.weights['embedding'] * result.embedding_score)
            
            result.weighted_score = weighted_score
            fused_results.append(result)
        
        # Sort by weighted score
        fused_results.sort(key=lambda x: x.weighted_score, reverse=True)
        
        return fused_results

class HybridSearchEngine:
    """Enhanced Hybrid Search Engine that integrates BM25 and Embedding retrieval"""
    
    def __init__(self, embedding_retriever, bm25_retriever, literary_analyzer=None,
                 fusion_method: str = 'rrf', rrf_k: int = 60, 
                 weights: Optional[Dict[str, float]] = None,
                 enable_parallel: bool = True):
        """
        Initialize the hybrid search engine
        
        Args:
            embedding_retriever: EmbeddingRetriever instance
            bm25_retriever: BM25Retriever instance  
            literary_analyzer: Optional literary analyzer
            fusion_method: 'rrf' or 'weighted' fusion method
            rrf_k: RRF parameter
            weights: Weights for different retrieval methods
            enable_parallel: Whether to enable parallel processing
        """
        self.embedding_retriever = embedding_retriever
        self.bm25_retriever = bm25_retriever
        self.literary_analyzer = literary_analyzer
        self.fusion_method = fusion_method
        self.enable_parallel = enable_parallel
        
        # Initialize RRF fusion
        self.rrf = EnhancedRRF(k=rrf_k, weights=weights)
        
        # Performance tracking
        self.stats = {
            'total_queries': 0,
            'avg_bm25_time': 0.0,
            'avg_embedding_time': 0.0,
            'avg_fusion_time': 0.0
        }
    
    def search(self, text_chunks: List[Dict[str, Any]], 
              query_keywords: List[str], 
              k: int = 100,
              min_results: int = 3) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and embedding retrieval
        
        Args:
            text_chunks: List of text chunks to search
            query_keywords: Query keywords
            k: Number of results to return
            
        Returns:
            List of search results ranked by hybrid score
        """
        start_time = time.time()
        
        print(f"🔍 启动混合搜索 - 查询关键词: {query_keywords}")
        print(f"📊 检索模式: {self.fusion_method.upper()}, 并行处理: {self.enable_parallel}")
        
        # Update retriever keywords
        self.embedding_retriever.keywords = query_keywords
        self.bm25_retriever.keywords = query_keywords
        
        if self.enable_parallel:
            # Parallel retrieval
            bm25_results, embedding_results = self._parallel_retrieve(text_chunks, k)
        else:
            # Sequential retrieval
            bm25_results, embedding_results = self._sequential_retrieve(text_chunks, k)
        
        # Fusion
        fusion_start = time.time()
        fused_results = self._fuse_results(bm25_results, embedding_results)
        fusion_time = time.time() - fusion_start
        
        # 保证最少返回结果机制
        if len(fused_results) < min_results and len(text_chunks) >= min_results:
            print(f"🔧 结果不足，启动最少返回机制... 当前: {len(fused_results)}, 需要: {min_results}")
            
            # 获取已有结果的索引
            used_indices = {result.index for result in fused_results}
            available_indices = [i for i in range(len(text_chunks)) if i not in used_indices]
            
            # 添加补充结果
            needed = min_results - len(fused_results)
            if len(available_indices) >= needed:
                import random
                selected_indices = random.sample(available_indices, needed)
            else:
                selected_indices = available_indices
            
            for idx in selected_indices:
                chunk = text_chunks[idx]
                fallback_result = SearchResult(
                    chunk_id=chunk.get('id', f'chunk_{idx}'),
                    content=chunk.get('content', ''),
                    page_num=chunk.get('page_num', idx + 1),
                    word_count=len(chunk.get('content', '').split()),
                    index=idx,
                    bm25_score=0.01,
                    embedding_score=0.01,
                    rrf_score=0.01,
                    weighted_score=0.01,
                    found_keywords=query_keywords
                )
                fused_results.append(fallback_result)
            
            print(f"📝 已补充 {len(selected_indices)} 个结果，总计 {len(fused_results)} 个")
        
        # Convert to final format
        final_results = self._convert_to_final_format(fused_results[:k])
        
        # Update statistics
        total_time = time.time() - start_time
        self.stats['total_queries'] += 1
        self.stats['avg_fusion_time'] = (self.stats['avg_fusion_time'] * (self.stats['total_queries'] - 1) + fusion_time) / self.stats['total_queries']
        
        print(f"✅ 混合搜索完成 - 耗时: {total_time:.2f}s, 返回 {len(final_results)} 个结果")
        print(f"📈 融合时间: {fusion_time:.3f}s, 融合方法: {self.fusion_method}")
        
        return final_results
    
    def _parallel_retrieve(self, text_chunks: List[Dict[str, Any]], 
                          k: int) -> Tuple[List[SearchResult], List[SearchResult]]:
        """Perform parallel retrieval using both BM25 and embedding methods"""
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both retrieval tasks
            bm25_future = executor.submit(self._retrieve_bm25, text_chunks, k)
            embedding_future = executor.submit(self._retrieve_embedding, text_chunks, k)
            
            # Get results
            bm25_results = bm25_future.result()
            embedding_results = embedding_future.result()
        
        return bm25_results, embedding_results
    
    def _sequential_retrieve(self, text_chunks: List[Dict[str, Any]], 
                           k: int) -> Tuple[List[SearchResult], List[SearchResult]]:
        """Perform sequential retrieval using both BM25 and embedding methods"""
        
        bm25_results = self._retrieve_bm25(text_chunks, k)
        embedding_results = self._retrieve_embedding(text_chunks, k)
        
        return bm25_results, embedding_results
    
    def _retrieve_bm25(self, text_chunks: List[Dict[str, Any]], k: int) -> List[SearchResult]:
        """Retrieve results using BM25"""
        start_time = time.time()
        
        try:
            raw_results = self.bm25_retriever.retrieve(text_chunks, k)
            
            # Convert to SearchResult objects
            results = []
            for result in raw_results:
                search_result = SearchResult(
                    chunk_id=result['chunk_id'],
                    content=result['content'],
                    page_num=result.get('page_num', 1),
                    word_count=result.get('word_count', 0),
                    index=result['index'],
                    bm25_score=result.get('bm25_score', 0.0),
                    found_keywords=result.get('found_keywords', []),
                    literary_analysis=result.get('literary_analysis', {})
                )
                results.append(search_result)
            
            # Update timing statistics
            elapsed_time = time.time() - start_time
            self.stats['avg_bm25_time'] = (self.stats['avg_bm25_time'] * (self.stats['total_queries']) + elapsed_time) / max(1, self.stats['total_queries'] + 1)
            
            print(f"🔤 BM25检索完成 - 耗时: {elapsed_time:.3f}s, 召回: {len(results)} 个结果")
            return results
            
        except Exception as e:
            print(f"❌ BM25检索失败: {e}")
            return []
    
    def _retrieve_embedding(self, text_chunks: List[Dict[str, Any]], k: int) -> List[SearchResult]:
        """Retrieve results using embedding similarity"""
        start_time = time.time()
        
        try:
            raw_results = self.embedding_retriever.retrieve(text_chunks, k)
            
            # Convert to SearchResult objects
            results = []
            for result in raw_results:
                search_result = SearchResult(
                    chunk_id=result['chunk_id'],
                    content=result['content'],
                    page_num=result.get('page_num', 1),
                    word_count=result.get('word_count', 0),
                    index=result['index'],
                    embedding_score=result.get('similarity_score', 0.0),
                    found_keywords=result.get('found_keywords', []),
                    literary_analysis=result.get('literary_analysis', {})
                )
                results.append(search_result)
            
            # Update timing statistics
            elapsed_time = time.time() - start_time
            self.stats['avg_embedding_time'] = (self.stats['avg_embedding_time'] * (self.stats['total_queries']) + elapsed_time) / max(1, self.stats['total_queries'] + 1)
            
            print(f"🧠 Embedding检索完成 - 耗时: {elapsed_time:.3f}s, 召回: {len(results)} 个结果")
            return results
            
        except Exception as e:
            print(f"❌ Embedding检索失败: {e}")
            return []
    
    def _fuse_results(self, bm25_results: List[SearchResult], 
                     embedding_results: List[SearchResult]) -> List[SearchResult]:
        """Fuse results from both retrieval methods"""
        
        if self.fusion_method == 'rrf':
            return self.rrf.fuse_results(bm25_results, embedding_results)
        elif self.fusion_method == 'weighted':
            return self.rrf.fuse_with_score_normalization(bm25_results, embedding_results)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _convert_to_final_format(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Convert SearchResult objects to final dictionary format"""
        
        final_results = []
        for result in results:
            # Calculate the main score based on fusion method
            main_score = result.rrf_score if self.fusion_method == 'rrf' else result.weighted_score
            
            # 检查是否是补充结果（低分数）
            is_fallback = (result.bm25_score <= 0.01 and result.embedding_score <= 0.01 and 
                          result.rrf_score <= 0.01 and result.weighted_score <= 0.01)
            
            final_result = {
                'chunk_id': result.chunk_id,
                'content': result.content,
                'text': result.content,  # For compatibility
                'page_num': result.page_num,
                'word_count': result.word_count,
                'index': result.index,
                'found_keywords': result.found_keywords,
                'literary_analysis': result.literary_analysis,
                'is_fallback': is_fallback,  # 标记补充结果
                
                # Main score field (for compatibility with filtering)
                'score': main_score,
                
                # Detailed scores
                'hybrid_score': main_score,
                'bm25_score': result.bm25_score,
                'similarity_score': result.embedding_score,
                
                # Ranking information
                'bm25_rank': result.bm25_rank,
                'embedding_rank': result.embedding_rank,
                'fusion_method': self.fusion_method
            }
            
            # Add specific scores based on fusion method
            if self.fusion_method == 'rrf':
                final_result['rrf_score'] = result.rrf_score
            else:
                final_result['weighted_score'] = result.weighted_score
            
            final_results.append(final_result)
        
        return final_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'total_queries': self.stats['total_queries'],
            'avg_bm25_time': self.stats['avg_bm25_time'],
            'avg_embedding_time': self.stats['avg_embedding_time'],
            'avg_fusion_time': self.stats['avg_fusion_time'],
            'fusion_method': self.fusion_method,
            'parallel_enabled': self.enable_parallel
        }
    
    def update_fusion_weights(self, weights: Dict[str, float]):
        """Update fusion weights dynamically"""
        self.rrf.weights = weights
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.rrf.weights = {k: v / total_weight for k, v in weights.items()}
        print(f"🔄 融合权重已更新: {self.rrf.weights}")

# Factory function for creating hybrid search engine
def create_hybrid_search_engine(embedding_retriever, bm25_retriever, 
                               literary_analyzer=None, 
                               fusion_method: str = 'rrf',
                               rrf_k: int = 60,
                               weights: Optional[Dict[str, float]] = None,
                               enable_parallel: bool = True) -> HybridSearchEngine:
    """
    Factory function to create a hybrid search engine
    
    Args:
        embedding_retriever: EmbeddingRetriever instance
        bm25_retriever: BM25Retriever instance
        literary_analyzer: Optional literary analyzer
        fusion_method: 'rrf' or 'weighted' fusion method
        rrf_k: RRF parameter
        weights: Weights for different retrieval methods
        enable_parallel: Whether to enable parallel processing
    
    Returns:
        HybridSearchEngine instance
    """
    return HybridSearchEngine(
        embedding_retriever=embedding_retriever,
        bm25_retriever=bm25_retriever,
        literary_analyzer=literary_analyzer,
        fusion_method=fusion_method,
        rrf_k=rrf_k,
        weights=weights,
        enable_parallel=enable_parallel
    )