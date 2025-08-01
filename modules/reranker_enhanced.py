#!/usr/bin/env python3
"""
增强版重排器模块 - 在原有基础上添加优化功能
新增功能：
1. 多模型ensemble重排
2. 学习式重排器
3. 语义一致性检查
4. 结果多样性优化
5. 上下文感知重排
"""

import numpy as np
import os
from sentence_transformers import CrossEncoder
import re
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# 导入原有重排器
from .reranker import CrossEncoderReranker, SimpleReranker


class MultiModelEnsembleReranker:
    """多模型ensemble重排器"""
    
    def __init__(self, model_names=None, keywords=None, weights=None):
        self.keywords = keywords
        self.models = {}
        self.weights = weights or {}
        
        # 默认模型配置
        if model_names is None:
            model_names = [
                'cross-encoder/ms-marco-MiniLM-L-6-v2',
                'cross-encoder/ms-marco-TinyBERT-L-2-v2'
            ]
        
        # 加载多个模型
        for model_name in model_names:
            try:
                print(f"正在加载重排模型: {model_name}")
                # 添加设备参数和更安全的加载方式
                model = CrossEncoder(model_name, device='cpu', trust_remote_code=False)
                self.models[model_name] = model
                if model_name not in self.weights:
                    self.weights[model_name] = 1.0
                print(f"✅ 模型 {model_name} 加载成功")
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
                            try:
                                shutil.rmtree(cache_dir)
                            except:
                                pass
                    
                    print(f"🔄 清理缓存后重试: {model_name}")
                    model = CrossEncoder(model_name, device='cpu', trust_remote_code=False)
                    self.models[model_name] = model
                    if model_name not in self.weights:
                        self.weights[model_name] = 1.0
                    print(f"✅ 清理缓存后模型 {model_name} 加载成功")
                    
                except Exception as e2:
                    print(f"⚠️ 清理缓存后仍然失败: {e2}")
                    continue
        
        if not self.models:
            print("❌ 没有可用的重排模型")
    
    def rerank(self, candidates, k=None):
        """使用多模型ensemble进行重排"""
        if not self.models or not candidates:
            return candidates
        
        print(f"🔄 使用 {len(self.models)} 个模型进行ensemble重排...")
        
        try:
            query = ' '.join(self.keywords) if self.keywords else "relevant content"
            query_text_pairs = [(query, item['content']) for item in candidates]
            
            # 收集所有模型的预测
            all_scores = {}
            for model_name, model in self.models.items():
                print(f"  使用模型 {model_name} 预测...")
                scores = model.predict(query_text_pairs, show_progress_bar=False)
                all_scores[model_name] = scores
            
            # 计算ensemble分数
            ensemble_scores = []
            for i in range(len(candidates)):
                weighted_score = 0.0
                total_weight = 0.0
                
                for model_name, scores in all_scores.items():
                    weight = self.weights.get(model_name, 1.0)
                    weighted_score += scores[i] * weight
                    total_weight += weight
                
                ensemble_score = weighted_score / total_weight if total_weight > 0 else 0.0
                ensemble_scores.append(ensemble_score)
            
            # 分数后处理
            ensemble_scores = np.array(ensemble_scores)
            
            # 使用更温和的归一化
            p10 = np.percentile(ensemble_scores, 10)
            p90 = np.percentile(ensemble_scores, 90)
            
            reranked_results = []
            for i, item in enumerate(candidates):
                raw_score = float(ensemble_scores[i])
                
                # 归一化到 [0.1, 0.9] 范围
                if p90 > p10:
                    normalized_score = 0.1 + 0.8 * (raw_score - p10) / (p90 - p10)
                    normalized_score = max(0.1, min(0.9, normalized_score))
                else:
                    normalized_score = 0.5
                
                new_item = item.copy()
                new_item['ensemble_rerank_score'] = normalized_score
                new_item['raw_ensemble_score'] = raw_score
                reranked_results.append(new_item)
            
            # 排序
            reranked_results.sort(key=lambda x: x['ensemble_rerank_score'], reverse=True)
            
            if k is not None:
                reranked_results = reranked_results[:k]
            
            print(f"✅ Ensemble重排完成，返回 {len(reranked_results)} 个结果")
            
            if reranked_results:
                scores = [item['ensemble_rerank_score'] for item in reranked_results]
                print(f"  Ensemble分数范围: {min(scores):.3f} - {max(scores):.3f}")
            
            return reranked_results
            
        except Exception as e:
            print(f"❌ Ensemble重排失败: {e}")
            return candidates
    
    def is_available(self):
        return len(self.models) > 0


class SemanticConsistencyChecker:
    """语义一致性检查器"""
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
    
    def check_consistency(self, candidates, query_keywords):
        """检查结果的语义一致性"""
        if not self.embedding_model or len(candidates) < 2:
            return candidates
        
        print("🔍 进行语义一致性检查...")
        
        try:
            # 提取候选文本
            candidate_texts = [item['content'] for item in candidates]
            query_text = ' '.join(query_keywords)
            
            # 生成embeddings
            all_texts = [query_text] + candidate_texts
            embeddings = self.embedding_model.encode(all_texts)
            
            query_embedding = embeddings[0:1]
            candidate_embeddings = embeddings[1:]
            
            # 计算与查询的相似度
            query_similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
            
            # 计算候选文本之间的相似度矩阵
            candidate_similarities = cosine_similarity(candidate_embeddings)
            
            # 为每个候选计算一致性分数
            consistency_scores = []
            for i in range(len(candidates)):
                # 与查询的一致性
                query_consistency = query_similarities[i]
                
                # 与其他高分候选的一致性
                other_similarities = candidate_similarities[i]
                # 选择前k个高分候选（排除自己）
                top_k = min(5, len(candidates))
                top_indices = np.argsort(other_similarities)[-top_k-1:-1]  # 排除自己
                peer_consistency = np.mean(other_similarities[top_indices]) if len(top_indices) > 0 else 0
                
                # 综合一致性分数
                consistency_score = 0.7 * query_consistency + 0.3 * peer_consistency
                consistency_scores.append(consistency_score)
            
            # 添加一致性分数到候选结果
            for i, candidate in enumerate(candidates):
                candidate['consistency_score'] = consistency_scores[i]
            
            print(f"✅ 语义一致性检查完成")
            return candidates
            
        except Exception as e:
            print(f"❌ 语义一致性检查失败: {e}")
            return candidates


class DiversityOptimizer:
    """结果多样性优化器"""
    
    def __init__(self, embedding_model=None, diversity_threshold=0.8):
        self.embedding_model = embedding_model
        self.diversity_threshold = diversity_threshold
    
    def optimize_diversity(self, candidates, target_count=None):
        """优化结果多样性，避免重复内容 - 改进版本"""
        if not self.embedding_model or len(candidates) <= 1:
            return candidates
        
        print(f"🎯 优化结果多样性 (目标: {target_count or len(candidates)} 个结果)...")
        
        try:
            # 提取候选文本内容
            candidate_texts = [item['content'] for item in candidates]
            
            # 生成embeddings
            embeddings = self.embedding_model.encode(candidate_texts)
            
            # 计算相似度矩阵
            similarity_matrix = cosine_similarity(embeddings)
            
            # 改进的多样性选择算法
            selected_indices = []
            remaining_indices = list(range(len(candidates)))
            
            # 首先选择最高分的候选
            if candidates:
                best_idx = 0  # 假设已经按分数排序
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            
            # 贪心选择剩余候选，平衡质量和多样性
            target_size = target_count if target_count else min(len(candidates), 20)
            
            while len(selected_indices) < target_size and remaining_indices:
                best_candidate_idx = None
                best_score = -1
                
                for idx in remaining_indices:
                    # 计算与已选择候选的最大相似度
                    max_similarity = 0
                    for selected_idx in selected_indices:
                        similarity = similarity_matrix[idx][selected_idx]
                        max_similarity = max(max_similarity, similarity)
                    
                    # 多样性分数 (1 - 最大相似度)
                    diversity_score = 1 - max_similarity
                    
                    # 获取候选的质量分数
                    quality_score = candidates[idx].get('rerank_score', 0)
                    
                    # 综合分数：平衡质量和多样性
                    # 动态调整权重：后期更注重多样性
                    diversity_weight = 0.3 + 0.4 * (len(selected_indices) / target_size)
                    quality_weight = 1 - diversity_weight
                    
                    combined_score = quality_weight * quality_score + diversity_weight * diversity_score
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_candidate_idx = idx
                
                if best_candidate_idx is not None:
                    selected_indices.append(best_candidate_idx)
                    remaining_indices.remove(best_candidate_idx)
                else:
                    break
            
            # 构建多样化结果
            diversified_results = [candidates[i] for i in selected_indices]
            
            # 添加多样性统计信息
            if len(diversified_results) > 1:
                final_embeddings = embeddings[selected_indices]
                final_similarities = cosine_similarity(final_embeddings)
                
                # 计算平均相似度（排除对角线）
                mask = np.ones_like(final_similarities, dtype=bool)
                np.fill_diagonal(mask, False)
                avg_similarity = np.mean(final_similarities[mask])
                
                print(f"✅ 多样性优化完成，平均相似度: {avg_similarity:.3f}")
            else:
                print(f"✅ 多样性优化完成")
            
            return diversified_results
            
        except Exception as e:
            print(f"❌ 多样性优化失败: {e}")
            return candidates[:target_count] if target_count else candidates
        
        print(f"🎯 优化结果多样性 (阈值: {self.diversity_threshold})...")
        
        try:
            # 生成embeddings
            candidate_texts = [item['content'] for item in candidates]
            embeddings = self.embedding_model.encode(candidate_texts)
            
            # 贪心选择多样化结果
            selected_candidates = []
            selected_embeddings = []
            remaining_candidates = candidates.copy()
            remaining_embeddings = embeddings.copy()
            
            # 首先选择最高分的候选
            if remaining_candidates:
                best_idx = 0  # 假设已经按分数排序
                selected_candidates.append(remaining_candidates.pop(best_idx))
                selected_embeddings.append(remaining_embeddings[best_idx])
                remaining_embeddings = np.delete(remaining_embeddings, best_idx, axis=0)
            
            # 贪心选择剩余候选
            target_count = target_count or len(candidates)
            while len(selected_candidates) < target_count and remaining_candidates:
                best_candidate_idx = None
                best_score = -1
                
                for i, candidate in enumerate(remaining_candidates):
                    # 计算与已选择候选的最大相似度
                    if selected_embeddings:
                        similarities = cosine_similarity(
                            remaining_embeddings[i:i+1], 
                            np.array(selected_embeddings)
                        )[0]
                        max_similarity = np.max(similarities)
                    else:
                        max_similarity = 0
                    
                    # 多样性分数：原始分数 - 相似度惩罚
                    original_score = candidate.get('rerank_score', candidate.get('similarity_score', 0))
                    diversity_penalty = max(0, max_similarity - self.diversity_threshold)
                    diversity_score = original_score - diversity_penalty
                    
                    if diversity_score > best_score:
                        best_score = diversity_score
                        best_candidate_idx = i
                
                # 选择最佳候选
                if best_candidate_idx is not None:
                    selected_candidate = remaining_candidates.pop(best_candidate_idx)
                    selected_embedding = remaining_embeddings[best_candidate_idx]
                    
                    selected_candidate['diversity_score'] = best_score
                    selected_candidates.append(selected_candidate)
                    selected_embeddings.append(selected_embedding)
                    
                    remaining_embeddings = np.delete(remaining_embeddings, best_candidate_idx, axis=0)
                else:
                    break
            
            print(f"✅ 多样性优化完成，选择了 {len(selected_candidates)} 个多样化结果")
            return selected_candidates
            
        except Exception as e:
            print(f"❌ 多样性优化失败: {e}")
            return candidates


class ContextAwareReranker:
    """上下文感知重排器"""
    
    def __init__(self, keywords=None):
        self.keywords = keywords or []
        
        # 定义上下文线索
        self.context_clues = {
            'emotional_intensity': [
                'deeply', 'intensely', 'overwhelming', 'burning', 'consuming',
                'terrible', 'horrible', 'dreadful', 'fierce', 'passionate'
            ],
            'character_development': [
                'character', 'personality', 'nature', 'soul', 'heart', 'mind',
                'transformation', 'change', 'becoming', 'turning into'
            ],
            'action_consequence': [
                'result', 'consequence', 'outcome', 'because', 'therefore',
                'led to', 'caused', 'brought about', 'resulted in'
            ],
            'dialogue_speech': [
                'said', 'spoke', 'declared', 'exclaimed', 'whispered',
                'cried', 'shouted', 'uttered', 'proclaimed'
            ]
        }
    
    def rerank_by_context(self, candidates):
        """基于上下文线索重新排序"""
        if not candidates:
            return candidates
        
        print("🎭 基于上下文线索重排...")
        
        for candidate in candidates:
            content = candidate['content'].lower()
            context_score = 0.0
            
            # 计算各种上下文线索的分数
            for context_type, clues in self.context_clues.items():
                type_score = 0.0
                for clue in clues:
                    if clue in content:
                        type_score += 1
                
                # 归一化并加权
                if clues:
                    type_score = type_score / len(clues)
                    
                    # 不同类型的权重
                    if context_type == 'emotional_intensity':
                        context_score += type_score * 0.4
                    elif context_type == 'character_development':
                        context_score += type_score * 0.3
                    elif context_type == 'action_consequence':
                        context_score += type_score * 0.2
                    elif context_type == 'dialogue_speech':
                        context_score += type_score * 0.1
            
            # 关键词密度奖励
            keyword_density = self._calculate_keyword_density(content, candidate.get('found_keywords', []))
            context_score += keyword_density * 0.2
            
            candidate['context_score'] = context_score
        
        # 结合原始分数和上下文分数
        for candidate in candidates:
            original_score = candidate.get('rerank_score', candidate.get('similarity_score', 0))
            context_score = candidate.get('context_score', 0)
            
            # 加权组合
            combined_score = 0.7 * original_score + 0.3 * context_score
            candidate['context_aware_score'] = combined_score
        
        # 按组合分数排序
        candidates.sort(key=lambda x: x.get('context_aware_score', 0), reverse=True)
        
        print(f"✅ 上下文感知重排完成")
        return candidates
    
    def _calculate_keyword_density(self, content, found_keywords):
        """计算关键词密度"""
        if not found_keywords:
            return 0.0
        
        words = content.split()
        if not words:
            return 0.0
        
        keyword_count = sum(content.count(kw.lower()) for kw in found_keywords)
        return min(1.0, keyword_count / len(words) * 10)  # 归一化到 [0, 1]


class EnhancedCrossEncoderReranker(CrossEncoderReranker):
    """增强版Cross-Encoder重排器"""
    
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', threshold=0.1, 
                 keywords=None, enable_ensemble=False, enable_consistency_check=True,
                 enable_diversity=True, enable_context_aware=True, embedding_model=None):
        super().__init__(model_name, threshold, keywords)
        
        # 增强组件
        self.ensemble_reranker = None
        if enable_ensemble:
            self.ensemble_reranker = MultiModelEnsembleReranker(keywords=keywords)
        
        self.consistency_checker = None
        if enable_consistency_check and embedding_model:
            self.consistency_checker = SemanticConsistencyChecker(embedding_model)
        
        self.diversity_optimizer = None
        if enable_diversity and embedding_model:
            self.diversity_optimizer = DiversityOptimizer(embedding_model)
        
        self.context_reranker = None
        if enable_context_aware:
            self.context_reranker = ContextAwareReranker(keywords)
    
    def rerank(self, candidates, k=None):
        """增强版重排序 - 修复版本，确保返回足够的结果"""
        if not candidates:
            return candidates
        
        print(f"🚀 使用增强版重排器处理 {len(candidates)} 个候选...")
        
        # 确定目标结果数量 - 不过于激进地减少
        target_k = k or len(candidates)
        working_k = max(target_k, min(len(candidates), 20))  # 至少保留20个或所有候选
        
        # 阶段1：基础重排
        if self.model:
            candidates = super().rerank(candidates, k=None)  # 不在这里截断
        
        # 阶段2：Ensemble重排（可选）
        if self.ensemble_reranker and self.ensemble_reranker.is_available():
            candidates = self.ensemble_reranker.rerank(candidates, k=None)
        
        # 阶段3：语义一致性检查（不过滤结果）
        if self.consistency_checker:
            candidates = self.consistency_checker.check_consistency(candidates, self.keywords)
        
        # 阶段4：上下文感知重排（不过滤结果）
        if self.context_reranker:
            candidates = self.context_reranker.rerank_by_context(candidates)
        
        # 阶段5：多样性优化（只在结果过多时使用，且保守处理）
        if self.diversity_optimizer and len(candidates) > working_k:
            # 只在候选数量明显超过需求时才进行多样性优化
            if len(candidates) > working_k * 1.5:
                candidates = self.diversity_optimizer.optimize_diversity(candidates, working_k)
            else:
                # 轻微的多样性处理，主要基于分数排序
                candidates = sorted(candidates, 
                                  key=lambda x: x.get('context_aware_score', 
                                                     x.get('ensemble_rerank_score', 
                                                          x.get('rerank_score', 
                                                               x.get('similarity_score', 0)))), 
                                  reverse=True)
        
        # 最终截断到目标数量
        if target_k and len(candidates) > target_k:
            candidates = candidates[:target_k]
        
        print(f"✅ 增强版重排完成，最终返回 {len(candidates)} 个结果")
        
        # 显示最终分数统计
        if candidates:
            final_scores = []
            for candidate in candidates:
                # 使用最终的分数
                if 'context_aware_score' in candidate:
                    final_scores.append(candidate['context_aware_score'])
                elif 'ensemble_rerank_score' in candidate:
                    final_scores.append(candidate['ensemble_rerank_score'])
                elif 'rerank_score' in candidate:
                    final_scores.append(candidate['rerank_score'])
                else:
                    final_scores.append(candidate.get('similarity_score', 0))
            
            if final_scores:
                print(f"  最终分数范围: {min(final_scores):.3f} - {max(final_scores):.3f}")
                
                high_quality = len([s for s in final_scores if s >= 0.7])
                medium_quality = len([s for s in final_scores if 0.4 <= s < 0.7])
                low_quality = len([s for s in final_scores if s < 0.4])
                print(f"  质量分布: 高({high_quality}) 中({medium_quality}) 低({low_quality})")
        
        return candidates


class AdvancedContextAwareReranker:
    """高级上下文感知重排器 - 针对文学作品优化"""
    
    def __init__(self, embedding_model=None, keywords=None):
        self.embedding_model = embedding_model
        self.keywords = keywords or []
        
        # 文学主题的上下文模式
        self.literary_contexts = {
            'power_ambition': {
                'keywords': ['power', 'ambition', 'throne', 'crown', 'rule', 'king', 'queen'],
                'positive_indicators': ['seek', 'desire', 'want', 'crave', 'hunger', 'pursue', 'aspire'],
                'emotional_markers': ['burning', 'consuming', 'overwhelming', 'fierce', 'desperate']
            },
            'guilt_conscience': {
                'keywords': ['guilt', 'conscience', 'remorse', 'shame', 'regret'],
                'positive_indicators': ['haunt', 'torment', 'trouble', 'burden', 'weigh'],
                'emotional_markers': ['heavy', 'crushing', 'unbearable', 'terrible', 'horrible']
            },
            'fear_terror': {
                'keywords': ['fear', 'terror', 'dread', 'horror', 'panic'],
                'positive_indicators': ['grip', 'seize', 'overcome', 'fill', 'consume'],
                'emotional_markers': ['paralyzing', 'numbing', 'overwhelming', 'terrible', 'dreadful']
            },
            'love_passion': {
                'keywords': ['love', 'passion', 'affection', 'devotion', 'heart'],
                'positive_indicators': ['burn', 'consume', 'fill', 'overwhelm', 'possess'],
                'emotional_markers': ['deep', 'intense', 'burning', 'passionate', 'overwhelming']
            }
        }
    
    def _analyze_literary_context(self, content, found_keywords):
        """分析文学上下文"""
        content_lower = content.lower()
        max_context_score = 0
        
        for theme, patterns in self.literary_contexts.items():
            theme_score = 0
            
            # 检查主题关键词
            theme_keywords = sum(1 for kw in patterns['keywords'] if kw in content_lower)
            
            # 检查积极指标
            positive_count = sum(1 for indicator in patterns['positive_indicators'] if indicator in content_lower)
            
            # 检查情感标记
            emotional_count = sum(1 for marker in patterns['emotional_markers'] if marker in content_lower)
            
            # 检查用户查询关键词的匹配
            query_match = sum(1 for kw in found_keywords if kw.lower() in [k.lower() for k in patterns['keywords']])
            
            if theme_keywords > 0 or query_match > 0:
                theme_score = 0.4 * query_match + 0.3 * positive_count + 0.3 * emotional_count
                max_context_score = max(max_context_score, theme_score)
        
        return min(max_context_score, 1.0)
    
    def rerank(self, candidates, k=None):
        """使用高级上下文感知重排"""
        if not candidates:
            return candidates
        
        print(f"🧠 使用高级上下文感知重排器处理 {len(candidates)} 个候选...")
        
        for candidate in candidates:
            content = candidate['content']
            found_keywords = candidate.get('found_keywords', [])
            
            # 分析文学上下文
            context_score = self._analyze_literary_context(content, found_keywords)
            
            # 获取原有分数
            original_score = candidate.get('rerank_score', candidate.get('similarity_score', 0))
            
            # 计算上下文感知分数
            candidate['advanced_context_score'] = 0.7 * original_score + 0.3 * context_score
        
        # 按上下文感知分数排序
        candidates.sort(key=lambda x: x.get('advanced_context_score', 0), reverse=True)
        
        if k is not None:
            candidates = candidates[:k]
        
        print(f"✅ 高级上下文感知重排完成，返回 {len(candidates)} 个结果")
        return candidates
    
    def is_available(self):
        return True


def create_enhanced_reranker(method="enhanced_cross_encoder", embedding_model=None, **kwargs):
    """工厂函数：创建增强版重排器 - 优化版本"""
    
    # 高级上下文感知重排器（专为文学作品优化）
    if method == "advanced_context_aware":
        return AdvancedContextAwareReranker(
            embedding_model=embedding_model,
            keywords=kwargs.get('keywords'),
        )
    
    # 增强Cross-Encoder重排器
    elif method == "enhanced_cross_encoder":
        return EnhancedCrossEncoderReranker(
            embedding_model=embedding_model,
            enable_ensemble=kwargs.get('enable_ensemble', False),
            enable_consistency_check=kwargs.get('enable_consistency_check', True),
            enable_diversity=kwargs.get('enable_diversity', True),
            enable_context_aware=kwargs.get('enable_context_aware', True),
            **{k: v for k, v in kwargs.items() if k not in [
                'enable_ensemble', 'enable_consistency_check', 
                'enable_diversity', 'enable_context_aware'
            ]}
        )
    
    # 多模型ensemble重排器
    elif method == "ensemble":
        return MultiModelEnsembleReranker(**kwargs)
    
    # 多样性优化重排器
    elif method == "diversity_optimizer":
        return DiversityOptimizer(
            embedding_model=embedding_model,
            diversity_threshold=kwargs.get('diversity_threshold', 0.8)
        )
    
    # 语义一致性检查器
    elif method == "consistency_checker":
        return SemanticConsistencyChecker(embedding_model=embedding_model)
    
    # 基础Cross-Encoder重排器
    elif method == "cross_encoder":
        return CrossEncoderReranker(**kwargs)
    
    # 基础上下文感知重排器
    elif method == "context_aware":
        return ContextAwareReranker(keywords=kwargs.get('keywords'))
    
    # 简单重排器
    elif method == "simple":
        return SimpleReranker(**kwargs)
    
    # 默认返回高级上下文感知重排器（最适合文学作品）
    else:
        return AdvancedContextAwareReranker(
            embedding_model=embedding_model,
            keywords=kwargs.get('keywords'),
        )
