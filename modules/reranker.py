#!/usr/bin/env python3
"""
重排器模块 - 基础版本
提供基本的重排功能
"""

import numpy as np
from sentence_transformers import CrossEncoder
import re
from collections import defaultdict


class SimpleReranker:
    """简单重排器：基于关键词匹配和智能上下文规则 - 精确优化版本"""
    
    def __init__(self, keywords=None, threshold=0.1):
        self.keywords = keywords or []
        self.threshold = threshold
        
        # 增强的主题相关权重
        self.keyword_weights = {
            'ambition': 1.6, 'power': 1.5, 'desire': 1.4, 'guilt': 1.4,
            'fear': 1.3, 'love': 1.3, 'dream': 1.2, 'death': 1.5,
            'murder': 1.6, 'blood': 1.4, 'betrayal': 1.4, 'madness': 1.3,
            'conscience': 1.4, 'remorse': 1.3, 'shame': 1.2, 'honor': 1.2,
            'destiny': 1.2, 'fate': 1.3, 'prophecy': 1.3, 'crown': 1.4,
            'throne': 1.5, 'king': 1.3, 'queen': 1.3, 'evil': 1.4
        }
        
        # 语境质量指标词汇
        self.quality_indicators = {
            'high_literary': ['prophecy', 'portent', 'omen', 'vision', 'destiny', 'fate', 
                            'conscience', 'soul', 'spirit', 'heart', 'passion'],
            'dramatic_action': ['murder', 'kill', 'death', 'blood', 'revenge', 'betray',
                              'crown', 'throne', 'power', 'rule', 'command'],
            'emotional_depth': ['love', 'hate', 'fear', 'terror', 'grief', 'joy',
                               'despair', 'hope', 'rage', 'fury', 'guilt', 'shame'],
            'dialogue_markers': ['said', 'cried', 'whispered', 'shouted', 'replied',
                                'answered', 'asked', 'exclaimed', 'declared']
        }
        
        # 负面质量指标
        self.quality_detractors = [
            'stage direction', 'scene', 'act', 'enter', 'exit', 'exeunt',
            'aside', 'soliloquy', 'to the audience'
        ]
    
    def rerank(self, candidates, k=None):
        """超精确多维度重排 - 大幅提升quote相关性捕捉能力"""
        if not candidates:
            return candidates
        
        print(f"🔄 使用超精确多维度重排器处理 {len(candidates)} 个候选...")
        
        for candidate in candidates:
            content = candidate['content'].lower()
            found_keywords = candidate.get('found_keywords', [])
            original_content = candidate['content']  # 保留原始大小写用于某些分析
            word_count = len(content.split())
            
            # === 第一层：增强的加权关键词密度分析 ===
            weighted_keyword_score = 0
            total_keyword_frequency = 0
            keyword_context_bonus = 0
            
            for kw in found_keywords:
                kw_lower = kw.lower()
                weight = self.keyword_weights.get(kw_lower, 1.0)
                
                # 多种匹配方式计算频率
                exact_freq = content.count(kw_lower)
                partial_freq = sum(1 for word in content.split() if kw_lower in word.lower())
                root_freq = sum(1 for word in content.split() if word.lower().startswith(kw_lower[:4]) and len(kw_lower) > 4)
                
                # 综合频率计算
                total_freq = exact_freq + partial_freq * 0.7 + root_freq * 0.4
                total_keyword_frequency += total_freq
                weighted_keyword_score += total_freq * weight
                
                # 关键词上下文质量评估
                if kw_lower in ['ambition', 'power', 'desire']:
                    power_contexts = ['throne', 'crown', 'king', 'queen', 'rule', 'control', 'authority', 'command']
                    context_matches = sum(1 for ctx in power_contexts if ctx in content)
                    keyword_context_bonus += context_matches * 0.15
                
                elif kw_lower in ['guilt', 'conscience', 'shame']:
                    moral_contexts = ['murder', 'blood', 'sin', 'wrong', 'evil', 'repent', 'forgive']
                    context_matches = sum(1 for ctx in moral_contexts if ctx in content)
                    keyword_context_bonus += context_matches * 0.2
                
                elif kw_lower in ['fear', 'terror', 'dread']:
                    fear_contexts = ['dark', 'horrible', 'dreadful', 'nightmare', 'death', 'murder']
                    context_matches = sum(1 for ctx in fear_contexts if ctx in content)
                    keyword_context_bonus += context_matches * 0.18
            
            # === 第二层：关键词多样性和分布分析 ===
            unique_keywords = set(found_keywords)
            keyword_diversity = len(unique_keywords) / max(len(self.keywords), 1)
            
            # 关键词分布权重 - 更精确的位置分析
            position_weight = 0
            content_words = content.split()
            
            if len(content_words) > 0:
                for kw in found_keywords:
                    kw_positions = []
                    for i, word in enumerate(content_words):
                        if kw.lower() in word.lower():
                            kw_positions.append(i)
                    
                    for pos in kw_positions:
                        relative_pos = pos / len(content_words)
                        # 更细致的位置权重分配
                        if relative_pos < 0.1:  # 开头10%
                            position_weight += 0.5
                        elif relative_pos > 0.9:  # 结尾10%
                            position_weight += 0.4
                        elif 0.4 <= relative_pos <= 0.6:  # 中心区域
                            position_weight += 0.3
                        elif relative_pos < 0.25 or relative_pos > 0.75:  # 前后25%
                            position_weight += 0.2
                        else:
                            position_weight += 0.1
            
            # === 第三层：内容质量和结构精确评估 ===
            # 长度质量评分（更精确的区间）
            if 25 <= word_count <= 60:      # 理想长度
                length_score = 1.0
            elif 15 <= word_count <= 100:   # 良好长度
                length_score = 0.9
            elif 10 <= word_count <= 150:   # 可接受长度
                length_score = 0.7
            elif word_count >= 5:           # 最小可用长度
                length_score = 0.5
            else:
                length_score = 0.2           # 过短内容
            
            # === 第四层：深度情感和文学质量分析 ===
            # 扩展的情感强度词汇
            high_intensity_words = [
                'overwhelming', 'consuming', 'burning', 'fierce', 'intense', 'powerful',
                'desperate', 'passionate', 'furious', 'terrifying', 'devastating', 'profound',
                'utter', 'complete', 'absolute', 'total', 'unbearable', 'excruciating'
            ]
            
            medium_intensity_words = [
                'strong', 'deep', 'significant', 'considerable', 'substantial', 'notable',
                'marked', 'serious', 'heavy', 'severe', 'acute', 'grave'
            ]
            
            literary_excellence_words = [
                'heart', 'soul', 'spirit', 'mind', 'conscience', 'breath', 'eyes', 'voice',
                'thought', 'feeling', 'emotion', 'memory', 'dream', 'vision', 'hope', 'fear'
            ]
            
            dramatic_power_words = [
                'murder', 'death', 'kill', 'blood', 'crown', 'throne', 'power', 'ambition',
                'guilt', 'betrayal', 'revenge', 'justice', 'fate', 'destiny', 'prophecy'
            ]
            
            # 计算各类得分
            high_intensity_count = sum(1 for word in high_intensity_words if word in content)
            medium_intensity_count = sum(1 for word in medium_intensity_words if word in content)
            literary_count = sum(1 for word in literary_excellence_words if word in content)
            dramatic_count = sum(1 for word in dramatic_power_words if word in content)
            
            # 归一化情感和文学得分
            intensity_score = min(1.0, (high_intensity_count * 0.4 + medium_intensity_count * 0.2))
            literary_score = min(1.0, literary_count / 4)
            dramatic_score = min(1.0, dramatic_count / 3)
            
            # === 第五层：对话和叙述平衡分析 ===
            quote_indicators = content.count('"') + content.count("'")
            dialogue_verbs = ['said', 'replied', 'answered', 'asked', 'cried', 'whispered', 'shouted', 'exclaimed', 'declared', 'muttered']
            dialogue_verb_count = sum(1 for verb in dialogue_verbs if verb in content)
            
            dialogue_score = min(1.0, (quote_indicators * 0.1 + dialogue_verb_count * 0.2))
            
            # === 第六层：语境连贯性和完整性评估 ===
            coherence_score = 1.0
            
            # 句子完整性检查
            if not content.strip().endswith(('.', '!', '?', '"', "'")):
                coherence_score *= 0.85
            
            # 碎片化检查
            sentence_count = len([s for s in content.split('.') if s.strip()])
            if sentence_count == 0:
                coherence_score *= 0.7
            elif word_count > 0 and word_count / max(sentence_count, 1) < 4:  # 句子过短
                coherence_score *= 0.9
            
            # 内容连贯性检查
            if content.count(',') + content.count(';') + content.count(':') > word_count * 0.3:
                coherence_score *= 0.9  # 标点过多可能表示碎片化
            
            # === 第七层：主题相关性深度分析 ===
            theme_relevance_bonus = 0
            
            # 检查主题词汇集群
            theme_clusters = {
                'power_ambition': ['power', 'ambition', 'throne', 'crown', 'king', 'queen', 'rule', 'authority', 'control'],
                'guilt_conscience': ['guilt', 'conscience', 'shame', 'remorse', 'regret', 'sin', 'wrong', 'evil'],
                'love_passion': ['love', 'passion', 'heart', 'soul', 'devotion', 'affection', 'beloved'],
                'death_violence': ['death', 'murder', 'kill', 'blood', 'violence', 'grave', 'corpse'],
                'fear_terror': ['fear', 'terror', 'dread', 'horror', 'panic', 'frightened', 'afraid'],
                'supernatural': ['dream', 'vision', 'prophecy', 'ghost', 'spirit', 'weird', 'supernatural']
            }
            
            for theme_name, theme_words in theme_clusters.items():
                cluster_matches = sum(1 for word in theme_words if word in content)
                if cluster_matches >= 2:  # 至少2个相关词汇才算主题相关
                    theme_relevance_bonus += cluster_matches * 0.08
            
            # === 综合评分计算 - 精细调整的权重系统 ===
            # 归一化基础分数
            normalized_keyword_score = weighted_keyword_score / max(word_count, 1)
            
            # 添加文学分析奖励
            literary_analysis_bonus = self._incorporate_literary_analysis(candidate)
            
            # 多维度分数组合（权重经过精心调优）
            base_score = 0.30 * normalized_keyword_score        # 关键词密度基础分
            diversity_bonus = 0.15 * keyword_diversity          # 关键词多样性
            context_bonus = 0.12 * min(keyword_context_bonus, 1.0)  # 关键词上下文
            position_bonus = 0.10 * min(position_weight, 1.0)   # 位置权重
            quality_bonus = 0.08 * length_score                 # 长度质量
            intensity_bonus = 0.08 * intensity_score            # 情感强度
            literary_bonus = 0.07 * literary_score              # 文学质量
            dramatic_bonus = 0.06 * dramatic_score              # 戏剧性
            dialogue_bonus = 0.02 * dialogue_score              # 对话内容
            coherence_bonus = 0.02 * coherence_score            # 连贯性
            theme_bonus = min(theme_relevance_bonus, 0.15)      # 主题相关性奖励（上限15%）
            analysis_bonus = literary_analysis_bonus            # 文学分析奖励
            
            final_rerank_score = (base_score + diversity_bonus + context_bonus + position_bonus + 
                                quality_bonus + intensity_bonus + literary_bonus + 
                                dramatic_bonus + dialogue_bonus + coherence_bonus + theme_bonus + analysis_bonus)
            
            candidate['rerank_score'] = min(final_rerank_score, 1.0)
            
            # 详细得分信息（用于调试和优化）
            candidate['score_breakdown'] = {
                'keyword_density': base_score,
                'keyword_diversity': diversity_bonus,
                'keyword_context': context_bonus,
                'position_weight': position_bonus,
                'length_quality': quality_bonus,
                'emotional_intensity': intensity_bonus,
                'literary_quality': literary_bonus,
                'dramatic_elements': dramatic_bonus,
                'dialogue_content': dialogue_bonus,
                'coherence': coherence_bonus,
                'theme_relevance': theme_bonus,
                'literary_analysis': analysis_bonus,
                'total_keywords': len(found_keywords),
                'keyword_frequency': total_keyword_frequency,
                'word_count': word_count
            }
        
        # 按重排分数排序
        candidates.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        if k is not None:
            candidates = candidates[:k]
        
        print(f"✅ 超精确多维度重排完成，返回 {len(candidates)} 个高质量结果")
        if candidates:
            top_score = candidates[0]['rerank_score']
            avg_score = np.mean([c['rerank_score'] for c in candidates])
            print(f"  重排分数: 最高({top_score:.3f}) 平均({avg_score:.3f})")
            
            # 显示得分分布统计
            score_ranges = {'高分(>0.7)': 0, '中分(0.4-0.7)': 0, '低分(<0.4)': 0}
            for c in candidates:
                score = c['rerank_score']
                if score > 0.7:
                    score_ranges['高分(>0.7)'] += 1
                elif score > 0.4:
                    score_ranges['中分(0.4-0.7)'] += 1
                else:
                    score_ranges['低分(<0.4)'] += 1
            print(f"  得分分布: {score_ranges}")
        
        return candidates

    def is_available(self):
        return True


class CrossEncoderReranker:
    """Cross-Encoder重排器 - 优化版本，提高精确性"""
    
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', threshold=0.1, keywords=None):
        self.model_name = model_name
        self.threshold = threshold
        self.keywords = keywords or []
        self.model = None
        
        # 增强的主题相关查询扩展
        self.query_expansions = {
            'ambition': 'ambition desire for power seeking throne control authority dominion rule sovereignty',
            'power': 'power authority control dominion rule command sovereignty kingship monarchy crown throne',
            'guilt': 'guilt conscience remorse shame regret repentance sin wrong crime fault blame',
            'fear': 'fear terror dread anxiety horror panic apprehension fright alarm worry',
            'love': 'love passion affection devotion romance heart soul tender gentle sweet dear',
            'desire': 'desire want need craving longing yearning hunger thirst wish hope dream',
            'dream': 'dream vision nightmare prophecy sleep unconscious weird supernatural portent omen sign',
            'betrayal': 'betrayal treachery deceit disloyalty backstab false lie cheat trick deceive',
            'madness': 'madness insanity lunacy derangement mental illness crazy mad reason sanity mind',
            'death': 'death murder kill blood violence destruction grave tomb corpse ghost spirit',
            'murder': 'murder kill death blood violence crime sin evil wicked dark terrible',
            'blood': 'blood bloody murder kill death violence red stain guilt crime',
            'crown': 'crown throne king queen royal power authority rule sovereignty dominion',
            'conscience': 'conscience guilt remorse shame regret moral ethics virtue sin wrong'
        }
        
        # 文学质量评估标准
        self.literary_quality_markers = {
            'metaphorical': ['like', 'as', 'seems', 'appears', 'resembles', 'symbol', 'metaphor'],
            'dramatic': ['terrible', 'horrible', 'dreadful', 'awful', 'fearful', 'dark', 'evil'],
            'emotional': ['passion', 'burning', 'consuming', 'overwhelming', 'intense', 'deep'],
            'philosophical': ['soul', 'spirit', 'mind', 'conscience', 'reason', 'thought', 'meditation']
        }
        
        try:
            print(f"正在加载Cross-Encoder模型: {model_name}")
            # 添加设备参数和更安全的加载方式
            self.model = CrossEncoder(model_name, device='cpu', trust_remote_code=False)
            print(f"✅ 模型 {model_name} 加载成功")
        except Exception as e:
            print(f"⚠️ 模型 {model_name} 加载失败: {e}")
            
            # 尝试备用模型
            backup_models = [
                'cross-encoder/ms-marco-TinyBERT-L-2-v2',
                'cross-encoder/ms-marco-MiniLM-L-2-v2'
            ]
            
            for backup_model in backup_models:
                if backup_model != model_name:
                    try:
                        print(f"🔄 尝试备用模型: {backup_model}")
                        self.model = CrossEncoder(backup_model, device='cpu', trust_remote_code=False)
                        print(f"✅ 备用模型 {backup_model} 加载成功")
                        self.model_name = backup_model
                        break
                    except Exception as e2:
                        print(f"⚠️ 备用模型 {backup_model} 也加载失败: {e2}")
                        continue
            
            if self.model is None:
                print("❌ 所有Cross-Encoder模型都加载失败，将使用基础重排方法")
                self.model = None
    
    def _build_enhanced_query(self, keywords):
        """构建多层次增强查询，大幅提高重排精确性"""
        base_query = ' '.join(keywords)
        
        # 1. 基础查询扩展
        expanded_terms = set()
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in self.query_expansions:
                expanded_terms.update(self.query_expansions[keyword_lower].split())
        
        # 2. 构建分层查询结构
        queries = []
        
        # 主查询：直接关键词匹配
        queries.append(f"passages about {base_query}")
        
        # 扩展查询：包含相关概念
        if expanded_terms:
            expanded_query = ' '.join(list(expanded_terms)[:10])  # 限制扩展词数量
            queries.append(f"text containing themes of {base_query} including {expanded_query}")
        
        # 文学意义查询：强调文学价值
        queries.append(f"literary passages with dramatic significance about {base_query}")
        
        # 情感深度查询：强调情感内容
        queries.append(f"emotionally intense content about {base_query} with deep meaning")
        
        # 返回主查询（用于兼容性）
        primary_query = queries[0]
        
        # 存储所有查询用于多查询评估
        self._all_queries = queries
        
        return primary_query
    
    def _calculate_advanced_content_quality(self, content, found_keywords):
        """高级内容质量计算 - 多维度精确评估"""
        content_lower = content.lower()
        
        # 1. 文学质量评估
        literary_score = 0
        for category, markers in self.literary_quality_markers.items():
            matches = sum(1 for marker in markers if marker in content_lower)
            category_score = min(matches / 3, 1.0)  # 归一化到0-1
            
            # 加权不同类别
            weights = {'metaphorical': 0.3, 'dramatic': 0.3, 'emotional': 0.25, 'philosophical': 0.15}
            literary_score += category_score * weights.get(category, 0.25)
        
        # 2. 关键词语境质量
        context_quality = 0
        for keyword in found_keywords:
            keyword_lower = keyword.lower()
            
            # 检查关键词的上下文使用
            if keyword_lower in self.query_expansions:
                expansion_words = self.query_expansions[keyword_lower].split()
                context_matches = sum(1 for word in expansion_words if word in content_lower)
                context_ratio = context_matches / len(expansion_words)
                context_quality += context_ratio
        
        if found_keywords:
            context_quality /= len(found_keywords)
        
        # 3. 结构完整性评估
        structural_quality = 1.0
        word_count = len(content.split())
        
        # 长度适中性
        if word_count < 10:
            structural_quality *= 0.6
        elif word_count > 200:
            structural_quality *= 0.8
        elif 20 <= word_count <= 100:
            structural_quality *= 1.0
        else:
            structural_quality *= 0.9
        
        # 完整性检查
        if not content.strip().endswith(('.', '!', '?', '"', "'")):
            structural_quality *= 0.8
        
        # 4. 对话和叙述平衡
        dialogue_markers = sum(1 for marker in ['"', "'", 'said', 'replied'] if marker in content_lower)
        narrative_markers = sum(1 for marker in ['he', 'she', 'it', 'they', 'was', 'were'] if marker in content_lower)
        balance_score = 1.0
        if dialogue_markers > 0 or narrative_markers > 0:
            total_markers = dialogue_markers + narrative_markers
            if total_markers > 0:
                dialogue_ratio = dialogue_markers / total_markers
                # 理想的对话-叙述比例
                if 0.2 <= dialogue_ratio <= 0.8:
                    balance_score = 1.0
                else:
                    balance_score = 0.8
        
        # 综合质量分数
        final_quality = (
            0.35 * literary_score +
            0.30 * context_quality +
            0.20 * structural_quality +
            0.15 * balance_score
        )
        
        return min(final_quality, 1.0)
    
    def _calculate_content_quality_score(self, content, found_keywords):
        """计算内容质量分数"""
        content_lower = content.lower()
        
        # 文学质量指标
        literary_indicators = [
            'said', 'cried', 'whispered', 'shouted', 'exclaimed', 'replied',
            'thought', 'felt', 'saw', 'heard', 'knew', 'believed',
            'heart', 'soul', 'mind', 'spirit', 'eyes', 'face'
        ]
        
        dramatic_indicators = [
            'terrible', 'horrible', 'dreadful', 'fierce', 'burning', 'consuming',
            'overwhelming', 'desperate', 'intense', 'powerful', 'deep', 'strong',
            'evil', 'wicked', 'dark', 'bloody', 'murder', 'death', 'kill'
        ]
        
        dialogue_indicators = ['"', "'", 'said', 'replied', 'answered', 'asked']
        
        # 计算各种指标
        literary_score = sum(1 for word in literary_indicators if word in content_lower) / len(literary_indicators)
        dramatic_score = sum(1 for word in dramatic_indicators if word in content_lower) / len(dramatic_indicators)
        dialogue_score = sum(1 for indicator in dialogue_indicators if indicator in content_lower) / len(dialogue_indicators)
        
        # 关键词集中度
        keyword_density = len(found_keywords) / max(len(content.split()), 1)
        
        # 综合质量分数
        quality_score = 0.3 * literary_score + 0.4 * dramatic_score + 0.2 * dialogue_score + 0.1 * keyword_density
        
        return min(quality_score, 1.0)
    
    def rerank(self, candidates, k=None):
        """
        终极精确Cross-Encoder重排 - 多查询变体 + 深度质量融合
        实现最精确的quote相关性判断和文学价值评估
        """
        if not self.model or not candidates:
            return candidates
        
        print(f"� 使用终极精确Cross-Encoder重排器处理 {len(candidates)} 个候选...")
        
        try:
            # === 构建多层次查询体系 ===
            primary_query = ' '.join(self.keywords)
            enhanced_query = self._build_enhanced_query(self.keywords)
            
            # 高级查询变体 - 捕捉不同层面的相关性
            query_variants = [
                primary_query,  # 基础查询
                enhanced_query,  # 扩展查询
                f"literary passages about {primary_query} with deep meaning",  # 文学深度
                f"dramatic and significant quotes about {primary_query}",  # 戏剧性
                f"emotionally powerful text containing {primary_query}",  # 情感强度
                f"thematically relevant content about {primary_query}",  # 主题相关
                f"contextually meaningful {primary_query} in literature"  # 语境意义
            ]
            
            # 为每个候选生成多个查询-文本对
            query_text_pairs = []
            candidate_indices = []
            query_types = []
            
            for i, item in enumerate(candidates):
                content = item['content']
                
                for j, query in enumerate(query_variants):
                    query_text_pairs.append((query, content))
                    candidate_indices.append(i)
                    query_types.append(j)
            
            print(f"  生成 {len(query_text_pairs)} 个多维度查询-文本对进行深度评估...")
            
            # 批量预测相关性分数（使用较小的batch_size确保稳定性）
            batch_size = min(32, len(query_text_pairs))
            all_scores = []
            
            for i in range(0, len(query_text_pairs), batch_size):
                batch_pairs = query_text_pairs[i:i + batch_size]
                batch_scores = self.model.predict(batch_pairs, show_progress_bar=False)
                all_scores.extend(batch_scores)
            
            # === 智能分数聚合和融合 ===
            candidate_scores = defaultdict(lambda: defaultdict(list))
            
            # 按候选和查询类型分组分数
            for score, candidate_idx, query_type in zip(all_scores, candidate_indices, query_types):
                candidate_scores[candidate_idx][query_type].append(max(0.0, float(score)))
            
            # 计算每个候选的多维度综合分数
            for i, candidate in enumerate(candidates):
                if i in candidate_scores:
                    scores_by_type = candidate_scores[i]
                    
                    # 不同查询类型的精细化权重
                    query_weights = [0.25, 0.20, 0.15, 0.12, 0.10, 0.10, 0.08]
                    
                    # 计算加权Cross-Encoder分数
                    weighted_ce_score = 0
                    total_weight = 0
                    
                    for query_type, weight in enumerate(query_weights):
                        if query_type in scores_by_type and scores_by_type[query_type]:
                            type_score = np.mean(scores_by_type[query_type])
                            weighted_ce_score += type_score * weight
                            total_weight += weight
                    
                    if total_weight > 0:
                        weighted_ce_score /= total_weight
                    
                    # 计算分数变异性（一致性指标）
                    all_candidate_scores = []
                    for type_scores in scores_by_type.values():
                        all_candidate_scores.extend(type_scores)
                    
                    if all_candidate_scores:
                        score_std = np.std(all_candidate_scores)
                        score_consistency = 1.0 - min(score_std, 0.5)  # 分数越一致，权重越高
                    else:
                        score_consistency = 0.0
                    
                    candidate['multi_query_ce_score'] = weighted_ce_score
                    candidate['score_consistency'] = score_consistency
                else:
                    candidate['multi_query_ce_score'] = 0.0
                    candidate['score_consistency'] = 0.0
                
                # === 深度内容质量分析 ===
                content = candidate['content']
                found_keywords = candidate.get('found_keywords', [])
                similarity_score = candidate.get('similarity_score', 0)
                
                # 高级内容质量评估
                content_quality = self._calculate_advanced_content_quality(content, found_keywords)
                
                # 结构和完整性评估
                structural_quality = self._evaluate_structural_integrity(content)
                
                # 文学价值和深度评估
                literary_depth = self._evaluate_literary_depth(content, found_keywords)
                
                # 主题一致性和相关性
                thematic_relevance = self._evaluate_thematic_relevance(content, found_keywords)
                
                # 情感强度和戏剧效果
                emotional_impact = self._evaluate_emotional_impact(content)
                
                # === 多维度分数融合 ===
                # 基础相似度组件
                similarity_component = similarity_score * 0.12
                
                # Cross-Encoder组件（主要权重）
                ce_component = candidate['multi_query_ce_score'] * 0.40
                
                # 分数一致性奖励
                consistency_component = candidate['score_consistency'] * 0.05
                
                # 内容质量组件
                quality_component = content_quality * 0.18
                
                # 结构质量组件
                structure_component = structural_quality * 0.08
                
                # 文学深度组件
                literary_component = literary_depth * 0.08
                
                # 主题相关性组件
                theme_component = thematic_relevance * 0.06
                
                # 情感影响组件
                emotion_component = emotional_impact * 0.03
                
                # 最终综合分数
                final_score = (similarity_component + ce_component + consistency_component +
                              quality_component + structure_component + literary_component + 
                              theme_component + emotion_component)
                
                candidate['rerank_score'] = min(final_score, 1.0)
                
                # 详细评分分解（用于分析和调试）
                candidate['score_breakdown'] = {
                    'similarity_base': similarity_component,
                    'multi_query_ce': ce_component,
                    'score_consistency': consistency_component,
                    'content_quality': quality_component,
                    'structural_quality': structure_component,
                    'literary_depth': literary_component,
                    'thematic_relevance': theme_component,
                    'emotional_impact': emotion_component,
                    'final_score': final_score
                }
            
        except Exception as e:
            print(f"⚠️ Cross-Encoder批量评估失败: {e}")
            # 回退到逐个处理
            self._fallback_individual_processing(candidates)
        
        # 按最终重排分数排序
        candidates.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        if k is not None:
            candidates = candidates[:k]
        
        print(f"✅ 终极精确Cross-Encoder重排完成，返回 {len(candidates)} 个顶级结果")
        
        if candidates:
            top_score = candidates[0]['rerank_score']
            avg_score = np.mean([c['rerank_score'] for c in candidates])
            top_ce_score = candidates[0].get('multi_query_ce_score', 0)
            avg_ce_score = np.mean([c.get('multi_query_ce_score', 0) for c in candidates])
            
            print(f"  最终分数: 最高({top_score:.3f}) 平均({avg_score:.3f})")
            print(f"  多查询CE分数: 最高({top_ce_score:.3f}) 平均({avg_ce_score:.3f})")
            
            # 质量分布分析
            excellence_tier = len([c for c in candidates if c['rerank_score'] > 0.8])
            high_quality = len([c for c in candidates if 0.6 <= c['rerank_score'] <= 0.8])
            medium_quality = len([c for c in candidates if 0.4 <= c['rerank_score'] < 0.6])
            
            print(f"  质量分布: 卓越({excellence_tier}) 高质量({high_quality}) 中等({medium_quality}) 其他({len(candidates)-excellence_tier-high_quality-medium_quality})")
        
        return candidates
    
    def _evaluate_structural_integrity(self, content):
        """评估文本结构完整性"""
        words = content.split()
        word_count = len(words)
        
        # 长度适中性评分
        if 20 <= word_count <= 80:
            length_score = 1.0
        elif 15 <= word_count <= 120:
            length_score = 0.85
        elif 10 <= word_count <= 150:
            length_score = 0.7
        else:
            length_score = 0.5
        
        # 完整性评分
        completeness_score = 1.0 if content.strip().endswith(('.', '!', '?', '"', "'")) else 0.8
        
        # 句子结构评分
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if sentences:
            avg_sentence_length = word_count / len(sentences)
            if 8 <= avg_sentence_length <= 25:
                structure_score = 1.0
            else:
                structure_score = 0.8
        else:
            structure_score = 0.6
        
        return (length_score + completeness_score + structure_score) / 3
    
    def _evaluate_literary_depth(self, content, found_keywords):
        """评估文学深度和艺术价值"""
        content_lower = content.lower()
        
        # 文学设备检测
        literary_devices = [
            'metaphor', 'simile', 'imagery', 'symbol', 'symbolism', 'irony', 'paradox',
            'foreshadowing', 'allegory', 'personification', 'dramatic', 'tragic'
        ]
        
        # 深度情感词汇
        profound_emotions = [
            'anguish', 'ecstasy', 'despair', 'rapture', 'torment', 'bliss',
            'melancholy', 'euphoria', 'desolation', 'transcendence'
        ]
        
        # 哲学和心理词汇
        philosophical_terms = [
            'soul', 'spirit', 'conscience', 'consciousness', 'existence', 'mortality',
            'destiny', 'fate', 'meaning', 'purpose', 'truth', 'reality'
        ]
        
        device_count = sum(1 for device in literary_devices if device in content_lower)
        emotion_count = sum(1 for emotion in profound_emotions if emotion in content_lower)
        philosophy_count = sum(1 for term in philosophical_terms if term in content_lower)
        
        # 综合文学深度分数
        literary_score = min((device_count * 0.3 + emotion_count * 0.4 + philosophy_count * 0.3), 1.0)
        
        return literary_score
    
    def _evaluate_thematic_relevance(self, content, found_keywords):
        """评估主题相关性和一致性"""
        content_lower = content.lower()
        
        # 主题词汇集群
        theme_clusters = {
            'power_ambition': ['power', 'ambition', 'throne', 'crown', 'king', 'queen', 'rule', 'authority', 'control', 'dominion'],
            'moral_guilt': ['guilt', 'conscience', 'shame', 'remorse', 'sin', 'wrong', 'evil', 'repent', 'forgive'],
            'love_passion': ['love', 'passion', 'heart', 'soul', 'beloved', 'devotion', 'affection', 'romance'],
            'death_violence': ['death', 'murder', 'kill', 'blood', 'violence', 'grave', 'corpse', 'ghost'],
            'fear_terror': ['fear', 'terror', 'dread', 'horror', 'panic', 'frightened', 'afraid'],
            'fate_destiny': ['fate', 'destiny', 'doom', 'prophecy', 'future', 'inevitable', 'predetermined']
        }
        
        # 计算主题一致性
        max_cluster_score = 0
        for cluster_name, cluster_words in theme_clusters.items():
            cluster_matches = sum(1 for word in cluster_words if word in content_lower)
            if cluster_matches >= 2:  # 至少2个相关词才算主题一致
                cluster_score = min(cluster_matches / len(cluster_words), 0.8)
                max_cluster_score = max(max_cluster_score, cluster_score)
        
        # 关键词密度奖励
        keyword_density = len(found_keywords) / max(len(content.split()), 1)
        density_score = min(keyword_density * 10, 0.3)  # 最多30%奖励
        
        return max_cluster_score + density_score
    
    def _evaluate_emotional_impact(self, content):
        """评估情感冲击力和戏剧效果"""
        content_lower = content.lower()
        
        # 高冲击情感词汇
        high_impact_emotions = [
            'overwhelming', 'devastating', 'crushing', 'shattering', 'excruciating',
            'unbearable', 'agonizing', 'heart-wrenching', 'soul-crushing'
        ]
        
        # 戏剧性动作词汇
        dramatic_actions = [
            'murder', 'kill', 'betray', 'deceive', 'destroy', 'shatter', 'crush',
            'reveal', 'discover', 'confront', 'confess', 'expose'
        ]
        
        # 强度副词
        intensity_adverbs = [
            'utterly', 'completely', 'absolutely', 'totally', 'entirely',
            'deeply', 'profoundly', 'intensely', 'desperately'
        ]
        
        impact_count = sum(1 for word in high_impact_emotions if word in content_lower)
        drama_count = sum(1 for word in dramatic_actions if word in content_lower)
        intensity_count = sum(1 for word in intensity_adverbs if word in content_lower)
        
        # 综合情感冲击分数
        emotional_score = min((impact_count * 0.4 + drama_count * 0.3 + intensity_count * 0.3), 1.0)
        
        return emotional_score
    
    def _fallback_individual_processing(self, candidates):
        """回退处理：逐个处理候选"""
        print("  回退到逐个处理模式...")
        primary_query = ' '.join(self.keywords)
        
        for candidate in candidates:
            try:
                content = candidate['content']
                pairs = [(primary_query, content)]
                scores = self.model.predict(pairs)
                
                ce_score = float(scores[0]) if isinstance(scores, np.ndarray) else float(scores)
                candidate['multi_query_ce_score'] = max(0.0, ce_score)
                candidate['score_consistency'] = 1.0  # 单查询默认一致性高
                
            except Exception as e:
                print(f"⚠️ 个别候选处理失败: {e}")
                candidate['multi_query_ce_score'] = 0.0
                candidate['score_consistency'] = 0.0

    def is_available(self):
        """检查模型是否可用"""
        return self.model is not None
    
    def _calculate_keyword_relevance_score(self, content, found_keywords):
        """计算关键词相关性分数"""
        if not found_keywords:
            return 0.0
        
        content_lower = content.lower()
        
        # 关键词密度
        keyword_density = len(found_keywords) / max(len(content.split()), 1)
        
        # 关键词多样性
        unique_keywords = len(set([kw.lower() for kw in found_keywords]))
        keyword_diversity = unique_keywords / max(len(self.keywords), 1)
        
        # 关键词上下文强度
        context_strength = 0
        for kw in found_keywords:
            if kw.lower() in self.query_expansions:
                expansion_words = self.query_expansions[kw.lower()].split()
                context_matches = sum(1 for word in expansion_words if word in content_lower)
                context_strength += context_matches / len(expansion_words)
        
        context_strength = context_strength / max(len(found_keywords), 1)
        
        # 综合分数
        relevance_score = 0.4 * keyword_density + 0.3 * keyword_diversity + 0.3 * context_strength
        return min(relevance_score * 5, 1.0)  # 放大分数但限制在1.0以内
    
    def _calculate_structural_quality_score(self, content):
        """计算结构质量分数"""
        # 长度适中性
        word_count = len(content.split())
        if 15 <= word_count <= 100:
            length_score = 1.0
        elif 10 <= word_count <= 150:
            length_score = 0.8
        else:
            length_score = 0.6
        
        # 完整性（是否有完整的句子结构）
        completeness_score = 1.0
        if not content.strip().endswith(('.', '!', '?', '"', "'")):
            completeness_score *= 0.8
        
        # 标点符号的合理性
        punctuation_count = sum(1 for char in content if char in '.!?;:,')
        punctuation_ratio = punctuation_count / max(word_count, 1)
        if 0.05 <= punctuation_ratio <= 0.3:
            punctuation_score = 1.0
        else:
            punctuation_score = 0.7
        
        return 0.5 * length_score + 0.3 * completeness_score + 0.2 * punctuation_score
    
    def _calculate_retrieval_confidence(self, item):
        """基于检索阶段信息计算信心分数"""
        confidence = 0.5  # 基础分数
        
        # 如果有similarity_score
        if 'similarity_score' in item:
            sim_score = item['similarity_score']
            if sim_score > 0.5:
                confidence += 0.3
            elif sim_score > 0.3:
                confidence += 0.2
            elif sim_score > 0.1:
                confidence += 0.1
        
        # 如果有检索阶段信息
        if 'stage' in item:
            stage = item['stage']
            if stage == 'high_quality':
                confidence += 0.2
            elif stage == 'balanced':
                confidence += 0.1
        
        # 如果有多种匹配类型
        if item.get('found_keywords') and 'similarity_score' in item:
            if item['similarity_score'] > 0.2:  # 混合匹配
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def is_available(self):
        return self.model is not None
    
    def _incorporate_literary_analysis(self, candidate):
        """利用文学分析结果提升重排精度"""
        literary_analysis = candidate.get('literary_analysis', {})
        if not literary_analysis:
            return 0
        
        literary_bonus = 0
        
        # 主题元素相关性奖励
        themes = literary_analysis.get('thematic_elements', {})
        if themes:
            # 检查主题与搜索关键词的匹配度
            search_theme_alignment = themes.get('search_theme_alignment', {})
            if search_theme_alignment:
                # 给予主题对齐奖励
                avg_theme_relevance = sum(search_theme_alignment.values()) / len(search_theme_alignment)
                literary_bonus += avg_theme_relevance * 0.3
            
            # 重要主题检测奖励
            important_themes = ['power_and_ambition', 'guilt_and_conscience', 'fate_and_destiny']
            theme_count = sum(1 for theme in important_themes if theme in themes)
            if theme_count > 0:
                literary_bonus += theme_count * 0.1
        
        # 语言模式复杂度奖励
        patterns = literary_analysis.get('linguistic_patterns', {})
        if patterns:
            # 修辞疑问和重复模式奖励
            if patterns.get('rhetorical_questions'):
                literary_bonus += 0.15
            if patterns.get('repetition_patterns'):
                literary_bonus += 0.1
            
            # 句法复杂度奖励
            syntactic_complexity = patterns.get('syntactic_complexity', {})
            if syntactic_complexity.get('complex_sentence_indicators'):
                literary_bonus += 0.12
        
        # 叙事结构质量奖励
        narrative = literary_analysis.get('narrative_structure', {})
        if narrative:
            # 对话质量奖励
            if narrative.get('has_dialogue'):
                dialogue_intensity = narrative.get('dialogue_intensity', 0)
                if dialogue_intensity > 0.1:  # 高对话密度
                    literary_bonus += 0.15
            
            # 叙事风格平衡奖励
            style = narrative.get('narrative_style', '')
            if style in ['action_oriented', 'descriptive_oriented']:
                literary_bonus += 0.08
        
        # 事实观察质量奖励
        observations = literary_analysis.get('factual_observations', [])
        if observations:
            # 基于观察的数量和类型给予奖励
            high_quality_obs = [obs for obs in observations if any(keyword in obs for keyword in ['密度', '强烈', '复杂', '主题'])]
            if high_quality_obs:
                literary_bonus += len(high_quality_obs) * 0.05
            
            # 特殊观察奖励
            for obs in observations:
                if '主题' in obs:
                    literary_bonus += 0.1
                if '复杂' in obs:
                    literary_bonus += 0.08
                if '强烈' in obs:
                    literary_bonus += 0.07
        
        return min(literary_bonus, 0.5)  # 限制最大奖励为0.5


def create_reranker(method="cross_encoder", keywords=None, **kwargs):
    """工厂函数：创建重排器，包括基础和增强版 - 优化版本"""
    
    # 高级上下文感知重排器（专为文学作品优化）
    if method == "advanced_context_aware":
        from .reranker_enhanced import create_enhanced_reranker
        return create_enhanced_reranker(method="advanced_context_aware", keywords=keywords, **kwargs)
    
    # 增强版重排器支持
    elif method == "enhanced_cross_encoder":
        from .reranker_enhanced import create_enhanced_reranker
        return create_enhanced_reranker(method=method, keywords=keywords, **kwargs)
    
    # 多模型ensemble重排器
    elif method == "ensemble":
        from .reranker_enhanced import create_enhanced_reranker
        return create_enhanced_reranker(method="ensemble", keywords=keywords, **kwargs)
    
    # 多样性优化重排器
    elif method == "diversity_optimizer":
        from .reranker_enhanced import create_enhanced_reranker
        return create_enhanced_reranker(method="diversity_optimizer", keywords=keywords, **kwargs)
    
    # 基础 Cross-Encoder 重排器（优化版本）
    elif method == "cross_encoder":
        return CrossEncoderReranker(keywords=keywords, **kwargs)
    
    # 简单重排器（优化版本）
    elif method == "simple":
        return SimpleReranker(keywords=keywords, **kwargs)
    
    # 默认使用高级上下文感知重排器（最适合文学作品检索）
    else:
        from .reranker_enhanced import create_enhanced_reranker
        return create_enhanced_reranker(method="advanced_context_aware", keywords=keywords, **kwargs)
