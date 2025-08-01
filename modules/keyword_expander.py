#!/usr/bin/env python3
"""
关键词扩展模块 - 优化版
负责使用WordNet等工具扩展关键词，提高召回率
支持语义相似度过滤、词性标注、上下文感知扩展等功能
"""

import nltk
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# 导入我们改进的模型加载函数
try:
    from .retriever import load_sentence_transformer_model, SimpleVocabModel
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from modules.retriever import load_sentence_transformer_model, SimpleVocabModel
    except ImportError:
        # 最后的备用方案：定义简单的加载函数
        def load_sentence_transformer_model(model_path):
            try:
                return SentenceTransformer(model_path)
            except:
                return None
        
        class SimpleVocabModel:
            def encode(self, texts):
                import numpy as np
                if isinstance(texts, str):
                    texts = [texts]
                return np.random.random((len(texts), 384))  # 简单的随机向量


class KeywordExpander:
    """关键词扩展器 - 优化版"""
    
    def __init__(self, method='wordnet', semantic_model=None, document_type='literary'):
        self.method = method
        self.document_type = document_type
        self.semantic_model = semantic_model
        self.use_simple_model = False
        
        if method == 'wordnet':
            self._ensure_wordnet_data()
        
        # 初始化语义模型（如果可用）
        if semantic_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # 尝试使用改进的模型加载机制
                print("🔍 正在加载关键词扩展语义模型...")
                self.semantic_model = load_sentence_transformer_model('sentence-transformers/all-MiniLM-L6-v2')
                
                if self.semantic_model is None:
                    print("🔄 语义模型加载失败，使用简单词汇模型")
                    self.semantic_model = SimpleVocabModel()
                    self.use_simple_model = True
                else:
                    print("✓ 语义模型已加载，将使用语义相似度过滤")
                    
            except Exception as e:
                print(f"⚠️ 语义模型加载失败: {e}")
                print("🔄 使用简单词汇模型")
                self.semantic_model = SimpleVocabModel()
                self.use_simple_model = True
        
        # 领域特定词典（文学作品）
        self.literary_domain_expansions = {
            # 情感和心理状态
            'ambition': ['aspiration', 'ambitiousness', 'drive', 'determination', 'pursuit', 'yearning', 'hunger', 'thirst'],
            'guilt': ['shame', 'remorse', 'regret', 'conscience', 'culpability', 'self-reproach', 'penitence', 'contrition'],
            'jealousy': ['envy', 'resentment', 'suspicion', 'possessiveness', 'covetousness', 'spite', 'malice'],
            'love': ['affection', 'devotion', 'passion', 'romance', 'adoration', 'fondness', 'tenderness', 'ardor'],
            'fear': ['terror', 'dread', 'anxiety', 'trepidation', 'apprehension', 'horror', 'fright', 'alarm'],
            'courage': ['bravery', 'valor', 'heroism', 'boldness', 'fortitude', 'daring', 'gallantry', 'mettle'],
            'anger': ['wrath', 'fury', 'rage', 'ire', 'indignation', 'outrage', 'choler', 'spleen'],
            
            # 道德和哲学概念
            'evil': ['wickedness', 'malice', 'corruption', 'darkness', 'depravity', 'iniquity', 'sin', 'vice'],
            'good': ['virtue', 'righteousness', 'nobility', 'honor', 'integrity', 'purity', 'goodness', 'merit'],
            'justice': ['fairness', 'equity', 'righteousness', 'lawfulness', 'retribution', 'vindication'],
            'honor': ['dignity', 'integrity', 'nobility', 'respect', 'prestige', 'reputation', 'esteem', 'glory'],
            'betrayal': ['treachery', 'disloyalty', 'deception', 'treason', 'backstabbing', 'perfidy', 'duplicity'],
            
            # 权力和政治
            'power': ['authority', 'control', 'dominance', 'influence', 'might', 'sway', 'command', 'sovereignty'],
            'king': ['monarch', 'sovereign', 'ruler', 'emperor', 'majesty', 'crown', 'throne', 'royalty'],
            'throne': ['crown', 'scepter', 'sovereignty', 'kingship', 'dominion', 'rule', 'monarchy'],
            'crown': ['diadem', 'coronet', 'sovereignty', 'royalty', 'majesty', 'kingship'],
            
            # 命运和预言
            'destiny': ['fate', 'fortune', 'prophecy', 'predestination', 'doom', 'lot', 'kismet'],
            'prophecy': ['prediction', 'foretelling', 'divination', 'oracle', 'prognostication', 'revelation'],
            'fate': ['destiny', 'doom', 'fortune', 'lot', 'providence', 'kismet', 'predestination'],
            
            # 超自然和魔法
            'magic': ['sorcery', 'wizardry', 'enchantment', 'spellcraft', 'witchcraft', 'conjuring'],
            'ghost': ['spirit', 'specter', 'phantom', 'apparition', 'wraith', 'shade', 'banshee'],
            'witch': ['sorceress', 'enchantress', 'hag', 'crone', 'sibyl', 'pythoness'],
            
            # 生死主题
            'death': ['mortality', 'demise', 'passing', 'perishing', 'end', 'expiration', 'doom', 'grave'],
            'murder': ['killing', 'assassination', 'slaying', 'homicide', 'bloodshed', 'execution'],
            'blood': ['gore', 'crimson', 'scarlet', 'bloodshed', 'slaughter', 'carnage'],
            
            # 复仇和报应
            'revenge': ['vengeance', 'retribution', 'retaliation', 'payback', 'reprisal', 'vindication'],
            'punishment': ['retribution', 'penalty', 'chastisement', 'discipline', 'correction', 'nemesis'],
        }
    
    def _ensure_wordnet_data(self):
        """确保WordNet数据已下载"""
        try:
            from nltk.corpus import wordnet
            # 测试是否可以访问WordNet
            wordnet.synsets('test')
        except LookupError:
            print("正在下载WordNet数据...")
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
            except Exception as e:
                print(f"⚠️ WordNet数据下载失败: {e}")
    
    def expand_keywords(self, keywords, max_synonyms_per_word=3, max_related_per_word=2, 
                       semantic_threshold=0.6, use_hierarchical=True):
        """
        扩展关键词列表 - 优化版，支持语义过滤和层次化扩展
        
        Args:
            keywords: 原始关键词列表
            max_synonyms_per_word: 每个词最多添加的同义词数量
            max_related_per_word: 每个词最多添加的相关词数量
            semantic_threshold: 语义相似度阈值
            use_hierarchical: 是否使用层次化扩展
        
        Returns:
            扩展后的关键词字典，包含权重信息
        """
        try:
            from nltk.corpus import wordnet
            
            # 使用字典存储关键词及其权重
            expanded_keywords = {}
            
            # 原始关键词权重为1.0
            for keyword in keywords:
                expanded_keywords[keyword] = 1.0
            
            original_count = len(keywords)
            
            for keyword in keywords:
                print(f"  正在扩展关键词: '{keyword}'")
                
                # 1. 使用领域特定词典（高权重）
                domain_expansions = self._get_domain_expansions(keyword)
                for exp_word in domain_expansions:
                    if exp_word not in expanded_keywords:
                        expanded_keywords[exp_word] = 0.9
                
                # 2. 获取高质量同义词（中高权重）
                synonyms = self._get_semantic_filtered_synonyms(
                    keyword, max_synonyms_per_word, semantic_threshold
                )
                for syn in synonyms:
                    if syn not in expanded_keywords:
                        expanded_keywords[syn] = 0.8
                
                # 3. 获取相关词（中等权重）
                if use_hierarchical:
                    related_words = self._get_hierarchical_related_words(
                        keyword, max_related_per_word
                    )
                    for rel_word, weight in related_words.items():
                        if rel_word not in expanded_keywords:
                            expanded_keywords[rel_word] = weight * 0.6
                else:
                    related = self._get_related_words(keyword, max_related_per_word)
                    for rel in related:
                        if rel not in expanded_keywords:
                            expanded_keywords[rel] = 0.6
                
                # 4. 获取词形变化（较低权重）
                morphological_variants = self._get_morphological_variants(keyword)
                for variant in morphological_variants:
                    if variant not in expanded_keywords:
                        expanded_keywords[variant] = 0.7
                     # 过滤掉过于通用的词汇
            filtered_expanded = self._filter_generic_words(
                expanded_keywords, keywords
            )
            
            print(f"  原始关键词: {original_count} 个")
            print(f"  扩展后关键词: {len(filtered_expanded)} 个")
            print(f"  新增关键词: {len(filtered_expanded) - original_count} 个")
            
            # 显示扩展结果摘要
            self._print_expansion_summary(filtered_expanded, keywords)
            
            return filtered_expanded
            
        except ImportError:
            print("⚠️ NLTK WordNet未安装，跳过关键词扩展")
            return {kw: 1.0 for kw in keywords}
        except Exception as e:
            print(f"⚠️ 关键词扩展时发生未知错误: {e}")
            return {kw: 1.0 for kw in keywords}

    def _get_domain_expansions(self, keyword):
        """获取领域特定的词汇扩展"""
        keyword_lower = keyword.lower()
        
        # 检查是否在领域词典中
        if keyword_lower in self.literary_domain_expansions:
            return self.literary_domain_expansions[keyword_lower][:3]  # 限制数量
        
        # 如果没有找到，返回空列表
        return []

    def _get_semantic_filtered_synonyms(self, word, max_count, threshold):
        """获取语义过滤后的同义词"""
        synonyms = self._get_high_quality_synonyms(word, max_count * 2)  # 获取更多候选
        
        if not self.semantic_model or not synonyms:
            return synonyms[:max_count]
        
        try:
            # 使用语义模型过滤
            word_embedding = self.semantic_model.encode([word])
            synonym_embeddings = self.semantic_model.encode(synonyms)
            
            similarities = cosine_similarity(word_embedding, synonym_embeddings)[0]
            
            # 过滤低相似度的词汇
            filtered_synonyms = []
            for i, sim in enumerate(similarities):
                if sim >= threshold:
                    filtered_synonyms.append(synonyms[i])
            
            return filtered_synonyms[:max_count]
        
        except Exception as e:
            print(f"⚠️ 语义过滤失败: {e}")
            return synonyms[:max_count]

    def _get_hierarchical_related_words(self, word, max_count):
        """获取层次化相关词汇"""
        try:
            from nltk.corpus import wordnet
            related_words = {}
            
            synsets = wordnet.synsets(word)
            if not synsets:
                return related_words
            
            for synset in synsets[:2]:  # 限制同义词集数量
                # 上位词
                for hypernym in synset.hypernyms()[:1]:
                    for lemma in hypernym.lemmas()[:1]:
                        name = lemma.name().replace('_', ' ')
                        if name != word and len(name) > 2:
                            related_words[name] = 0.7
                
                # 下位词
                for hyponym in synset.hyponyms()[:1]:
                    for lemma in hyponym.lemmas()[:1]:
                        name = lemma.name().replace('_', ' ')
                        if name != word and len(name) > 2:
                            related_words[name] = 0.6
                
                # 同级词（通过共同上位词）
                for hypernym in synset.hypernyms()[:1]:
                    for sibling in hypernym.hyponyms()[:2]:
                        if sibling != synset:
                            for lemma in sibling.lemmas()[:1]:
                                name = lemma.name().replace('_', ' ')
                                if name != word and len(name) > 2:
                                    related_words[name] = 0.5
            
            # 按权重排序并限制数量
            sorted_related = sorted(related_words.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_related[:max_count])
        
        except Exception as e:
            print(f"⚠️ 层次化扩展失败: {e}")
            return {}

    def _filter_generic_words(self, expanded_keywords, original_keywords):
        """过滤掉过于通用的扩展词汇，支持权重信息"""
        # 过滤掉的通用词汇
        generic_words = {
            'desire', 'want', 'need', 'wish', 'hope', 'like', 'love', 'feel', 'think', 'believe',
            'try', 'attempt', 'effort', 'work', 'do', 'make', 'get', 'have', 'take', 'give',
            'see', 'look', 'find', 'show', 'tell', 'say', 'speak', 'talk', 'ask', 'answer',
            'come', 'go', 'move', 'turn', 'walk', 'run', 'stand', 'sit', 'live', 'stay',
            'happen', 'occur', 'appear', 'seem', 'become', 'remain', 'continue', 'start',
            'begin', 'end', 'stop', 'finish', 'complete', 'change', 'improve', 'develop',
            'create', 'build', 'form', 'produce', 'provide', 'offer', 'serve', 'help',
            'use', 'apply', 'employ', 'utilize', 'spend', 'pay', 'cost', 'buy', 'sell',
            'win', 'lose', 'gain', 'earn', 'achieve', 'reach', 'meet', 'join', 'leave',
            'open', 'close', 'cut', 'break', 'fix', 'repair', 'clean', 'wash', 'wear',
            'carry', 'hold', 'keep', 'put', 'place', 'set', 'lay', 'throw', 'catch',
            'play', 'enjoy', 'fun', 'game', 'sport', 'music', 'song', 'book', 'read',
            'write', 'draw', 'paint', 'color', 'picture', 'photo', 'image', 'watch',
            'listen', 'hear', 'sound', 'voice', 'word', 'language', 'speak', 'talk'
        }
        
        # Check if expanded_keywords is a dict (with weights) or a set/list
        if isinstance(expanded_keywords, dict):
            filtered = {}
            for keyword, weight in expanded_keywords.items():
                # 保留原始关键词
                if keyword in original_keywords:
                    filtered[keyword] = weight
                # 保留非通用词汇
                elif keyword not in generic_words:
                    # 额外检查：确保不是过于简单的词汇
                    if len(keyword) > 3 and not keyword.isdigit():
                        filtered[keyword] = weight
            return filtered
        else:
            # 保留原始关键词
            filtered = set(original_keywords)
            for keyword in expanded_keywords:
                # 保留原始关键词
                if keyword in original_keywords:
                    filtered.add(keyword)
                # 保留非通用词汇
                elif keyword not in generic_words:
                    # 额外检查：确保不是过于简单的词汇
                    if len(keyword) > 3 and not keyword.isdigit():
                        filtered.add(keyword)
            return filtered

    def _print_expansion_summary(self, expanded_keywords, original_keywords):
        """打印扩展结果摘要"""
        new_keywords = [k for k in expanded_keywords.keys() if k not in original_keywords]
        if new_keywords:
            print(f"  新增关键词: {new_keywords[:5]}")  # 只显示前5个

    def _get_morphological_variants(self, word):
        """获取词形变体（简化版）"""
        variants = set()
        
        # 基本的后缀变换
        suffixes_to_try = ['s', 'es', 'ed', 'ing', 'er', 'est', 'ly', 'ness', 'ment', 'tion', 'able', 'ful']
        suffixes_to_remove = ['s', 'es', 'ed', 'ing', 'er', 'est', 'ly', 'ness', 'ment', 'tion', 'able', 'ful']
        
        # 尝试添加后缀
        for suffix in suffixes_to_try:
            variants.add(word + suffix)
        
        # 尝试去除后缀
        for suffix in suffixes_to_remove:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                variants.add(word[:-len(suffix)])
        
        # 去除原词本身
        variants.discard(word)
        
        # 限制返回数量
        return list(variants)[:3]
    
    def _get_synonyms(self, word, max_count=5):
        """获取同义词"""
        try:
            from nltk.corpus import wordnet
            
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().lower().replace('_', ' ')
                    if len(synonym) > 2 and synonym != word:
                        synonyms.add(synonym)
                        if len(synonyms) >= max_count:
                            break
                if len(synonyms) >= max_count:
                    break
            
            return list(synonyms)
            
        except Exception:
            return []
    
    def _get_related_words(self, word, max_count=3):
        """获取相关词（上位词和下位词）"""
        try:
            from nltk.corpus import wordnet
            
            related = set()
            
            for syn in wordnet.synsets(word):
                # 上位词 (hypernyms)
                for hyper in syn.hypernyms():
                    for lemma in hyper.lemmas():
                        related_word = lemma.name().lower().replace('_', ' ')
                        if len(related_word) > 2 and related_word != word:
                            related.add(related_word)
                            if len(related) >= max_count:
                                break
                    if len(related) >= max_count:
                        break
                
                # 下位词 (hyponyms) - 限制数量避免过度扩展
                if len(related) < max_count:
                    for hypo in syn.hyponyms()[:2]:  # 只取前2个
                        for lemma in hypo.lemmas():
                            related_word = lemma.name().lower().replace('_', ' ')
                            if len(related_word) > 2 and related_word != word:
                                related.add(related_word)
                                if len(related) >= max_count:
                                    break
                        if len(related) >= max_count:
                            break
                
                if len(related) >= max_count:
                    break
            
            return list(related)
            
        except Exception:
            return []
    
    def _get_high_quality_synonyms(self, word, max_count=3):
        """获取高质量同义词，避免过于通用的词汇"""
        try:
            from nltk.corpus import wordnet
            
            synonyms = set()
            
            # 过滤掉的通用词汇
            generic_words = {'desire', 'want', 'need', 'like', 'love', 'get', 'have', 'make', 'do', 'go', 'come', 'take', 'give', 'see', 'know', 'think', 'feel', 'say', 'tell', 'find', 'use', 'work', 'try', 'ask', 'seem', 'turn', 'move', 'live', 'believe', 'hold', 'bring', 'happen', 'write', 'provide', 'sit', 'stand', 'lose', 'pay', 'meet', 'include', 'continue', 'set', 'learn', 'change', 'lead', 'understand', 'watch', 'follow', 'stop', 'create', 'speak', 'read', 'allow', 'add', 'spend', 'grow', 'open', 'walk', 'win', 'offer', 'remember', 'consider', 'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect', 'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain'}
            
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().lower().replace('_', ' ')
                    if (len(synonym) > 2 and 
                        synonym != word and 
                        synonym not in generic_words and
                        not synonym.endswith('ing') and  # 避免动名词
                        not synonym.endswith('ed')):     # 避免过去分词
                        synonyms.add(synonym)
                        if len(synonyms) >= max_count:
                            break
                if len(synonyms) >= max_count:
                    break
            
            return list(synonyms)
            
        except Exception:
            return []
    
    def process_keywords(self, keywords_input):
        """
        处理关键词输入，支持逗号分隔的字符串或列表
        
        Args:
            keywords_input: 关键词输入（字符串或列表）
        
        Returns:
            处理后的关键词列表
        """
        if isinstance(keywords_input, str):
            # 支持逗号分隔的关键词
            keywords = [k.strip().lower() for k in keywords_input.split(',')]
        elif isinstance(keywords_input, list):
            keywords = [k.strip().lower() for k in keywords_input]
        else:
            keywords = [str(keywords_input).strip().lower()]
        
        return [k for k in keywords if k]  # 过滤空关键词
