#!/usr/bin/env python3
"""
文本主题分析器模块
分析文本的主要主题、情感倾向、文学元素等
"""

import numpy as np
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
import json
import os


class LiteraryThemeAnalyzer:
    """文学作品主题分析器"""
    
    def __init__(self):
        # 主要文学主题词典
        self.theme_keywords = {
            'ambition': {
                'primary': ['ambition', 'ambitions', 'ambitious', 'aspiration', 'aspire', 'strive'],
                'secondary': ['power', 'throne', 'crown', 'rule', 'control', 'dominion', 'authority', 
                            'command', 'sovereignty', 'empire', 'kingdom', 'reign', 'master'],
                'contextual': ['climb', 'rise', 'achieve', 'obtain', 'gain', 'pursue', 'seek', 'reach']
            },
            'guilt': {
                'primary': ['guilt', 'guilty', 'conscience', 'remorse', 'shame', 'regret'],
                'secondary': ['sin', 'wrong', 'evil', 'wicked', 'corrupt', 'blame', 'fault', 'crime'],
                'contextual': ['repent', 'confess', 'forgive', 'punish', 'atone', 'sorry', 'ashamed']
            },
            'power': {
                'primary': ['power', 'powerful', 'authority', 'control', 'dominion', 'rule'],
                'secondary': ['king', 'queen', 'throne', 'crown', 'royal', 'noble', 'lord', 'master'],
                'contextual': ['command', 'order', 'obey', 'submit', 'govern', 'lead', 'conquer']
            },
            'love': {
                'primary': ['love', 'beloved', 'lover', 'passion', 'romantic', 'romance'],
                'secondary': ['heart', 'soul', 'dear', 'sweet', 'tender', 'gentle', 'kiss', 'embrace'],
                'contextual': ['devotion', 'affection', 'cherish', 'adore', 'treasure', 'faithful']
            },
            'death': {
                'primary': ['death', 'dead', 'die', 'dying', 'killed', 'murder', 'murdered'],
                'secondary': ['grave', 'tomb', 'corpse', 'ghost', 'spirit', 'soul', 'afterlife'],
                'contextual': ['mortal', 'mortality', 'fatal', 'doom', 'fate', 'destiny', 'end']
            },
            'betrayal': {
                'primary': ['betray', 'betrayal', 'treachery', 'traitor', 'deceive', 'deceit'],
                'secondary': ['lie', 'false', 'unfaithful', 'disloyal', 'cheat', 'trick'],
                'contextual': ['trust', 'faith', 'loyalty', 'honest', 'truth', 'promise', 'oath']
            },
            'fear': {
                'primary': ['fear', 'afraid', 'terror', 'dread', 'horror', 'panic', 'frighten'],
                'secondary': ['scary', 'frightening', 'terrifying', 'horrible', 'dreadful', 'awful'],
                'contextual': ['anxiety', 'worry', 'concern', 'nervous', 'tremble', 'shake', 'flee']
            },
            'madness': {
                'primary': ['mad', 'madness', 'insane', 'insanity', 'crazy', 'lunacy', 'deranged'],
                'secondary': ['mental', 'mind', 'brain', 'thoughts', 'reason', 'sanity', 'rational'],
                'contextual': ['confusion', 'delusion', 'hallucination', 'obsession', 'mania']
            },
            'fate': {
                'primary': ['fate', 'destiny', 'fortune', 'doom', 'predetermined', 'inevitable'],
                'secondary': ['future', 'prophecy', 'foretell', 'predict', 'omen', 'portent'],
                'contextual': ['chance', 'luck', 'fortune', 'providence', 'divine', 'gods']
            },
            'honor': {
                'primary': ['honor', 'honour', 'honorable', 'noble', 'dignity', 'integrity'],
                'secondary': ['respect', 'reputation', 'glory', 'fame', 'renown', 'prestige'],
                'contextual': ['virtue', 'moral', 'ethics', 'principle', 'value', 'character']
            }
        }
        
        # 情感强度词汇
        self.emotion_intensity = {
            'high': ['overwhelming', 'consuming', 'burning', 'fierce', 'intense', 'powerful',
                    'desperate', 'passionate', 'furious', 'terrifying', 'devastating'],
            'medium': ['strong', 'deep', 'significant', 'notable', 'considerable', 'substantial'],
            'low': ['mild', 'slight', 'gentle', 'soft', 'quiet', 'calm', 'peaceful']
        }
        
        # 文学设备词汇
        self.literary_devices = {
            'metaphor': ['like', 'as', 'metaphor', 'symbol', 'represent', 'signify'],
            'irony': ['irony', 'ironic', 'paradox', 'contradictory', 'opposite'],
            'foreshadowing': ['hint', 'suggest', 'forebode', 'omen', 'portent', 'sign'],
            'imagery': ['see', 'sight', 'vision', 'image', 'picture', 'appear', 'look']
        }
        
        # 角色类型识别
        self.character_types = {
            'protagonist': ['hero', 'main', 'protagonist', 'central', 'lead'],
            'antagonist': ['villain', 'enemy', 'opponent', 'antagonist', 'evil'],
            'tragic_hero': ['tragic', 'flawed', 'downfall', 'hubris', 'pride'],
            'innocent': ['innocent', 'pure', 'naive', 'young', 'child']
        }
    
    def analyze_text_themes(self, text_chunks: List[Dict], progress_callback=None) -> Dict:
        """分析文本的主要主题"""
        def update_progress(msg):
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)
                
        update_progress("🔍 开始文本主题分析...")
        
        # 合并所有文本内容
        full_text = ' '.join([chunk.get('content', '') for chunk in text_chunks]).lower()
        
        # 主题分析
        update_progress("  分析主题分布...")
        theme_scores = self._analyze_themes(full_text)
        
        # 情感分析
        update_progress("  分析情感倾向...")
        emotion_analysis = self._analyze_emotions(full_text)
        
        # 文学设备分析
        update_progress("  分析文学手法...")
        literary_analysis = self._analyze_literary_devices(full_text)
        
        # 角色分析
        update_progress("  分析角色特征...")
        character_analysis = self._analyze_characters(text_chunks)
        
        # 结构分析
        update_progress("  分析文本结构...")
        structure_analysis = self._analyze_text_structure(text_chunks)
        
        # 生成主题摘要
        update_progress("  生成主题摘要...")
        theme_summary = self._generate_theme_summary(theme_scores, emotion_analysis)
        
        analysis_result = {
            'themes': theme_scores,
            'emotions': emotion_analysis,
            'literary_devices': literary_analysis,
            'characters': character_analysis,
            'structure': structure_analysis,
            'summary': theme_summary,
            'text_statistics': {
                'total_chunks': len(text_chunks),
                'total_words': len(full_text.split()),
                'unique_words': len(set(full_text.split())),
                'average_chunk_length': np.mean([len(chunk.get('content', '').split()) for chunk in text_chunks])
            }
        }
        
        update_progress(f"✅ 主题分析完成，识别出 {len([t for t, s in theme_scores.items() if s > 0.1])} 个主要主题")
        
        return analysis_result
    
    def _analyze_themes(self, text: str) -> Dict[str, float]:
        """分析文本中的主题强度 - 修复版本"""
        theme_scores = {}
        words = text.split()
        total_words = len(words)
        
        if total_words == 0:
            return {theme: 0.0 for theme in self.theme_keywords.keys()}
        
        # 首先计算所有主题的原始分数
        raw_scores = {}
        for theme, keywords in self.theme_keywords.items():
            score = 0.0
            
            # 计算主要关键词分数
            primary_count = sum(text.count(word) for word in keywords['primary'])
            score += primary_count * 3.0  # 增加主要关键词权重
            
            # 计算次要关键词分数
            secondary_count = sum(text.count(word) for word in keywords['secondary'])
            score += secondary_count * 1.5  # 增加次要关键词权重
            
            # 计算上下文关键词分数
            contextual_count = sum(text.count(word) for word in keywords['contextual'])
            score += contextual_count * 0.8
            
            raw_scores[theme] = score
        
        # 计算总分数并进行相对归一化
        total_raw_score = sum(raw_scores.values())
        
        if total_raw_score > 0:
            # 使用相对归一化：每个主题的分数是其在总分中的比例
            for theme, raw_score in raw_scores.items():
                if raw_score > 0:
                    # 计算相对强度
                    relative_strength = raw_score / total_raw_score
                    # 应用平滑函数避免过度集中
                    smoothed_score = relative_strength ** 0.7  # 平方根平滑
                    theme_scores[theme] = min(smoothed_score, 1.0)
                else:
                    theme_scores[theme] = 0.0
        else:
            # 如果没有匹配的主题关键词，所有主题得分为0
            theme_scores = {theme: 0.0 for theme in self.theme_keywords.keys()}
        
        # 重新归一化确保分数合理分布
        max_score = max(theme_scores.values()) if theme_scores.values() else 0
        if max_score > 0:
            normalization_factor = 1.0 / max_score
            for theme in theme_scores:
                theme_scores[theme] = min(theme_scores[theme] * normalization_factor, 1.0)
        
        return theme_scores
    
    def _analyze_emotions(self, text: str) -> Dict:
        """分析文本的情感强度和类型"""
        emotion_analysis = {
            'intensity_distribution': {},
            'dominant_emotions': [],
            'emotional_complexity': 0.0
        }
        
        # 分析情感强度分布
        for intensity, words in self.emotion_intensity.items():
            count = sum(text.count(word) for word in words)
            emotion_analysis['intensity_distribution'][intensity] = count
        
        # 确定主导情感
        total_intensity = sum(emotion_analysis['intensity_distribution'].values())
        if total_intensity > 0:
            for intensity, count in emotion_analysis['intensity_distribution'].items():
                if count / total_intensity > 0.3:
                    emotion_analysis['dominant_emotions'].append(intensity)
        
        # 计算情感复杂度
        non_zero_intensities = sum(1 for count in emotion_analysis['intensity_distribution'].values() if count > 0)
        emotion_analysis['emotional_complexity'] = non_zero_intensities / len(self.emotion_intensity)
        
        return emotion_analysis
    
    def _analyze_literary_devices(self, text: str) -> Dict[str, int]:
        """分析文学设备的使用"""
        device_counts = {}
        
        for device, indicators in self.literary_devices.items():
            count = sum(text.count(indicator) for indicator in indicators)
            device_counts[device] = count
        
        return device_counts
    
    def _analyze_characters(self, text_chunks: List[Dict]) -> Dict:
        """分析角色类型和特征"""
        character_analysis = {
            'character_types': {},
            'character_mentions': {},
            'dialogue_density': 0.0
        }
        
        # 合并文本
        full_text = ' '.join([chunk.get('content', '') for chunk in text_chunks]).lower()
        
        # 分析角色类型
        for char_type, indicators in self.character_types.items():
            count = sum(full_text.count(indicator) for indicator in indicators)
            character_analysis['character_types'][char_type] = count
        
        # 分析对话密度
        total_chunks = len(text_chunks)
        dialogue_chunks = sum(1 for chunk in text_chunks 
                             if '"' in chunk.get('content', '') or "'" in chunk.get('content', ''))
        
        if total_chunks > 0:
            character_analysis['dialogue_density'] = dialogue_chunks / total_chunks
        
        # 提取可能的角色名称（大写词汇，排除常见词）
        common_words = {'THE', 'AND', 'OR', 'BUT', 'IF', 'THEN', 'WHEN', 'WHERE', 'WHY', 'HOW', 
                       'WHAT', 'WHO', 'WHICH', 'THAT', 'THIS', 'THESE', 'THOSE', 'I', 'YOU', 
                       'HE', 'SHE', 'IT', 'WE', 'THEY', 'MY', 'YOUR', 'HIS', 'HER', 'ITS', 
                       'OUR', 'THEIR', 'ME', 'HIM', 'HER', 'US', 'THEM'}
        
        possible_names = []
        for chunk in text_chunks:
            words = chunk.get('content', '').split()
            for word in words:
                clean_word = re.sub(r'[^\w]', '', word)
                if (clean_word.isupper() and len(clean_word) > 2 and 
                    clean_word not in common_words):
                    possible_names.append(clean_word)
        
        name_counts = Counter(possible_names)
        # 只保留出现多次的名称
        character_analysis['character_mentions'] = {
            name: count for name, count in name_counts.items() if count >= 3
        }
        
        return character_analysis
    
    def _analyze_text_structure(self, text_chunks: List[Dict]) -> Dict:
        """分析文本结构特征"""
        structure_analysis = {
            'chunk_length_distribution': {},
            'content_density_by_page': {},
            'thematic_progression': {}
        }
        
        # 分析文本块长度分布
        lengths = [len(chunk.get('content', '').split()) for chunk in text_chunks]
        if lengths:
            structure_analysis['chunk_length_distribution'] = {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': min(lengths),
                'max': max(lengths),
                'median': np.median(lengths)
            }
        
        # 按页面分析内容密度
        page_content = defaultdict(list)
        for chunk in text_chunks:
            page_num = chunk.get('page_num', 1)
            content_length = len(chunk.get('content', '').split())
            page_content[page_num].append(content_length)
        
        for page, lengths in page_content.items():
            structure_analysis['content_density_by_page'][page] = {
                'total_words': sum(lengths),
                'chunks_count': len(lengths),
                'avg_chunk_length': np.mean(lengths) if lengths else 0
            }
        
        return structure_analysis
    
    def _generate_theme_summary(self, theme_scores: Dict[str, float], 
                               emotion_analysis: Dict) -> Dict[str, str]:
        """生成主题分析摘要"""
        # 找出主要主题（分数 > 0.1）
        major_themes = [(theme, score) for theme, score in theme_scores.items() if score > 0.1]
        major_themes.sort(key=lambda x: x[1], reverse=True)
        
        # 找出次要主题（分数 0.05-0.1）
        minor_themes = [(theme, score) for theme, score in theme_scores.items() 
                       if 0.05 <= score <= 0.1]
        minor_themes.sort(key=lambda x: x[1], reverse=True)
        
        # 生成摘要
        summary = {
            'primary_themes': [theme for theme, _ in major_themes[:3]],
            'secondary_themes': [theme for theme, _ in minor_themes[:3]],
            'theme_complexity': len(major_themes) + len(minor_themes),
            'dominant_emotional_tone': emotion_analysis.get('dominant_emotions', []),
            'thematic_richness': 'high' if len(major_themes) >= 4 else 
                               'medium' if len(major_themes) >= 2 else 'low'
        }
        
        return summary
    
    def get_theme_based_keywords(self, analysis_result: Dict) -> List[str]:
        """基于主题分析结果生成扩展关键词"""
        expanded_keywords = []
        
        # 基于主要主题添加关键词
        primary_themes = analysis_result['summary']['primary_themes']
        
        for theme in primary_themes:
            if theme in self.theme_keywords:
                # 添加主要和次要关键词
                expanded_keywords.extend(self.theme_keywords[theme]['primary'][:3])
                expanded_keywords.extend(self.theme_keywords[theme]['secondary'][:5])
        
        # 基于角色分析添加关键词
        char_mentions = analysis_result['characters']['character_mentions']
        # 添加最常提到的角色名
        top_characters = sorted(char_mentions.items(), key=lambda x: x[1], reverse=True)[:3]
        expanded_keywords.extend([name.lower() for name, _ in top_characters])
        
        # 去重并返回
        return list(set(expanded_keywords))
    
    def save_analysis_result(self, analysis_result: Dict, output_path: str):
        """保存分析结果到文件"""
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 转换numpy类型为Python原生类型以便JSON序列化
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_for_json(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            json_compatible_result = convert_for_json(analysis_result)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_compatible_result, f, indent=2, ensure_ascii=False)
            
            print(f"📁 主题分析结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"❌ 保存分析结果失败: {e}")
    
    def print_analysis_summary(self, analysis_result: Dict):
        """打印分析结果摘要"""
        print("\n" + "="*60)
        print("📚 文本主题分析摘要")
        print("="*60)
        
        # 基本统计
        stats = analysis_result['text_statistics']
        print(f"📊 文本统计:")
        print(f"  总文本块数: {stats['total_chunks']}")
        print(f"  总词数: {stats['total_words']}")
        print(f"  平均块长度: {stats['average_chunk_length']:.1f} 词")
        
        # 主要主题
        summary = analysis_result['summary']
        print(f"\n🎭 主要主题:")
        for i, theme in enumerate(summary['primary_themes'], 1):
            score = analysis_result['themes'][theme]
            print(f"  {i}. {theme.title()} (强度: {score:.3f})")
        
        # 次要主题
        if summary['secondary_themes']:
            print(f"\n📝 次要主题:")
            for theme in summary['secondary_themes']:
                score = analysis_result['themes'][theme]
                print(f"  - {theme.title()} (强度: {score:.3f})")
        
        # 情感分析
        emotions = analysis_result['emotions']
        print(f"\n💭 情感特征:")
        print(f"  主导情感强度: {', '.join(emotions['dominant_emotions']) if emotions['dominant_emotions'] else '平衡'}")
        print(f"  情感复杂度: {emotions['emotional_complexity']:.2f}")
        
        # 角色分析
        characters = analysis_result['characters']
        if characters['character_mentions']:
            print(f"\n👥 主要角色:")
            for name, count in list(characters['character_mentions'].items())[:5]:
                print(f"  - {name}: {count} 次提及")
        
        print(f"  对话密度: {characters['dialogue_density']:.2%}")
        
        # 文学特征
        print(f"\n✨ 主题丰富度: {summary['thematic_richness'].upper()}")
        
        print("="*60)


def create_theme_analyzer() -> LiteraryThemeAnalyzer:
    """创建主题分析器实例"""
    return LiteraryThemeAnalyzer()
