#!/usr/bin/env python3
"""
文学分析模块
提供真实的文学要素分析功能，包括：
1. 人物共现分析
2. 主题频率统计
3. 情感倾向分析
4. 场景分布分析
"""

import re
import nltk
from collections import Counter, defaultdict
from itertools import combinations
import pandas as pd
import numpy as np
from datetime import datetime
import os

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag


class LiteraryAnalyzer:
    """文学分析器"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
        # 预定义的人物名称模式（可扩展）
        self.character_patterns = [
            # 莎士比亚《麦克白》中的主要人物
            r'\b(Macbeth|Lady Macbeth|Duncan|Banquo|Malcolm|Macduff|Ross|Lennox|Angus|Menteith|Caithness)\b',
            # 《哈利波特》中的主要人物
            r'\b(Harry|Potter|Hermione|Granger|Ron|Weasley|Dumbledore|Snape|Voldemort|Hagrid|McGonagall)\b',
            # 通用人名模式
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # 姓名格式
            r'\b[A-Z][a-z]{2,}\b'  # 专有名词（可能是人名）
        ]
        
        # 预定义的主题关键词
        self.theme_keywords = {
            'power': ['power', 'authority', 'control', 'dominance', 'rule', 'command', 'throne', 'crown', 'king', 'queen'],
            'betrayal': ['betray', 'betrayal', 'treachery', 'deceive', 'deception', 'backstab', 'unfaithful'],
            'guilt': ['guilt', 'guilty', 'conscience', 'remorse', 'regret', 'shame', 'sin', 'forgive'],
            'death': ['death', 'die', 'kill', 'murder', 'slay', 'blood', 'corpse', 'grave', 'funeral'],
            'love': ['love', 'beloved', 'affection', 'romance', 'heart', 'dear', 'sweet', 'kiss'],
            'fear': ['fear', 'afraid', 'terror', 'horror', 'dread', 'panic', 'frighten', 'scared'],
            'ambition': ['ambition', 'ambitious', 'desire', 'want', 'aspire', 'dream', 'goal', 'achieve'],
            'revenge': ['revenge', 'vengeance', 'avenge', 'retaliate', 'payback', 'retribution'],
            'honor': ['honor', 'honour', 'noble', 'dignity', 'respect', 'virtue', 'glory'],
            'madness': ['mad', 'madness', 'insane', 'crazy', 'lunatic', 'mental', 'mind', 'sanity']
        }
        
        # 情感词汇
        self.emotion_keywords = {
            'positive': ['joy', 'happy', 'glad', 'pleased', 'delight', 'love', 'hope', 'peace', 'triumph'],
            'negative': ['sad', 'anger', 'hate', 'fear', 'despair', 'sorrow', 'pain', 'suffer', 'grief'],
            'neutral': ['think', 'consider', 'believe', 'seem', 'appear', 'perhaps', 'maybe']
        }
    
    def extract_characters(self, text_chunks):
        """提取文本中的人物名称 - 优化版本"""
        characters = Counter()
        character_positions = defaultdict(list)  # 记录每个人物出现的位置
        
        # 扩展停用词以过滤无关词汇
        extended_stop_words = self.stop_words.union({
            'the', 'and', 'his', 'her', 'him', 'she', 'he', 'they', 'them', 'their',
            'was', 'were', 'been', 'being', 'have', 'has', 'had', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'this', 'that',
            'these', 'those', 'who', 'what', 'where', 'when', 'why', 'how', 'which',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'just', 'now', 'even', 'also', 'however', 'although', 'because'
        })
        
        # 限制处理的文本块数量
        max_chunks = 100
        processed_chunks = text_chunks[:max_chunks] if len(text_chunks) > max_chunks else text_chunks
        
        for chunk_idx, chunk in enumerate(processed_chunks):
            try:
                content = chunk.get('content', '')
                chunk_id = chunk.get('id', f'chunk_{chunk_idx}')
                
                # 限制单个文本块的长度以提高性能
                if len(content) > 5000:  # 限制为5000字符
                    content = content[:5000]
                
                # 使用多种模式提取人物
                for pattern in self.character_patterns:
                    try:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in matches:
                            # 标准化人物名称
                            char_name = match.strip().title()
                            
                            # 更严格的过滤条件
                            if (len(char_name) > 2 and 
                                char_name.lower() not in extended_stop_words and
                                not char_name.lower() in ['act', 'scene', 'enter', 'exit', 'exeunt'] and
                                not re.match(r'^[IVX]+$', char_name) and  # 排除罗马数字
                                not char_name.isdigit() and  # 排除纯数字
                                any(c.isalpha() for c in char_name)):  # 必须包含字母
                                
                                characters[char_name] += 1
                                character_positions[char_name].append({
                                    'chunk_id': chunk_id,
                                    'chunk_index': chunk_idx,
                                    'page_num': chunk.get('page_num', 1)
                                })
                    except Exception as pattern_error:
                        print(f"⚠️ 模式匹配错误: {pattern_error}")
                        continue
                        
            except Exception as chunk_error:
                print(f"⚠️ 处理文本块 {chunk_idx} 时出错: {chunk_error}")
                continue
        
        return dict(characters), dict(character_positions)
    
    def analyze_character_cooccurrence(self, text_chunks, min_occurrences=2):
        """分析人物共现关系 - 优化版本"""
        try:
            # 限制处理的文本块数量以避免性能问题
            max_chunks = 100  # 限制最多处理100个文本块
            if len(text_chunks) > max_chunks:
                print(f"⚠️ 文本块数量过多({len(text_chunks)})，将只处理前{max_chunks}个以提高性能")
                text_chunks = text_chunks[:max_chunks]
            
            characters, character_positions = self.extract_characters(text_chunks)
            
            # 过滤出现频率低的人物
            frequent_characters = {name: count for name, count in characters.items() 
                                 if count >= min_occurrences}
            
            if len(frequent_characters) < 2:
                return {}, frequent_characters, {}
            
            # 只保留最常见的人物以提高性能
            max_characters = 10
            if len(frequent_characters) > max_characters:
                print(f"⚠️ 人物数量过多({len(frequent_characters)})，将只分析前{max_characters}个最常见人物")
                frequent_characters = dict(sorted(frequent_characters.items(), 
                                                key=lambda x: x[1], reverse=True)[:max_characters])
            
            # 分析共现关系
            cooccurrence_matrix = defaultdict(int)
            cooccurrence_details = defaultdict(list)
            
            for chunk_idx, chunk in enumerate(text_chunks):
                content = chunk.get('content', '').lower()
                chunk_characters = []
                
                # 找出在当前chunk中出现的人物
                for char_name in frequent_characters.keys():
                    if char_name.lower() in content:
                        chunk_characters.append(char_name)
                
                # 记录共现关系
                if len(chunk_characters) >= 2:
                    for char1, char2 in combinations(chunk_characters, 2):
                        # 确保一致的排序
                        if char1 > char2:
                            char1, char2 = char2, char1
                        
                        cooccurrence_matrix[(char1, char2)] += 1
                        # 限制详情记录数量
                        if len(cooccurrence_details[(char1, char2)]) < 5:  # 最多记录5个示例
                            cooccurrence_details[(char1, char2)].append({
                                'chunk_id': chunk.get('id', f'chunk_{chunk_idx}'),
                                'page_num': chunk.get('page_num', 1),
                                'content_preview': content[:100] + '...' if len(content) > 100 else content
                            })
            
            # 将tuple键转换为字符串键以便JSON序列化
            cooccurrence_dict = {f"{k[0]}-{k[1]}": v for k, v in cooccurrence_matrix.items()}
            cooccurrence_details_dict = {f"{k[0]}-{k[1]}": v for k, v in cooccurrence_details.items()}
            
            return cooccurrence_dict, frequent_characters, cooccurrence_details_dict
            
        except Exception as e:
            print(f"⚠️ 人物共现分析出错: {e}")
            return {}, {}, {}
    
    def analyze_themes(self, text_chunks):
        """分析主题频率"""
        theme_counts = defaultdict(int)
        theme_details = defaultdict(list)
        
        for chunk_idx, chunk in enumerate(text_chunks):
            content = chunk.get('content', '').lower()
            words = word_tokenize(content)
            
            for theme, keywords in self.theme_keywords.items():
                theme_score = 0
                found_keywords = []
                
                for keyword in keywords:
                    keyword_count = content.count(keyword.lower())
                    if keyword_count > 0:
                        theme_score += keyword_count
                        found_keywords.extend([keyword] * keyword_count)
                
                if theme_score > 0:
                    theme_counts[theme] += theme_score
                    theme_details[theme].append({
                        'chunk_id': chunk.get('id', f'chunk_{chunk_idx}'),
                        'page_num': chunk.get('page_num', 1),
                        'score': theme_score,
                        'keywords_found': found_keywords,
                        'content_preview': chunk.get('content', '')[:200] + '...' 
                            if len(chunk.get('content', '')) > 200 else chunk.get('content', '')
                    })
        
        return dict(theme_counts), dict(theme_details)
    
    def analyze_emotions(self, text_chunks):
        """分析情感倾向"""
        emotion_counts = defaultdict(int)
        emotion_details = defaultdict(list)
        
        for chunk_idx, chunk in enumerate(text_chunks):
            content = chunk.get('content', '').lower()
            
            for emotion, keywords in self.emotion_keywords.items():
                emotion_score = 0
                found_keywords = []
                
                for keyword in keywords:
                    keyword_count = content.count(keyword.lower())
                    if keyword_count > 0:
                        emotion_score += keyword_count
                        found_keywords.extend([keyword] * keyword_count)
                
                if emotion_score > 0:
                    emotion_counts[emotion] += emotion_score
                    emotion_details[emotion].append({
                        'chunk_id': chunk.get('id', f'chunk_{chunk_idx}'),
                        'page_num': chunk.get('page_num', 1),
                        'score': emotion_score,
                        'keywords_found': found_keywords
                    })
        
        return dict(emotion_counts), dict(emotion_details)
    
    def analyze_narrative_structure(self, text_chunks):
        """分析叙事结构（场景分布）"""
        # 识别场景标记
        scene_markers = [
            r'act\s+[ivx]+',
            r'scene\s+[ivx]+',
            r'chapter\s+\d+',
            r'part\s+[ivx]+',
            r'enter\s+\w+',
            r'exit\s+\w+'
        ]
        
        scenes = []
        current_scene = None
        scene_content_length = []
        
        for chunk_idx, chunk in enumerate(text_chunks):
            content = chunk.get('content', '').lower()
            
            # 检查是否包含场景标记
            for pattern in scene_markers:
                if re.search(pattern, content, re.IGNORECASE):
                    if current_scene is not None:
                        # 结束当前场景
                        scenes.append(current_scene)
                    
                    # 开始新场景
                    current_scene = {
                        'start_chunk': chunk_idx,
                        'start_page': chunk.get('page_num', 1),
                        'marker': re.search(pattern, content, re.IGNORECASE).group(),
                        'length': 0
                    }
                    break
            
            # 如果在场景中，累计长度
            if current_scene is not None:
                current_scene['length'] += len(chunk.get('content', ''))
        
        # 添加最后一个场景
        if current_scene is not None:
            scenes.append(current_scene)
        
        return scenes
    
    def generate_comprehensive_analysis(self, text_chunks, output_dir='outputs', min_occurrences=2, progress_callback=None):
        """生成综合文学分析报告"""
        def update_progress(msg):
            try:
                if progress_callback:
                    progress_callback(msg)
                else:
                    print(msg)
            except Exception as e:
                print(f"进度更新失败: {e}")
                print(msg)
        
        # 初始化结果字典
        analysis_results = {
            'characters': {'frequency': {}, 'cooccurrence': {}, 'cooccurrence_details': {}},
            'themes': {'frequency': {}, 'details': {}},
            'emotions': {'frequency': {}, 'details': {}},
            'narrative': {'scenes': []},
            'metadata': {
                'total_chunks': len(text_chunks),
                'analysis_time': datetime.now().isoformat(),
                'errors': []
            }
        }
        
        try:
            update_progress("🔍 开始文学分析...")
            
            # 1. 人物分析
            try:
                update_progress("📝 分析人物关系...")
                
                # 添加性能监控
                import time
                start_time = time.time()
                
                # 限制文本块数量以提高性能
                max_chunks_for_analysis = 50  # 最多分析50个文本块
                chunks_to_analyze = text_chunks[:max_chunks_for_analysis] if len(text_chunks) > max_chunks_for_analysis else text_chunks
                
                if len(text_chunks) > max_chunks_for_analysis:
                    update_progress(f"📊 为提高性能，将分析前{max_chunks_for_analysis}个文本块（共{len(text_chunks)}个）")
                
                cooccurrence, characters, cooccurrence_details = self.analyze_character_cooccurrence(chunks_to_analyze, min_occurrences=min_occurrences)
                
                elapsed_time = time.time() - start_time
                update_progress(f"✓ 人物分析完成 - 发现 {len(characters)} 个人物 (耗时: {elapsed_time:.2f}秒)")
                
                analysis_results['characters'] = {
                    'frequency': characters,
                    'cooccurrence': cooccurrence,
                    'cooccurrence_details': cooccurrence_details
                }
                
            except Exception as e:
                error_msg = f"人物分析失败: {str(e)}"
                analysis_results['metadata']['errors'].append(error_msg)
                update_progress(f"⚠️ {error_msg}")
            
            # 2. 主题分析
            try:
                update_progress("🎭 分析主题分布...")
                themes, theme_details = self.analyze_themes(text_chunks)
                analysis_results['themes'] = {
                    'frequency': themes,
                    'details': theme_details
                }
                update_progress(f"✓ 主题分析完成 - 发现 {len(themes)} 个主题")
            except Exception as e:
                error_msg = f"主题分析失败: {str(e)}"
                analysis_results['metadata']['errors'].append(error_msg)
                update_progress(f"⚠️ {error_msg}")
            
            # 3. 情感分析
            try:
                update_progress("😊 分析情感倾向...")
                emotions, emotion_details = self.analyze_emotions(text_chunks)
                analysis_results['emotions'] = {
                    'frequency': emotions,
                    'details': emotion_details
                }
                update_progress(f"✓ 情感分析完成 - 检测到 {len(emotions)} 种情感")
            except Exception as e:
                error_msg = f"情感分析失败: {str(e)}"
                analysis_results['metadata']['errors'].append(error_msg)
                update_progress(f"⚠️ {error_msg}")
            
            # 4. 叙事结构分析
            try:
                update_progress("📖 分析叙事结构...")
                scenes = self.analyze_narrative_structure(text_chunks)
                analysis_results['narrative']['scenes'] = scenes
                update_progress(f"✓ 叙事结构分析完成 - 发现 {len(scenes)} 个场景")
            except Exception as e:
                error_msg = f"叙事结构分析失败: {str(e)}"
                analysis_results['metadata']['errors'].append(error_msg)
                update_progress(f"⚠️ {error_msg}")
            
            # 5. 生成报告文件
            try:
                update_progress("💾 生成分析报告...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = os.path.join(output_dir, f'literary_analysis_{timestamp}.txt')
                
                report_content = self._generate_text_report(analysis_results)
                
                os.makedirs(output_dir, exist_ok=True)
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                update_progress(f"✅ 文学分析完成，报告已保存: {report_path}")
                
            except Exception as e:
                error_msg = f"报告生成失败: {str(e)}"
                analysis_results['metadata']['errors'].append(error_msg)
                update_progress(f"⚠️ {error_msg}")
        
        except Exception as e:
            error_msg = f"分析过程中发生严重错误: {str(e)}"
            analysis_results['metadata']['errors'].append(error_msg)
            update_progress(f"❌ {error_msg}")
        
        # 确保返回有效结果
        return analysis_results
    
    def _generate_text_report(self, analysis_results):
        """生成文本格式的分析报告"""
        report = []
        report.append("=" * 60)
        report.append("文学分析报告")
        report.append("=" * 60)
        report.append(f"分析时间: {analysis_results['metadata']['analysis_time']}")
        report.append(f"分析文本块数量: {analysis_results['metadata']['total_chunks']}")
        report.append("")
        
        # 人物分析
        report.append("📚 人物分析")
        report.append("-" * 30)
        characters = analysis_results['characters']['frequency']
        if characters:
            report.append("主要人物出现频率:")
            for char, count in sorted(characters.items(), key=lambda x: x[1], reverse=True)[:10]:
                report.append(f"  {char}: {count} 次")
            report.append("")
            
            cooccurrence = analysis_results['characters']['cooccurrence']
            if cooccurrence:
                report.append("人物共现关系 (前10对):")
                for char_pair, count in sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)[:10]:
                    char1, char2 = char_pair.split('-', 1)  # 拆分字符串键
                    report.append(f"  {char1} ↔ {char2}: 共同出现 {count} 次")
        else:
            report.append("  未检测到明显的人物名称")
        report.append("")
        
        # 主题分析
        report.append("🎭 主题分析")
        report.append("-" * 30)
        themes = analysis_results['themes']['frequency']
        if themes:
            report.append("主要主题频率:")
            for theme, count in sorted(themes.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {theme.title()}: {count} 次提及")
        else:
            report.append("  未检测到预定义主题")
        report.append("")
        
        # 情感分析
        report.append("😊 情感倾向分析")
        report.append("-" * 30)
        emotions = analysis_results['emotions']['frequency']
        if emotions:
            total_emotions = sum(emotions.values())
            for emotion, count in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_emotions) * 100 if total_emotions > 0 else 0
                report.append(f"  {emotion.title()}: {count} 次 ({percentage:.1f}%)")
        else:
            report.append("  未检测到明显的情感倾向")
        report.append("")
        
        # 叙事结构
        report.append("📖 叙事结构分析")
        report.append("-" * 30)
        scenes = analysis_results['narrative']['scenes']
        if scenes:
            report.append(f"检测到 {len(scenes)} 个场景/章节:")
            for i, scene in enumerate(scenes, 1):
                report.append(f"  {i}. {scene['marker']} (页面 {scene['start_page']}, 长度: {scene['length']} 字符)")
        else:
            report.append("  未检测到明显的场景分隔")
        
        return "\n".join(report)
    
    def create_character_network_data(self, cooccurrence, characters):
        """创建人物关系网络数据（用于可视化）"""
        nodes = []
        edges = []
        
        # 创建节点
        for char, freq in characters.items():
            nodes.append({
                'id': char,
                'label': char,
                'size': min(freq * 5, 50),  # 限制节点大小
                'frequency': freq
            })
        
        # 创建边
        for char_pair, weight in cooccurrence.items():
            char1, char2 = char_pair.split('-', 1)  # 拆分字符串键
            edges.append({
                'source': char1,
                'target': char2,
                'weight': weight,
                'label': f"{weight} 次共现"
            })
        
        return {'nodes': nodes, 'edges': edges}


# 创建全局分析器实例
literary_analyzer = LiteraryAnalyzer()
