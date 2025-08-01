#!/usr/bin/env python3
"""
可视化模块 - 增强版
支持页码分布、主题热力图等高级可视化功能
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datetime import datetime
import os

# 尝试导入seaborn，如果没有就跳过
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
# 尝试导入词云库
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


class AdvancedVisualizer:
    """高级可视化器 - 支持页码分布和主题分析"""
    
    def __init__(self, output_dir='outputs', language='zh'):
        self.output_dir = output_dir
        self.language = language  # 'zh' for Chinese, 'en' for English
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 设置字体以支持中文显示
        self._setup_chinese_fonts()
        
        # 初始化标签字典
        self._init_labels()
            
        if SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
    
    def _setup_chinese_fonts(self):
        """设置中文字体，解决方块字问题"""
        import matplotlib.font_manager as fm
        import platform
        
        # 根据操作系统设置中文字体
        if platform.system() == 'Darwin':  # macOS
            chinese_fonts = [
                'PingFang SC', 'Pingfang SC', 'PingFang SC Regular',
                'Songti SC', 'STSong', 'STHeiti', 'STXihei', 
                'Arial Unicode MS', 'Helvetica Neue'
            ]
        elif platform.system() == 'Windows':
            chinese_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'NSimSun']
        else:  # Linux
            chinese_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans CN', 'DejaVu Sans']
        
        # 查找系统中可用的中文字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        print(f"DEBUG: 系统可用字体总数: {len(available_fonts)}")
        
        # 筛选出可能的中文字体
        chinese_available = []
        for font in available_fonts:
            font_lower = font.lower()
            if any(keyword in font_lower for keyword in ['pingfang', 'song', 'hei', 'kai', 'fang', 'unicode', 'cjk', 'han']):
                chinese_available.append(font)
        
        print(f"DEBUG: 发现可能的中文字体: {chinese_available[:10]}")  # 只显示前10个
        
        # 选择第一个可用的中文字体
        selected_font = None
        for font in chinese_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        # 如果没找到精确匹配，尝试模糊匹配
        if not selected_font and chinese_available:
            # 优先选择PingFang类字体（macOS最佳中文显示）
            for font in chinese_available:
                if 'pingfang' in font.lower() or 'ping fang' in font.lower():
                    selected_font = font
                    break
            
            # 如果还没找到，选择第一个可用的中文字体
            if not selected_font:
                selected_font = chinese_available[0]
        
        # 设置字体参数 - 强制使用Unicode字体确保中文显示
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei', 'Microsoft YaHei']
        if selected_font:
            plt.rcParams['font.sans-serif'].insert(0, selected_font)
        
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10  # 略小一点避免重叠
        
        # 设置图表的默认参数以避免重叠
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['figure.autolayout'] = True
        plt.rcParams['axes.labelpad'] = 8
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['axes.titlepad'] = 15
        
        # 强制清除字体缓存，确保新设置生效
        import matplotlib.font_manager
        try:
            matplotlib.font_manager.fontManager.__init__()
        except Exception:
            try:
                import matplotlib
                if hasattr(matplotlib.font_manager, '_get_fontconfig_fonts'):
                    matplotlib.font_manager._get_fontconfig_fonts.cache_clear()
            except Exception:
                print("⚠️ 字体缓存刷新失败，但不影响功能")
        
        # 保存选择的字体供词云使用
        self.chinese_font = selected_font if selected_font else 'Arial Unicode MS'
        print(f"✅ 设置中文字体: {self.chinese_font}")
        
        # 测试中文显示
        try:
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试中文', ha='center', va='center', fontfamily=self.chinese_font)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.close(fig)
            print("✅ 中文字体测试通过")
        except Exception as e:
            print(f"⚠️ 中文字体测试失败: {e}")
    
    def _init_labels(self):
        """初始化中英文标签字典"""
        self.labels = {
            'zh': {
                # 通用标签
                'page_num': 'PDF页码',
                'frequency': '频率',
                'similarity_score': '相似度分数',
                'page_range': '页码',
                'character_count': '出现次数',
                'character_name': '人物名称',
                'theme_type': '主题类型',
                'emotion_type': '情感类型',
                'scene_length': '场景长度',
                'cooccurrence_count': '共现次数',
                'keyword_theme': '关键词/主题',
                'occurrence_frequency': '出现频率',
                
                # 图表标题
                'page_distribution': 'PDF页面中相关内容分布',
                'keyword_frequency': '关键词频率分布',
                'similarity_distribution': '相似度分数分布',
                'page_similarity_relation': '相关性分数与页码关系',
                'page_density_heatmap': '页面范围内容密度热力图',
                'keyword_cooccurrence': '关键词共现热力图',
                'theme_frequency_distribution': '关键词/主题频率分布',
                'wordcloud_title': '关键词云图',
                'character_frequency_title': '主要人物出现频率',
                'theme_distribution_title': '主要主题分布',
                'emotion_analysis_title': '情感倾向分析',
                'character_cooccurrence_title': '人物共现关系',
                'content_distribution_title': '场景在各页面的分布',
                'summary_stats_title': '关键统计信息',
                'literary_analysis_title': '文学分析综合报告',
                
                # 统计信息
                'total_passages': '总段落数',
                'pages_involved': '涉及页数',
                'max_page': '最大页码',
                'avg_per_page': '平均每页段落数',
                'avg_value': '平均值',
                'trend_line': '趋势线',
                'analysis_summary': '分析摘要',
                'characters_detected': '检测到人物',
                'main_character': '主要人物',
                'active_themes': '活跃主题',
                'dominant_theme': '主导主题',
                'emotion_total': '情感表达总计',
                'dominant_emotion': '主导情感',
                'scenes_detected': '检测场景',
                'longest_scene': '最长场景',
                'other': '其他',
                
                # 提示信息
                'no_data': '没有结果数据\n无法生成页码分布图',
                'no_valid_pages': '没有有效的页码信息\n无法生成页码分布图',
                'no_keywords_found': '未找到关键词',
                'no_similarity_scores': '无相似度分数',
                'no_page_score_data': '无页码/分数数据',
                'no_keyword_data': '无关键词数据\n无法生成共现热力图',
                'insufficient_keywords': '关键词数量不足\n需要至少2个关键词',
                'no_character_info': '未检测到人物信息',
                'no_theme_info': '未检测到主题信息',
                'no_emotion_info': '未检测到情感信息',
                'no_character_relations': '未检测到人物关系',
                'no_page_distribution': '未检测到页面分布信息',
                'no_stats_data': '暂无统计数据'
            },
            'en': {
                # 通用标签
                'page_num': 'PDF Page',
                'frequency': 'Frequency',
                'similarity_score': 'Similarity Score',
                'page_range': 'Page Number',
                'character_count': 'Occurrences',
                'character_name': 'Character Name',
                'theme_type': 'Theme Type',
                'emotion_type': 'Emotion Type',
                'scene_length': 'Scene Length',
                'cooccurrence_count': 'Co-occurrence Count',
                'keyword_theme': 'Keywords/Themes',
                'occurrence_frequency': 'Occurrence Frequency',
                
                # 图表标题
                'page_distribution': 'Content Distribution Across PDF Pages',
                'keyword_frequency': 'Keyword Frequency Distribution',
                'similarity_distribution': 'Similarity Score Distribution',
                'page_similarity_relation': 'Similarity Score vs Page Number',
                'page_density_heatmap': 'Page Range Content Density Heatmap',
                'keyword_cooccurrence': 'Keyword Co-occurrence Heatmap',
                'theme_frequency_distribution': 'Keywords/Themes Frequency Distribution',
                'wordcloud_title': 'Keywords Word Cloud',
                'character_frequency_title': 'Main Characters Frequency',
                'theme_distribution_title': 'Main Themes Distribution',
                'emotion_analysis_title': 'Emotion Analysis',
                'character_cooccurrence_title': 'Character Co-occurrence',
                'content_distribution_title': 'Scene Distribution Across Pages',
                'summary_stats_title': 'Key Statistics',
                'literary_analysis_title': 'Literary Analysis Report',
                
                # 统计信息
                'total_passages': 'Total Passages',
                'pages_involved': 'Pages Involved',
                'max_page': 'Max Page',
                'avg_per_page': 'Avg Passages per Page',
                'avg_value': 'Average',
                'trend_line': 'Trend Line',
                'analysis_summary': 'Analysis Summary',
                'characters_detected': 'Characters Detected',
                'main_character': 'Main Character',
                'active_themes': 'Active Themes',
                'dominant_theme': 'Dominant Theme',
                'emotion_total': 'Total Emotions',
                'dominant_emotion': 'Dominant Emotion',
                'scenes_detected': 'Scenes Detected',
                'longest_scene': 'Longest Scene',
                'other': '',  # 英文中不需要量词
                
                # 提示信息
                'no_data': 'No result data\nCannot generate page distribution chart',
                'no_valid_pages': 'No valid page information\nCannot generate page distribution chart',
                'no_keywords_found': 'No keywords found',
                'no_similarity_scores': 'No similarity scores',
                'no_page_score_data': 'No page/score data',
                'no_keyword_data': 'No keyword data\nCannot generate co-occurrence heatmap',
                'insufficient_keywords': 'Insufficient keywords\nNeed at least 2 keywords',
                'no_character_info': 'No character information detected',
                'no_theme_info': 'No theme information detected',
                'no_emotion_info': 'No emotion information detected',
                'no_character_relations': 'No character relations detected',
                'no_page_distribution': 'No page distribution information detected',
                'no_stats_data': 'No statistical data available'
            }
        }
    
    def get_label(self, key):
        """获取当前语言的标签"""
        return self.labels[self.language].get(key, key)
    
    def set_language(self, language):
        """设置图表语言"""
        if language in ['zh', 'en']:
            self.language = language
        else:
            print(f"⚠️ 不支持的语言: {language}，保持当前语言: {self.language}")
    
    def get_font_family(self):
        """根据语言获取合适的字体"""
        if self.language == 'zh':
            return getattr(self, 'chinese_font', 'Arial Unicode MS')
        else:
            return 'Arial'
    
    def _get_wordcloud_font_path(self):
        """获取词云专用的字体路径"""
        import matplotlib.font_manager as fm
        import platform
        
        # 如果已经有检测到的中文字体，尝试找到对应的字体文件
        if hasattr(self, 'chinese_font') and self.chinese_font:
            try:
                # 通过字体管理器查找字体文件路径
                font_files = fm.findSystemFonts()
                font_props = fm.FontProperties(family=self.chinese_font)
                
                # 尝试通过名称匹配找到字体文件
                for font_file in font_files:
                    try:
                        font_prop = fm.FontProperties(fname=font_file)
                        if font_prop.get_name() == self.chinese_font:
                            print(f"🎨 找到词云字体文件: {font_file}")
                            return font_file
                    except:
                        continue
                        
            except Exception as e:
                print(f"⚠️ 动态字体路径查找失败: {e}")
        
        # 备用方案：预设路径
        import platform
        
        if platform.system() == 'Darwin':  # macOS
            possible_paths = [
                '/System/Library/Fonts/Arial Unicode MS.ttf',
                '/System/Library/Fonts/Supplemental/Arial Unicode MS.ttf',
                '/Library/Fonts/Arial Unicode MS.ttf',
                '/System/Library/Fonts/STHeiti Light.ttc',
                '/System/Library/Fonts/STHeiti Medium.ttc',
                '/System/Library/Fonts/PingFang.ttc',
                '/System/Library/Fonts/Helvetica.ttc',
            ]
        elif platform.system() == 'Windows':
            possible_paths = [
                'C:/Windows/Fonts/simhei.ttf',
                'C:/Windows/Fonts/msyh.ttc',
                'C:/Windows/Fonts/simsun.ttc',
                'C:/Windows/Fonts/arial.ttf',
            ]
        else:  # Linux
            possible_paths = [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                '/usr/share/fonts/TTF/DejaVuSans.ttf',
            ]
        
        # 检查哪个路径存在
        for path in possible_paths:
            if os.path.exists(path):
                print(f"🎨 使用词云字体路径: {path}")
                return path
        
        print("⚠️ 未找到合适的词云字体，使用默认字体")
        return None
    
    def plot_page_distribution(self, results, keywords, save_path=None):
        """绘制页码分布图"""
        if not results:
            print("没有结果数据，跳过页码分布图")
            # 返回一个提示图
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, self.get_label('no_data'), 
                   ha='center', va='center', fontsize=16,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   fontfamily=self.get_font_family())
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.tight_layout()
            return fig
        
        # 提取页码信息
        page_numbers = []
        for result in results:
            page_num = result.get('page_num', 1)
            if isinstance(page_num, (int, float)) and page_num > 0:
                page_numbers.append(int(page_num))
        
        if not page_numbers:
            print("没有有效的页码信息，跳过页码分布图")
            # 返回一个提示图
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, self.get_label('no_valid_pages'), 
                   ha='center', va='center', fontsize=16,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   fontfamily=self.get_font_family())
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.tight_layout()
            return fig
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        font_family = self.get_font_family()
        
        # 1. 页码直方图
        page_counts = Counter(page_numbers)
        pages = sorted(page_counts.keys())
        counts = [page_counts[p] for p in pages]
        
        ax1.bar(pages, counts, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.set_xlabel(self.get_label('page_num'), fontfamily=font_family)
        ax1.set_ylabel(self.get_label('frequency'), fontfamily=font_family)
        
        # 构建标题
        if self.language == 'zh':
            title = f"{self.get_label('page_distribution')}\n关键词: {', '.join(keywords)}"
        else:
            title = f"{self.get_label('page_distribution')}\nKeywords: {', '.join(keywords)}"
        ax1.set_title(title, fontfamily=font_family)
        ax1.grid(True, alpha=0.3)
        
        # 设置轴标签字体
        for label in ax1.get_xticklabels():
            label.set_fontfamily(font_family)
        for label in ax1.get_yticklabels():
            label.set_fontfamily(font_family)
        
        # 添加统计信息
        total_passages = len(page_numbers)
        unique_pages = len(set(page_numbers))
        max_page = max(page_numbers)
        
        if self.language == 'zh':
            stats_text = f'{self.get_label("total_passages")}: {total_passages}\n{self.get_label("pages_involved")}: {unique_pages}\n{self.get_label("max_page")}: {max_page}'
        else:
            stats_text = f'{self.get_label("total_passages")}: {total_passages}\n{self.get_label("pages_involved")}: {unique_pages}\n{self.get_label("max_page")}: {max_page}'
        
        ax1.text(0.02, 0.98, stats_text, 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontfamily=font_family)
        
        # 2. 页码密度热力图
        # 将页码按区间分组，显示内容密度
        max_page = max(page_numbers)
        page_ranges = []
        densities = []
        
        # 分成10个区间
        range_size = max(1, max_page // 10)
        for i in range(0, max_page, range_size):
            start_page = i + 1
            end_page = min(i + range_size, max_page)
            range_name = f"{start_page}-{end_page}"
            
            # 计算该区间的内容密度
            count_in_range = sum(1 for p in page_numbers if start_page <= p <= end_page)
            density = count_in_range / range_size if range_size > 0 else 0
            
            page_ranges.append(range_name)
            densities.append(density)
        
        # 绘制热力图
        density_matrix = np.array(densities).reshape(1, -1)
        im = ax2.imshow(density_matrix, cmap='YlOrRd', aspect='auto')
        ax2.set_xticks(range(len(page_ranges)))
        ax2.set_xticklabels(page_ranges, rotation=45, fontfamily=font_family)
        ax2.set_yticks([])
        ax2.set_title(self.get_label('page_density_heatmap'), fontfamily=font_family)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.1)
        cbar.set_label(self.get_label('avg_per_page'), fontfamily=font_family)
        
        plt.tight_layout()
        
        # 保存图表并返回Figure对象，方便在Streamlit中展示
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f'page_distribution_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # 返回图形对象
        return fig
        return fig
    
    def plot_theme_analysis(self, results, keywords, save_path=None):
        """绘制主题分析图"""
        if not results:
            print("没有结果数据，跳过主题分析图")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        font_family = self.get_font_family()
        
        # 1. 关键词频率分布
        all_keywords = []
        for result in results:
            found_keywords = result.get('found_keywords', [])
            all_keywords.extend(found_keywords)
        
        if all_keywords:
            keyword_counts = Counter(all_keywords)
            keywords_list = list(keyword_counts.keys())
            counts = list(keyword_counts.values())
            
            ax1.barh(keywords_list, counts, color='lightcoral')
            ax1.set_xlabel(self.get_label('frequency'), fontfamily=font_family)
            ax1.set_title(self.get_label('keyword_frequency'), fontfamily=font_family)
            ax1.grid(True, alpha=0.3)
            
            # 设置轴标签字体
            for label in ax1.get_xticklabels():
                label.set_fontfamily(font_family)
            for label in ax1.get_yticklabels():
                label.set_fontfamily(font_family)
        else:
            ax1.text(0.5, 0.5, self.get_label('no_keywords_found'), ha='center', va='center', transform=ax1.transAxes, fontfamily=font_family)
            ax1.set_title(self.get_label('keyword_frequency'), fontfamily=font_family)
        
        # 2. 相关性分数分布
        similarity_scores = []
        for result in results:
            score = result.get('similarity_score') or result.get('hybrid_score') or result.get('bm25_score', 0)
            if score > 0:
                similarity_scores.append(score)
        
        if similarity_scores:
            ax2.hist(similarity_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
            ax2.set_xlabel(self.get_label('similarity_score'), fontfamily=font_family)
            ax2.set_ylabel(self.get_label('frequency'), fontfamily=font_family)
            ax2.set_title(self.get_label('similarity_distribution'), fontfamily=font_family)
            ax2.grid(True, alpha=0.3)
            
            # 设置轴标签字体
            for label in ax2.get_xticklabels():
                label.set_fontfamily(font_family)
            for label in ax2.get_yticklabels():
                label.set_fontfamily(font_family)
            
            # 添加统计线
            mean_score = np.mean(similarity_scores)
            avg_label = f'{self.get_label("avg_value")}: {mean_score:.3f}'
            ax2.axvline(mean_score, color='red', linestyle='--', label=avg_label)
            legend = ax2.legend()
            legend.get_texts()[0].set_fontfamily(font_family)
        else:
            ax2.text(0.5, 0.5, self.get_label('no_similarity_scores'), ha='center', va='center', transform=ax2.transAxes, fontfamily=font_family)
            ax2.set_title(self.get_label('similarity_distribution'), fontfamily=font_family)
        
        # 3. 段落长度分布 - 已移除
        ax3.axis('off')
        if self.language == 'zh':
            ax3.set_title('段落长度分布已移除', fontfamily=font_family)
        else:
            ax3.set_title('Paragraph Length Distribution Removed', fontfamily=font_family)
        
        # 4. 页码vs相关性散点图
        pages = []
        scores = []
        for result in results:
            page_num = result.get('page_num', 1)
            score = result.get('similarity_score') or result.get('hybrid_score') or result.get('bm25_score', 0)
            if page_num > 0 and score > 0:
                pages.append(page_num)
                scores.append(score)
        
        if pages and scores:
            ax4.scatter(pages, scores, alpha=0.6, color='purple')
            ax4.set_xlabel(self.get_label('page_range'), fontfamily=font_family)
            ax4.set_ylabel(self.get_label('similarity_score'), fontfamily=font_family)
            ax4.set_title(self.get_label('page_similarity_relation'), fontfamily=font_family)
            ax4.grid(True, alpha=0.3)
            
            # 设置轴标签字体
            for label in ax4.get_xticklabels():
                label.set_fontfamily(font_family)
            for label in ax4.get_yticklabels():
                label.set_fontfamily(font_family)
            
            # 添加趋势线
            if len(pages) > 1:
                z = np.polyfit(pages, scores, 1)
                p = np.poly1d(z)
                ax4.plot(pages, p(pages), "r--", alpha=0.8, label=self.get_label('trend_line'))
                legend = ax4.legend()
                legend.get_texts()[0].set_fontfamily(font_family)
        else:
            ax4.text(0.5, 0.5, self.get_label('no_page_score_data'), ha='center', va='center', transform=ax4.transAxes, fontfamily=font_family)
            ax4.set_title(self.get_label('page_similarity_relation'), fontfamily=font_family)
        
        plt.tight_layout()
        
        # 保存图表并返回Figure对象，方便在Streamlit中展示
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f'theme_analysis_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def plot_word_cloud(self, results, keywords, max_words=200):
        """绘制关键词词云图"""
        if not WORDCLOUD_AVAILABLE:
            print("词云库未安装，跳过词云图")
            # 返回一个提示图
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, '词云库未安装\n请安装 wordcloud 包', 
                   ha='center', va='center', fontsize=16,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                   fontfamily=self.get_font_family())
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.tight_layout()
            return fig
            
        # 收集所有匹配关键词和文本内容
        all_text_words = []
        all_keywords = []
        
        # 从结果中提取更多词汇
        for r in results:
            # 获取found_keywords
            found_kw = r.get('found_keywords', [])
            if found_kw:
                all_keywords.extend(found_kw)
            
            # 从文本内容中提取更多词汇
            text_content = r.get('text', r.get('content', ''))
            if text_content:
                # 简单的词汇提取 - 支持中英文
                import re
                # 提取英文单词
                english_words = re.findall(r'\b[a-zA-Z]{3,}\b', text_content.lower())
                # 提取中文词汇（2个字符以上）
                chinese_words = re.findall(r'[\u4e00-\u9fff]{2,}', text_content)
                
                # 过滤常见停用词
                stop_words = {
                    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                    'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 
                    'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those',
                    'his', 'her', 'him', 'she', 'was', 'were', 'been', 'have', 'has', 'had',
                    'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must',
                    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '也', 
                    '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那'
                }
                
                # 过滤英文停用词
                filtered_english = [w for w in english_words if w not in stop_words and len(w) > 3]
                # 过滤中文停用词
                filtered_chinese = [w for w in chinese_words if w not in stop_words and len(w) >= 2]
                
                all_text_words.extend(filtered_english + filtered_chinese)
        
        # 合并关键词和文本词汇
        all_words = all_keywords + all_text_words
        
        # 如果没有找到词汇，使用原始搜索关键词
        if not all_words and keywords:
            all_words = keywords * 5  # 重复关键词以增加权重
            
        if not all_words:
            print("无关键词数据，跳过词云图")
            # 返回一个提示图
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, self.get_label('no_keywords_found') + '\n无法生成词云图', 
                   ha='center', va='center', fontsize=16,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   fontfamily=self.get_font_family())
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.tight_layout()
            return fig
            
        # 创建词频统计
        word_freq = Counter(all_words)
        
        # 确保至少有一些词汇
        if len(word_freq) < 5:
            # 如果词汇太少，添加一些通用词汇
            if self.language == 'zh':
                common_words = ['分析', '搜索', '文本', '内容', '关键词', '主题', '情感', '人物', '故事', '叙述']
            else:
                common_words = ['analysis', 'text', 'search', 'literature', 'character', 'theme', 'emotion', 'story', 'narrative', 'content']
            
            for word in common_words:
                if word not in word_freq:
                    word_freq[word] = 1
        
        # 生成词云文本 - 根据频率重复词汇
        text_for_wordcloud = []
        for word, count in word_freq.most_common(max_words):
            # 根据频率重复单词，但限制最大重复次数以保持平衡
            repeat_count = min(count, 10)  # 最多重复10次
            text_for_wordcloud.extend([word] * repeat_count)
        
        text = ' '.join(text_for_wordcloud)
        
        # 如果文本仍然太短，补充内容
        if len(text.split()) < 20:
            text += ' ' + ' '.join(keywords * 3) if keywords else ''
        
        try:
            # 使用动态检测到的中文字体
            font_path = self._get_wordcloud_font_path()
            
            # 创建词云配置
            wordcloud_config = {
                'width': 800, 
                'height': 400, 
                'max_words': max_words,
                'background_color': 'white', 
                'collocations': False,
                'relative_scaling': 0.5,
                'min_font_size': 12,
                'max_font_size': 80,
                'colormap': 'viridis',
                'prefer_horizontal': 0.7,
                'min_word_length': 2
            }
            
            # 只有在找到字体文件时才设置font_path
            if font_path:
                wordcloud_config['font_path'] = font_path
            
            wc = WordCloud(**wordcloud_config)
            
            # 生成词云
            if text.strip():
                wc.generate(text)
            else:
                # 如果没有文本，使用关键词创建基本词云
                fallback_text = ' '.join(keywords * 3) if keywords else 'text analysis search'
                wc.generate(fallback_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(self.get_label('wordcloud_title'), fontfamily=self.get_font_family(), pad=20)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"词云生成失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回错误提示图
            fig, ax = plt.subplots(figsize=(10, 5))
            error_msg = f'词云生成失败\n{str(e)[:100]}...' if len(str(e)) > 100 else f'词云生成失败\n{str(e)}'
            ax.text(0.5, 0.5, error_msg, 
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                   fontfamily=self.get_font_family())
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.tight_layout()
            return fig
    
    def plot_cooccurrence_heatmap(self, results, keywords, top_n=10):
        """绘制关键词共现热力图 - 改进版"""
        print(f"DEBUG: 开始生成共现热力图，输入结果数: {len(results) if results else 0}, 关键词: {keywords}")
        
        try:
            # 收集共现计数
            from itertools import combinations
            co_counts = Counter()
            
            # 先尝试从found_keywords获取
            all_keywords = []
            for r in results:
                found_kw = r.get('found_keywords', [])
                if found_kw:
                    all_keywords.extend(found_kw)
            
            print(f"DEBUG: 从found_keywords获取到 {len(all_keywords)} 个关键词")
            
            # 如果没有found_keywords，使用原始搜索关键词
            if not all_keywords and keywords:
                # 创建基于文本内容的关键词匹配
                for r in results:
                    text_content = r.get('text', '').lower()
                    for keyword in keywords:
                        if keyword.lower() in text_content:
                            count = text_content.count(keyword.lower())
                            all_keywords.extend([keyword] * count)
                print(f"DEBUG: 基于文本匹配获取到 {len(all_keywords)} 个关键词")
            
            # 最后的回退：如果还是没有数据，创建示例数据
            if not all_keywords:
                if keywords and len(keywords) >= 2:
                    # 使用搜索关键词创建基本共现
                    all_keywords = keywords * 3  # 给每个关键词一些权重
                    print(f"DEBUG: 使用搜索关键词创建示例数据: {len(all_keywords)} 个")
                else:
                    print('DEBUG: 无法生成共现热力图 - 缺少关键词数据')
                    return self._create_placeholder_heatmap("无关键词数据\n无法生成共现热力图")
            
            # 计算共现
            for r in results:
                kws = r.get('found_keywords', [])
                if not kws and keywords:
                    # 如果没有found_keywords，基于文本内容匹配
                    text_content = r.get('text', '').lower()
                    kws = [k for k in keywords if k.lower() in text_content]
                
                unique_kws = list(set(kws))
                if len(unique_kws) >= 2:
                    for a, b in combinations(sorted(unique_kws), 2):
                        co_counts[(a, b)] += 1
            
            print(f"DEBUG: 计算得到 {len(co_counts)} 个共现对")
            
            # 选取最常见关键词
            keyword_counts = Counter(all_keywords)
            top_keywords = [kw for kw, _ in keyword_counts.most_common(min(top_n, len(keyword_counts)))]
            
            print(f"DEBUG: 选择了 {len(top_keywords)} 个顶级关键词: {top_keywords}")
            
            if len(top_keywords) < 2:
                print(f'DEBUG: 关键词数量不足 ({len(top_keywords)})，创建占位图')
                return self._create_placeholder_heatmap(f"关键词数量不足 ({len(top_keywords)})\n需要至少2个关键词")
            
            # 构造矩阵
            size = len(top_keywords)
            matrix = []
            for x in top_keywords:
                row = []
                for y in top_keywords:
                    if x == y:
                        # 对角线显示该关键词的频率
                        row.append(keyword_counts.get(x, 0))
                    else:
                        # 查找共现次数
                        cooccur_count = co_counts.get((x, y), co_counts.get((y, x), 0))
                        row.append(cooccur_count)
                matrix.append(row)
            
            print(f"DEBUG: 构建了 {size}x{size} 的共现矩阵")
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(8, 6))
            font_family = self.get_font_family()
            
            try:
                if SEABORN_AVAILABLE:
                    print("DEBUG: 使用seaborn生成热力图")
                    import seaborn as sns
                    sns.heatmap(matrix, xticklabels=top_keywords, yticklabels=top_keywords,
                                cmap='YlGnBu', annot=True, fmt='d', ax=ax, cbar=True)
                else:
                    print("DEBUG: 使用matplotlib生成热力图")
                    import numpy as np
                    im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto')
                    
                    # 设置刻度和标签
                    ax.set_xticks(range(size))
                    ax.set_yticks(range(size))
                    ax.set_xticklabels(top_keywords, rotation=45, ha='right')
                    ax.set_yticklabels(top_keywords)
                    
                    # 添加数值标注
                    for i in range(size):
                        for j in range(size):
                            text = ax.text(j, i, matrix[i][j], ha="center", va="center", color="black")
                    
                    # 添加颜色条
                    plt.colorbar(im, ax=ax)
                
                ax.set_title(self.get_label('keyword_cooccurrence'), fontfamily=font_family, pad=20)
                
                # 设置轴标签字体
                for label in ax.get_xticklabels():
                    label.set_fontfamily(font_family)
                for label in ax.get_yticklabels():
                    label.set_fontfamily(font_family)
                
                plt.tight_layout()
                print("DEBUG: 共现热力图生成成功")
                
                # 确保图形正确显示
                fig.canvas.draw()  # 强制绘制
                return fig
                
            except Exception as plot_error:
                print(f"DEBUG: 绘图失败: {plot_error}")
                # 创建简单的柱状图作为替代
                return self._create_alternative_cooccurrence_chart(top_keywords, keyword_counts, font_family)
                
        except Exception as e:
            print(f"DEBUG: 共现热力图生成失败: {e}")
            import traceback
            traceback.print_exc()
            return self._create_placeholder_heatmap(f"生成失败: {str(e)}")
    
    def _create_placeholder_heatmap(self, message):
        """创建占位符热力图"""
        fig, ax = plt.subplots(figsize=(8, 6))
        font_family = self.get_font_family()
        ax.text(0.5, 0.5, message, 
               ha='center', va='center', fontsize=14,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
               fontfamily=font_family)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(self.get_label('keyword_cooccurrence'), fontfamily=font_family)
        plt.tight_layout()
        return fig
    
    def _create_alternative_cooccurrence_chart(self, keywords, keyword_counts, font_family):
        """创建替代的关键词频率图"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        counts = [keyword_counts.get(kw, 0) for kw in keywords]
        colors = plt.cm.viridis(np.linspace(0, 1, len(keywords)))
        
        bars = ax.bar(keywords, counts, color=colors, alpha=0.7)
        ax.set_title(f"{self.get_label('keyword_cooccurrence')} (简化版)", fontfamily=font_family)
        ax.set_xlabel(self.get_label('keyword_theme'), fontfamily=font_family)
        ax.set_ylabel(self.get_label('occurrence_frequency'), fontfamily=font_family)
        
        # 旋转标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 添加数值标注
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontfamily=font_family)
        
        # 设置轴标签字体
        for label in ax.get_xticklabels():
            label.set_fontfamily(font_family)
        for label in ax.get_yticklabels():
            label.set_fontfamily(font_family)
        
        plt.tight_layout()
        
        # 确保图形正确显示
        try:
            fig.canvas.draw()
        except Exception as draw_error:
            print(f"⚠️ 替代图形绘制警告: {draw_error}")
        
        return fig

    def plot_theme_frequency(self, results, keywords, top_n=10):
        """绘制主题/关键词出现频率柱状图"""
        # 收集所有关键词和重要词汇
        all_keywords = []
        keyword_counts = Counter()
        
        # 1. 先尝试从found_keywords获取
        for r in results:
            found_kw = r.get('found_keywords', [])
            if found_kw:
                all_keywords.extend(found_kw)
        
        # 2. 如果没有found_keywords，从文本内容中提取重要词汇
        if not all_keywords:
            import re
            for r in results:
                text_content = r.get('text', r.get('content', ''))
                if text_content:
                    # 提取英文单词（3个字母以上）
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', text_content.lower())
                    # 过滤停用词
                    stop_words = {
                        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                        'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 
                        'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those',
                        'his', 'her', 'him', 'she', 'was', 'were', 'been', 'have', 'has', 'had',
                        'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must',
                        'said', 'say', 'says', 'one', 'two', 'now', 'out', 'who', 'get', 'use',
                        'man', 'new', 'way', 'day', 'time', 'year', 'work', 'part', 'take',
                        'place', 'make', 'end', 'first', 'last', 'good', 'great', 'old', 'own'
                    }
                    filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
                    all_keywords.extend(filtered_words)
        
        # 3. 如果还是没有，使用原始搜索关键词并统计频率
        if not all_keywords and keywords:
            for keyword in keywords:
                count = 0
                for r in results:
                    text = r.get('text', r.get('content', '')).lower()
                    if keyword.lower() in text:
                        count += text.lower().count(keyword.lower())
                if count > 0:
                    keyword_counts[keyword] = count
                    all_keywords.extend([keyword] * count)
        
        # 统计词频
        if all_keywords:
            keyword_counts = Counter(all_keywords)
        
        # 4. 如果仍然没有数据，创建一些示例数据
        if not keyword_counts:
            print('无关键词数据，使用示例数据')
            # 使用搜索关键词作为示例
            if keywords:
                for i, kw in enumerate(keywords[:top_n]):
                    keyword_counts[kw] = len(keywords) - i
            else:
                # 创建文学主题示例
                sample_themes = ['love', 'death', 'power', 'betrayal', 'honor', 'fear', 'ambition']
                for i, theme in enumerate(sample_themes):
                    keyword_counts[theme] = len(sample_themes) - i
        
        if not keyword_counts:
            print('无关键词数据，跳过主题频率图')
            # 返回一个提示图
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, '无关键词数据\n无法生成频率图', 
                   ha='center', va='center', fontsize=16,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.tight_layout()
            return fig
            
        # 获取最常见的词汇
        most = keyword_counts.most_common(min(top_n, len(keyword_counts)))
        if not most:
            print('无有效关键词数据，跳过主题频率图')
            # 返回一个提示图
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, '无有效关键词数据\n无法生成频率图', 
                   ha='center', va='center', fontsize=16,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.tight_layout()
            return fig
            
        labels, counts = zip(*most)
        fig, ax = plt.subplots(figsize=(10, 6))
        font_family = self.get_font_family()
        
        # 使用渐变色
        colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
        bars = ax.bar(labels, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel(self.get_label('keyword_theme'), fontfamily=self.get_font_family())
        ax.set_ylabel(self.get_label('occurrence_frequency'), fontfamily=self.get_font_family())
        ax.set_title(self.get_label('theme_frequency_distribution'), fontfamily=self.get_font_family())
        
        # 设置轴标签字体
        font_family = self.get_font_family()
        for label in ax.get_xticklabels():
            label.set_fontfamily(font_family)
        for label in ax.get_yticklabels():
            label.set_fontfamily(font_family)
        
        # 旋转标签以避免重叠
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 在柱子上显示数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold',
                   fontfamily=font_family)
        
        # 添加网格
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        return fig
    
    def plot_comprehensive_analysis(self, results, keywords, save_path=None):
        """绘制综合分析图表"""
        if not results:
            print("没有结果数据，跳过综合分析图")
            return
        
        # 分别绘制页码分布和主题分析
        page_dist_path = self.plot_page_distribution(results, keywords)
        theme_analysis_path = self.plot_theme_analysis(results, keywords)
        
        return {
            'page_distribution': page_dist_path,
            'theme_analysis': theme_analysis_path
        }
    
    def generate_summary_report(self, results, keywords, pdf_info=None):
        """生成文本摘要报告"""
        if not results:
            return "没有结果数据可分析。"
        
        # 基本统计
        total_results = len(results)
        unique_pages = len(set(r.get('page_num', 1) for r in results))
        
        # 关键词统计
        all_keywords = []
        for result in results:
            found_keywords = result.get('found_keywords', [])
            all_keywords.extend(found_keywords)
        
        keyword_counts = Counter(all_keywords)
        
        # 相关性统计
        scores = []
        for result in results:
            score = result.get('similarity_score') or result.get('hybrid_score') or result.get('bm25_score', 0)
            if score > 0:
                scores.append(score)
        
        # 页码统计
        page_numbers = [r.get('page_num', 1) for r in results if r.get('page_num', 1) > 0]
        
        # 生成报告
        report = f"""
=== PDF 内容分析摘要报告 ===
分析时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
搜索关键词: {", ".join(keywords)}

📊 基本统计:
- 找到相关段落: {total_results} 个
- 涉及页面数量: {unique_pages} 页
- 平均每页相关段落: {total_results / unique_pages:.1f} 个

🔍 关键词分析:
"""
        
        if keyword_counts:
            for keyword, count in keyword_counts.most_common(5):
                report += f"- '{keyword}': 出现 {count} 次\n"
        else:
            report += "- 未检测到关键词匹配\n"
        
        if scores:
            report += f"""
📈 相关性分析:
- 最高相关性: {max(scores):.3f}
- 平均相关性: {np.mean(scores):.3f}
- 相关性标准差: {np.std(scores):.3f}
"""
        
        if page_numbers:
            report += f"""
📄 页码分布:
- 最早出现页码: 第 {min(page_numbers)} 页
- 最晚出现页码: 第 {max(page_numbers)} 页
- 内容集中度: {len(set(page_numbers)) / (max(page_numbers) - min(page_numbers) + 1):.2%}
"""
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f'analysis_summary_{timestamp}.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ 分析摘要报告已保存: {report_path}")
        
        return report

    def plot_literary_analysis(self, analysis_result, save_path=None):
        """为文学分析结果生成综合可视化图表"""
        if not analysis_result:
            print("没有文学分析数据，跳过可视化")
            return None
            
        # 创建一个大的图表包含多个子图
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 人物出现频率图 (左上)
        ax1 = plt.subplot(2, 3, 1)
        self._plot_character_frequency(analysis_result, ax1)
        
        # 2. 主题分布饼图 (右上)
        ax2 = plt.subplot(2, 3, 2)
        self._plot_theme_distribution(analysis_result, ax2)
        
        # 3. 情感倾向柱状图 (左中)
        ax3 = plt.subplot(2, 3, 3)
        self._plot_emotion_analysis(analysis_result, ax3)
        
        # 4. 人物共现网络图 (左下)
        ax4 = plt.subplot(2, 3, 4)
        self._plot_character_cooccurrence(analysis_result, ax4)
        
        # 5. 内容在页面中的分布 (右中)
        ax5 = plt.subplot(2, 3, 5)
        self._plot_content_distribution(analysis_result, ax5)
        
        # 6. 关键统计信息 (右下)
        ax6 = plt.subplot(2, 3, 6)
        self._plot_summary_stats(analysis_result, ax6)
        
        # 设置总标题字体
        chinese_font = getattr(self, 'chinese_font', 'Arial Unicode MS')
        plt.suptitle('文学分析综合报告', fontsize=16, y=0.95, 
                    fontfamily=chinese_font)
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f'literary_analysis_{timestamp}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 文学分析可视化已保存: {save_path}")
        
        return fig
    
    def _plot_character_frequency(self, analysis_result, ax):
        """绘制人物出现频率图"""
        # 从正确的数据结构中提取人物数据
        characters_data = analysis_result.get('characters', {})
        characters = characters_data.get('frequency', {}) if isinstance(characters_data, dict) else characters_data
        
        chinese_font = getattr(self, 'chinese_font', 'Arial Unicode MS')
        
        if not characters:
            ax.text(0.5, 0.5, '未检测到人物信息', ha='center', va='center', transform=ax.transAxes,
                   fontfamily=chinese_font)
            ax.set_title('人物出现频率', fontfamily=chinese_font)
            return
            
        # 取前10个最常出现的人物
        top_characters = dict(sorted(characters.items(), key=lambda x: x[1], reverse=True)[:10])
        
        names = list(top_characters.keys())
        counts = list(top_characters.values())
        
        bars = ax.bar(names, counts, color='lightblue', edgecolor='navy', alpha=0.7)
        ax.set_title('主要人物出现频率', fontfamily=chinese_font, fontsize=12)
        ax.set_xlabel('人物名称', fontfamily=chinese_font)
        ax.set_ylabel('出现次数', fontfamily=chinese_font)
        
        # 设置轴标签字体
        for label in ax.get_xticklabels():
            label.set_fontfamily(chinese_font)
        for label in ax.get_yticklabels():
            label.set_fontfamily(chinese_font)
        
        # 旋转x轴标签以避免重叠
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 在柱子上显示数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', 
                   fontfamily=chinese_font, fontsize=9)
    
    def _plot_theme_distribution(self, analysis_result, ax):
        """绘制主题分布饼图"""
        # 从正确的数据结构中提取主题数据
        themes_data = analysis_result.get('themes', {})
        themes = themes_data.get('frequency', {}) if isinstance(themes_data, dict) else themes_data
        
        if not themes:
            ax.text(0.5, 0.5, '未检测到主题信息', ha='center', va='center', transform=ax.transAxes,
                   fontfamily=getattr(self, 'chinese_font', 'Arial Unicode MS'))
            ax.set_title('主题分布', fontfamily=getattr(self, 'chinese_font', 'Arial Unicode MS'))
            return
            
        # 过滤掉计数为0的主题
        filtered_themes = {k: v for k, v in themes.items() if v > 0}
        
        if not filtered_themes:
            ax.text(0.5, 0.5, '未检测到有效主题', ha='center', va='center', transform=ax.transAxes,
                   fontfamily=getattr(self, 'chinese_font', 'Arial Unicode MS'))
            ax.set_title('主题分布', fontfamily=getattr(self, 'chinese_font', 'Arial Unicode MS'))
            return
        
        # 合并小比例项目以避免标签重叠
        total = sum(filtered_themes.values())
        main_themes = {}
        small_themes_count = 0
        
        for theme, count in filtered_themes.items():
            percentage = count / total * 100
            if percentage >= 5.0:  # 只显示占比大于5%的主题
                main_themes[theme] = count
            else:
                small_themes_count += count
        
        # 如果有小主题，合并为"其他"
        if small_themes_count > 0:
            main_themes['其他'] = small_themes_count
            
        labels = list(main_themes.keys())
        sizes = list(main_themes.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        # 使用更好的饼图配置，避免标签重叠
        wedges, texts, autotexts = ax.pie(sizes, labels=None, autopct='%1.1f%%', 
                                         colors=colors, startangle=90,
                                         pctdistance=0.85, labeldistance=1.1)
        
        # 手动设置字体
        chinese_font = getattr(self, 'chinese_font', 'Arial Unicode MS')
        
        # 设置标题
        ax.set_title('主要主题分布', fontfamily=chinese_font, fontsize=12, pad=20)
        
        # 创建图例代替标签，避免重叠
        ax.legend(wedges, labels, title="主题类型", loc="center left", 
                 bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10,
                 prop={'family': chinese_font})
        
        # 调整百分比文字
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontfamily(chinese_font)
            autotext.set_color('white')
            autotext.set_weight('bold')
    
    def _plot_emotion_analysis(self, analysis_result, ax):
        """绘制情感分析柱状图"""
        # 从正确的数据结构中提取情感数据
        emotions_data = analysis_result.get('emotions', {})
        emotions = emotions_data.get('frequency', {}) if isinstance(emotions_data, dict) else emotions_data
        
        chinese_font = getattr(self, 'chinese_font', 'Arial Unicode MS')
        
        if not emotions:
            ax.text(0.5, 0.5, '未检测到情感信息', ha='center', va='center', transform=ax.transAxes,
                   fontfamily=chinese_font)
            ax.set_title('情感倾向分析', fontfamily=chinese_font)
            return
        
        emotion_types = list(emotions.keys())
        emotion_counts = list(emotions.values())
        colors = ['green', 'red', 'gray'][:len(emotion_types)]  # 防止颜色不够
        
        bars = ax.bar(emotion_types, emotion_counts, color=colors, alpha=0.7)
        ax.set_title('情感倾向分析', fontfamily=chinese_font, fontsize=12)
        ax.set_xlabel('情感类型', fontfamily=chinese_font)
        ax.set_ylabel('出现频次', fontfamily=chinese_font)
        
        # 设置轴标签字体
        for label in ax.get_xticklabels():
            label.set_fontfamily(chinese_font)
        for label in ax.get_yticklabels():
            label.set_fontfamily(chinese_font)
        
        # 在柱子上显示数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom',
                   fontfamily=chinese_font, fontsize=9)
    
    def _plot_character_cooccurrence(self, analysis_result, ax):
        """绘制人物共现关系图"""
        # 从正确的数据结构中提取人物共现数据
        characters_data = analysis_result.get('characters', {})
        character_pairs = characters_data.get('cooccurrence', {}) if isinstance(characters_data, dict) else {}
        
        chinese_font = getattr(self, 'chinese_font', 'Arial Unicode MS')
        
        if not character_pairs:
            ax.text(0.5, 0.5, '未检测到人物关系', ha='center', va='center', transform=ax.transAxes,
                   fontfamily=chinese_font)
            ax.set_title('人物共现关系', fontfamily=chinese_font)
            return
        
        # 取前10个最常共现的人物对
        top_pairs = dict(sorted(character_pairs.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # 简化显示：只显示人物对的名称和共现次数
        pair_labels = list(top_pairs.keys())
        pair_counts = list(top_pairs.values())
        
        if pair_labels:
            bars = ax.barh(pair_labels, pair_counts, color='lightcoral', alpha=0.7)
            ax.set_title('人物共现关系', fontfamily=chinese_font, fontsize=12)
            ax.set_xlabel('共现次数', fontfamily=chinese_font)
            
            # 设置轴标签字体
            for label in ax.get_xticklabels():
                label.set_fontfamily(chinese_font)
            for label in ax.get_yticklabels():
                label.set_fontfamily(chinese_font)
            
            # 在柱子上显示数值
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{int(width)}', ha='left', va='center',
                       fontfamily=chinese_font, fontsize=9)
        else:
            ax.text(0.5, 0.5, '无有效人物关系数据', ha='center', va='center', transform=ax.transAxes,
                   fontfamily=chinese_font)
            ax.set_title('人物共现关系', fontfamily=chinese_font)
    
    def _plot_content_distribution(self, analysis_result, ax):
        """绘制内容在页面中的分布"""
        # 从叙事分析中提取页面分布信息
        narrative_data = analysis_result.get('narrative', {})
        scenes = narrative_data.get('scenes', []) if isinstance(narrative_data, dict) else []
        
        font_family = self.get_font_family()
        
        if not scenes:
            ax.text(0.5, 0.5, self.get_label('no_page_distribution'), ha='center', va='center', transform=ax.transAxes,
                   fontfamily=font_family)
            ax.set_title(self.get_label('content_distribution_title'), fontfamily=font_family)
            return
        
        # 从场景数据中提取页码和长度信息
        pages = [scene.get('start_page', 1) for scene in scenes]
        content_lengths = [scene.get('length', 0) for scene in scenes]
        
        if pages and content_lengths:
            ax.plot(pages, content_lengths, 'o-', color='purple', linewidth=2, markersize=6)
            ax.set_title(self.get_label('content_distribution_title'), fontfamily=font_family, fontsize=12)
            ax.set_xlabel(self.get_label('page_range'), fontfamily=font_family)
            ax.set_ylabel(self.get_label('scene_length'), fontfamily=font_family)
            ax.grid(True, alpha=0.3)
            
            # 设置轴标签字体
            for label in ax.get_xticklabels():
                label.set_fontfamily(font_family)
            for label in ax.get_yticklabels():
                label.set_fontfamily(font_family)
            
            # 标注峰值 - 修复重叠问题
            if content_lengths:
                max_idx = content_lengths.index(max(content_lengths))
                max_page = pages[max_idx]
                max_length = max(content_lengths)
                
                # 使用更简洁的注释，避免重叠
                if self.language == 'zh':
                    annotation_text = f'峰值: 第{max_page}页\n({max_length}字符)'
                else:
                    annotation_text = f'Peak: Page {max_page}\n({max_length} chars)'
                
                # 计算注释位置，避免重叠
                x_offset = 20 if max_idx < len(pages) / 2 else -80
                y_offset = 20
                
                ax.annotate(annotation_text,
                           xy=(max_page, max_length),
                           xytext=(x_offset, y_offset), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'),
                           fontfamily=font_family, fontsize=8)
        else:
            ax.text(0.5, 0.5, self.get_label('no_page_distribution'), ha='center', va='center', transform=ax.transAxes,
                   fontfamily=font_family)
            ax.set_title(self.get_label('content_distribution_title'), fontfamily=font_family)
    
    def _plot_summary_stats(self, analysis_result, ax):
        """绘制关键统计信息"""
        ax.axis('off')  # 隐藏坐标轴
        
        font_family = self.get_font_family()
        
        # 收集统计信息
        stats = []
        
        # 人物统计
        characters_data = analysis_result.get('characters', {})
        characters = characters_data.get('frequency', {}) if isinstance(characters_data, dict) else characters_data
        if characters:
            stats.append(f"{self.get_label('characters_detected')}: {len(characters)}{self.get_label('other') if self.language == 'zh' else ''}")
            top_character = max(characters.items(), key=lambda x: x[1])
            if self.language == 'zh':
                stats.append(f"{self.get_label('main_character')}: {top_character[0]} ({top_character[1]}次)")
            else:
                stats.append(f"{self.get_label('main_character')}: {top_character[0]} ({top_character[1]} times)")
        
        # 主题统计
        themes_data = analysis_result.get('themes', {})
        themes = themes_data.get('frequency', {}) if isinstance(themes_data, dict) else themes_data
        if themes:
            active_themes = {k: v for k, v in themes.items() if v > 0}
            stats.append(f"{self.get_label('active_themes')}: {len(active_themes)}{self.get_label('other') if self.language == 'zh' else ''}")
            if active_themes:
                top_theme = max(active_themes.items(), key=lambda x: x[1])
                if self.language == 'zh':
                    stats.append(f"{self.get_label('dominant_theme')}: {top_theme[0]} ({top_theme[1]}次)")
                else:
                    stats.append(f"{self.get_label('dominant_theme')}: {top_theme[0]} ({top_theme[1]} times)")
        
        # 情感统计
        emotions_data = analysis_result.get('emotions', {})
        emotions = emotions_data.get('frequency', {}) if isinstance(emotions_data, dict) else emotions_data
        if emotions:
            total_emotions = sum(emotions.values())
            if self.language == 'zh':
                stats.append(f"{self.get_label('emotion_total')}: {total_emotions}次")
            else:
                stats.append(f"{self.get_label('emotion_total')}: {total_emotions} times")
            
            if emotions:
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                if self.language == 'zh':
                    stats.append(f"{self.get_label('dominant_emotion')}: {dominant_emotion[0]} ({dominant_emotion[1]}次)")
                else:
                    stats.append(f"{self.get_label('dominant_emotion')}: {dominant_emotion[0]} ({dominant_emotion[1]} times)")
        
        # 叙事统计
        narrative_data = analysis_result.get('narrative', {})
        if narrative_data:
            scenes = narrative_data.get('scenes', [])
            if scenes:
                if self.language == 'zh':
                    stats.append(f"{self.get_label('scenes_detected')}: {len(scenes)}个")
                else:
                    stats.append(f"{self.get_label('scenes_detected')}: {len(scenes)}")
        
        # 设置标题
        ax.set_title(self.get_label('summary_stats_title'), fontfamily=font_family, fontsize=12, pad=20)
        
        # 显示统计信息
        if not stats:
            ax.text(0.5, 0.5, self.get_label('no_stats_data'), ha='center', va='center', 
                   transform=ax.transAxes, fontfamily=font_family, fontsize=14)
        else:
            # 创建一个美观的统计面板 - 纯文字版本
            title_emoji = '📊 ' if self.language == 'zh' else '📊 '
            ax.text(0.1, 0.9, f'{title_emoji}{self.get_label("analysis_summary")}', 
                   transform=ax.transAxes, fontfamily=font_family, fontsize=14, fontweight='bold')
            
            # 显示统计信息，每行一个
            for i, stat in enumerate(stats):
                y_pos = 0.75 - i * 0.1
                if y_pos > 0.1:  # 确保不超出边界
                    ax.text(0.1, y_pos, f"• {stat}", transform=ax.transAxes,
                           fontfamily=font_family, fontsize=11)
        
        # 设置图表边界
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)


# 全局可视化实例（默认中文）
visualizer = AdvancedVisualizer(language='zh')
