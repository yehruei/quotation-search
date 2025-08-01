#!/usr/bin/env python3
"""
Quotation Search App - Production Ready Edition
为多用户服务器部署优化的版本
"""

import streamlit as st
import os
import sys
import pandas as pd
import tempfile
import traceback
import random
import re
import time
import hashlib
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from modules.text_processor import process_pdf
from modules.retriever import search as standard_search
from modules.reranker import create_reranker
from modules.keyword_expander import KeywordExpander
from modules.literary_analyzer import LiteraryAnalyzer
from modules.theme_analyzer import LiteraryThemeAnalyzer
from modules.visualizer import AdvancedVisualizer

# 全局配置
CONFIG = {
    'MAX_FILE_SIZE': 50 * 1024 * 1024,  # 50MB
    'MAX_CONCURRENT_USERS': 10,
    'CACHE_TTL': 3600,  # 1小时
    'MAX_RESULTS_PER_USER': 50,
    'RATE_LIMIT_PER_HOUR': 10,
    'UPLOAD_DIR': 'uploads',
    'CACHE_DIR': 'cache',
    'TEMP_DIR': 'temp'
}

# 全局变量管理
class GlobalState:
    def __init__(self):
        self._model_path = None
        self._model_loaded = False
        self._loading_lock = threading.Lock()
        self._user_sessions = {}
        self._rate_limits = {}
        self._executor = ThreadPoolExecutor(max_workers=CONFIG['MAX_CONCURRENT_USERS'])
    
    def get_model_path(self):
        if not self._model_loaded:
            with self._loading_lock:
                if not self._model_loaded:
                    self._model_path = self._load_model_path()
                    self._model_loaded = True
        return self._model_path
    
    def _load_model_path(self):
        """加载模型路径 - 优化版"""
        try:
            return get_model_path()
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return None
    
    def check_rate_limit(self, user_id):
        """检查用户速率限制"""
        now = datetime.now()
        hour_key = now.strftime("%Y%m%d%H")
        
        if user_id not in self._rate_limits:
            self._rate_limits[user_id] = {}
        
        if hour_key not in self._rate_limits[user_id]:
            self._rate_limits[user_id][hour_key] = 0
        
        # 清理旧数据
        old_keys = [k for k in self._rate_limits[user_id].keys() if k < (now - timedelta(hours=2)).strftime("%Y%m%d%H")]
        for key in old_keys:
            del self._rate_limits[user_id][key]
        
        current_requests = self._rate_limits[user_id][hour_key]
        if current_requests >= CONFIG['RATE_LIMIT_PER_HOUR']:
            return False
        
        self._rate_limits[user_id][hour_key] += 1
        return True

# 全局状态实例
global_state = GlobalState()

# 缓存管理
@st.cache_data(ttl=CONFIG['CACHE_TTL'], max_entries=100)
def cached_process_pdf(file_hash, chunk_size, chunk_overlap, remove_stopwords):
    """缓存的PDF处理"""
    cache_file = os.path.join(CONFIG['CACHE_DIR'], f"pdf_{file_hash}.cache")
    
    if os.path.exists(cache_file):
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"缓存读取失败: {e}")
    
    return None

def save_pdf_cache(file_hash, texts, chunk_size, chunk_overlap, remove_stopwords):
    """保存PDF处理结果到缓存"""
    try:
        import pickle
        cache_file = os.path.join(CONFIG['CACHE_DIR'], f"pdf_{file_hash}.cache")
        with open(cache_file, 'wb') as f:
            pickle.dump(texts, f)
        logger.info(f"PDF缓存已保存: {file_hash}")
    except Exception as e:
        logger.error(f"缓存保存失败: {e}")

def get_file_hash(file_bytes):
    """计算文件哈希值"""
    return hashlib.md5(file_bytes).hexdigest()

def get_user_session_id():
    """获取用户会话ID"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(
            f"{time.time()}_{random.randint(1000, 9999)}".encode()
        ).hexdigest()
    return st.session_state.session_id

@st.cache_resource(ttl=CONFIG['CACHE_TTL'])
def get_model_path():
    """
    获取模型路径，优先使用本地模型，如果没有则返回None使用简单模型
    """
    import os
    
    # 检查项目内的本地模型位置
    base_path = os.getcwd()
    local_paths = [
        os.path.join(base_path, 'final_model_data/sentence_transformer_model'),
        os.path.join(base_path, 'models_cache/models--sentence-transformers--all-MiniLM-L6-v2'),
        os.path.join(base_path, 'models_cache/sentence-transformers--all-MiniLM-L6-v2'),
    ]
    
    # 检查本地路径是否存在且包含必要文件
    for local_path in local_paths:
        if os.path.exists(local_path):
            # 检查是否有必要的模型文件
            required_files = ['config.json']
            config_file = os.path.join(local_path, 'config.json')
            if os.path.exists(config_file):
                print(f"✅ 找到本地模型: {local_path}")
                return local_path
            else:
                print(f"⚠️ 路径存在但缺少config.json: {local_path}")
    
    # 检查用户的sentence-transformers缓存目录
    import os
    home_cache_paths = [
        os.path.expanduser("~/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2"),
        os.path.expanduser("~/.cache/huggingface/transformers/models--sentence-transformers--all-MiniLM-L6-v2"),
    ]
    
    for cache_path in home_cache_paths:
        if os.path.exists(cache_path):
            config_file = os.path.join(cache_path, 'config.json')
            if os.path.exists(config_file):
                print(f"✅ 找到系统缓存模型: {cache_path}")
                return cache_path
    
    # 如果没有找到本地模型，返回None使用简单模型
    print("🔄 未找到本地模型，将使用简单词汇模型（无需网络连接）")
    return None

def display_results(results):
    print(f"DEBUG: display_results 被调用，results 长度: {len(results) if results else '空或None'}")
    if results:
        print(f"DEBUG: 第一个结果内容: {results[0]}")
    
    if not results:
        st.warning("未找到相关结果。")
        print("DEBUG: 显示'未找到相关结果'警告")
        return
    
    # Display summary
    if len(results) > 0:
        st.success(f"找到 {len(results)} 个相关结果")
        print(f"DEBUG: 显示成功消息，找到 {len(results)} 个结果")
    
    for i, result in enumerate(results, 1):
        st.markdown("---")
        score = result.get('score', 0.0)
        is_fallback = result.get('is_fallback', False)
        search_method = result.get('method', '')
        
        # Score display with user-friendly labels (using 30-100 range)
        if score >= 80:
            score_color = "🟢"
            score_label = "高度相关"
            score_class = "score-high"
        elif score >= 65:
            score_color = "🟡"
            score_label = "较为相关"
            score_class = "score-medium"
        elif score >= 45:
            score_color = "🟠"
            score_label = "一般相关"
            score_class = "score-medium"
        else:
            score_color = "🔴"
            score_label = "相关性较低"
            score_class = "score-low"
        
        # 为补充结果和不同搜索方法添加特殊标识
        if is_fallback:
            fallback_indicator = " 📎 补充结果"
            score_label = "补充内容"
            score_color = "🔍"
        else:
            fallback_indicator = ""
        
        # 搜索方法标识
        if search_method:
            method_indicator = f" [{search_method}]"
        else:
            method_indicator = ""
        
        # Get page information 
        page_num = result.get('page_num', 'Unknown')
        
        # Use simpler, more reliable display approach
        st.markdown("---")
        
        # Result header with score badge
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**结果 {i}**")
            st.caption(f"📄 第 {page_num} 页")
        with col2:
            if score >= 80:
                st.markdown(f"🟢 **{score:.0f}分** (高度相关)")
            elif score >= 65:
                st.markdown(f"🟡 **{score:.0f}分** (较为相关)")
            elif score >= 45:
                st.markdown(f"🟠 **{score:.0f}分** (一般相关)")
            else:
                st.markdown(f"🔴 **{score:.0f}分** (相关性较低)")
            
            # Add fallback and method indicators
            if fallback_indicator:
                st.caption(f"📎 {fallback_indicator}")
            if method_indicator:
                st.caption(f"🔧 {method_indicator}")
        
        # Enhanced content display with highlighting
        content = result.get('text', '')
        found_keywords = result.get('found_keywords', [])
        
        # 智能分割内容以突出显示主要匹配部分
        highlighted_content = highlight_main_content(content, found_keywords)
        
        # 调试HTML内容
        print(f"DEBUG: highlighted_content 前100字符: {highlighted_content[:100]}")
        
        # 显示带有高亮的内容 - 改进HTML渲染稳定性
        try:
            # 验证HTML内容完整性
            if is_valid_html_content(highlighted_content):
                st.markdown(highlighted_content, unsafe_allow_html=True)
                print("✅ HTML内容正常渲染")
            else:
                print("⚠️ HTML内容验证失败，使用备用方案")
                raise ValueError("HTML验证失败")
        except Exception as e:
            print(f"HTML渲染错误: {e}")
            # 回退到简单的文本显示但保留基本格式
            st.markdown("**📄 内容：**")
            if found_keywords:
                # 简单的关键词高亮（非HTML）
                display_content = content
                for keyword in found_keywords[:3]:  # 只处理前3个关键词
                    if keyword in display_content:
                        display_content = display_content.replace(keyword, f"**{keyword}**")
                st.markdown(display_content)
                st.caption(f"🔍 匹配关键词: {', '.join(found_keywords[:5])}")
            else:
                st.markdown(content)
            print("✅ 回退到简单文本显示")
        
        # Additional metadata if available
        if found_keywords:
            st.markdown(f"**🔍 匹配关键词:** {', '.join(found_keywords[:5])}")  # Show first 5 keywords

def is_valid_html_content(html_content):
    """
    验证HTML内容的完整性和有效性
    
    Args:
        html_content: 待验证的HTML内容
    
    Returns:
        bool: HTML内容是否有效
    """
    if not html_content or not isinstance(html_content, str):
        return False
    
    # 检查基本的HTML结构
    if not ('<div' in html_content and '</div>' in html_content):
        return False
    
    # 计算标签匹配
    import re
    
    # 提取所有HTML标签
    opening_tags = re.findall(r'<(\w+)[^>]*>', html_content)
    closing_tags = re.findall(r'</(\w+)>', html_content)
    
    # 检查标签是否配对（忽略自闭合标签如<br/>）
    self_closing_tags = {'br', 'hr', 'img', 'input', 'meta', 'link'}
    
    tag_stack = []
    for tag in opening_tags:
        if tag not in self_closing_tags:
            tag_stack.append(tag)
    
    for tag in closing_tags:
        if tag_stack and tag_stack[-1] == tag:
            tag_stack.pop()
        else:
            print(f"⚠️ HTML标签不匹配: 期望 {tag_stack[-1] if tag_stack else 'None'}, 实际 {tag}")
            return False
    
    # 如果还有未匹配的开放标签，则无效
    if tag_stack:
        print(f"⚠️ 未关闭的HTML标签: {tag_stack}")
        return False
    
    # 检查是否包含潜在的有害内容
    dangerous_patterns = [
        r'<script[^>]*>',
        r'javascript:',
        r'on\w+\s*=',  # 事件处理器
        r'<iframe[^>]*>',
        r'<object[^>]*>',
        r'<embed[^>]*>'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, html_content, re.IGNORECASE):
            print(f"⚠️ 检测到潜在危险的HTML内容")
            return False
    
    return True

def highlight_main_content(content, keywords):
    """
    智能高亮显示主要匹配内容和上下文 - 改进版
    
    Args:
        content: 原始文本内容
        keywords: 匹配的关键词列表
    
    Returns:
        带有HTML高亮标记的内容
    """
    # 确保输入安全
    if not content or not isinstance(content, str):
        return create_fallback_html("内容为空")
    
    if not keywords:
        return create_simple_content_html(content)
    
    try:
        # 确保内容安全 - 转义HTML特殊字符
        import html
        safe_content = html.escape(content)
        
        # 改进的句子分割，支持中英文
        import re
        
        # 更智能的句子分割
        sentence_pattern = r'[.!?。！？]+\s*'
        sentences = re.split(sentence_pattern, safe_content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 如果分割后句子太少，尝试其他分割方式
        if len(sentences) <= 1:
            sentences = [s.strip() for s in safe_content.split('\n') if s.strip()]
        
        if len(sentences) == 1 and len(safe_content) > 200:
            parts = re.split(r'[,;，；]\s*', safe_content)
            if len(parts) > 1:
                sentences = [s.strip() for s in parts if s.strip()]
        
        # 找出包含关键词的主要句子
        main_sentences = []
        context_sentences = []
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            has_keyword = False
            matched_keywords = []
            
            for keyword in keywords:
                if keyword.lower() in sentence_lower:
                    has_keyword = True
                    matched_keywords.append(keyword)
            
            if has_keyword:
                main_sentences.append((i, sentence, matched_keywords))
            else:
                context_sentences.append((i, sentence))
        
        # 构建HTML内容
        if not main_sentences:
            # 没有明确匹配，检查部分匹配
            partial_matches = []
            for i, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                for keyword in keywords:
                    if any(word in sentence_lower for word in keyword.lower().split()):
                        partial_matches.append((i, sentence))
                        break
            
            if partial_matches:
                main_text = '. '.join([sent for _, sent in partial_matches[:2]]) + '.'
                return create_highlighted_html(main_text, [], "相关内容")
            else:
                return create_simple_content_html(safe_content)
        else:
            # 有明确的关键词匹配
            main_text = '. '.join([sent for _, sent, _ in main_sentences])
            if not main_text.endswith('.'):
                main_text += '.'
            
            # 高亮关键词
            for keyword in keywords:
                escaped_keyword = html.escape(keyword)
                pattern = r'\b' + re.escape(escaped_keyword) + r'\b'
                highlight_span = f'<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; font-weight: bold; color: #333;">{escaped_keyword}</mark>'
                main_text = re.sub(pattern, highlight_span, main_text, flags=re.IGNORECASE)
            
            # 收集匹配的关键词
            all_matched_keywords = []
            for _, _, matched_kw in main_sentences:
                all_matched_keywords.extend(matched_kw)
            unique_matched = list(set(all_matched_keywords))
            
            # 添加上下文
            context_text = ""
            if context_sentences and len(context_sentences) <= 2:
                context_text = '. '.join([sent for _, sent in context_sentences])
                if context_text and not context_text.endswith('.'):
                    context_text += '.'
            
            return create_comprehensive_html(main_text, context_text, unique_matched)
            
    except Exception as e:
        print(f"HTML生成错误: {e}")
        import traceback
        traceback.print_exc()
        return create_fallback_html(f"处理出错: {str(e)}")

def create_simple_content_html(content):
    """创建简单内容的HTML"""
    return f'''
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #6c757d; font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;">
        <p style="margin: 0; color: #495057; line-height: 1.6;">{content}</p>
    </div>
    '''

def create_highlighted_html(main_text, keywords, title="主要匹配内容"):
    """创建高亮内容的HTML"""
    return f'''
    <div style="background-color: #ffffff; padding: 18px; border-radius: 10px; margin: 12px 0; border: 1px solid #e0e0e0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;">
        <div style="background-color: #d1ecf1; padding: 12px; border-radius: 6px;">
            <strong style="color: #0c5460;">🎯 {title}:</strong>
            <p style="margin: 5px 0; color: #0c5460; line-height: 1.6;">{main_text}</p>
        </div>
    </div>
    '''

def create_comprehensive_html(main_text, context_text, matched_keywords):
    """创建综合内容的HTML"""
    keyword_count = len(matched_keywords)
    
    html_parts = [f'''
    <div style="background-color: #ffffff; padding: 18px; border-radius: 10px; margin: 12px 0; border: 1px solid #e0e0e0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;">
        <div style="background-color: #e8f5e8; padding: 14px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid #4caf50; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <strong style="color: #2e7d32; font-size: 16px;">🎯 主要匹配内容</strong>
                <span style="background-color: #4caf50; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-left: 10px;">
                    {keyword_count} 个关键词匹配
                </span>
            </div>
            <p style="margin: 0; color: #1b5e20; line-height: 1.7; font-size: 15px;">{main_text}</p>
        </div>''']
    
    if context_text:
        html_parts.append(f'''
        <div style="background-color: #fff3e0; padding: 12px; border-radius: 6px; border-left: 4px solid #ff9800;">
            <strong style="color: #f57c00;">📄 相关上下文:</strong>
            <p style="margin: 5px 0; color: #e65100; line-height: 1.6; font-size: 14px;">{context_text}</p>
        </div>''')
    
    html_parts.append('    </div>')
    
    return ''.join(html_parts)

def create_fallback_html(error_msg):
    """创建错误回退的HTML"""
    return f'''
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #dc3545; font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;">
        <p style="margin: 0; color: #721c24; line-height: 1.6;">⚠️ {error_msg}</p>
    </div>
    '''

def simple_highlight_content(content, keywords):
    """
    简化版高亮函数 - 备用方案
    """
    if not keywords or not content:
        return content
    
    # 简单的文本高亮，使用Markdown语法
    display_content = content
    for keyword in keywords[:5]:  # 最多处理5个关键词
        # 使用Markdown的粗体语法进行高亮
        display_content = display_content.replace(keyword, f"**{keyword}**")
    
    return display_content

# --- UI Layout ---

def main():
    """主函数 - 生产环境优化版"""
    logger.info("🚀 应用程序启动...")
    
    # 获取用户会话ID
    user_id = get_user_session_id()
    
    # 检查速率限制
    if not global_state.check_rate_limit(user_id):
        st.error("⚠️ 您已达到每小时请求限制。请稍后再试。")
        st.info(f"每小时最多允许 {CONFIG['RATE_LIMIT_PER_HOUR']} 次请求")
        return
    
    try:
        st.set_page_config(
            page_title="Quotation Search - 智能文本分析平台", 
            layout="wide", 
            initial_sidebar_state="collapsed",
            page_icon="📚"
        )
        logger.info("✅ Streamlit页面配置完成")
    except Exception as e:
        logger.error(f"❌ 页面配置失败: {e}")
        st.error("页面配置失败，请刷新重试")
        return

    # 显示服务状态
    with st.sidebar:
        st.header("📊 服务状态")
        st.metric("当前用户", len(global_state._user_sessions))
        st.metric("今日请求", global_state._rate_limits.get(user_id, {}).get(datetime.now().strftime("%Y%m%d%H"), 0))
        
        # 系统信息
        with st.expander("🔧 系统信息"):
            import psutil
            st.write(f"CPU使用率: {psutil.cpu_percent()}%")
            st.write(f"内存使用率: {psutil.virtual_memory().percent}%")
            st.write(f"磁盘使用率: {psutil.disk_usage('/').percent}%")

    # 添加性能监控的CSS样式（简化版）
    st.markdown("""
    <style>
    .stApp {
        font-family: 'PingFang SC', 'Microsoft YaHei', 'Inter', sans-serif;
        background-color: #f8f9fa;
    }
    .main .block-container {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        font-family: 'PingFang SC', 'Microsoft YaHei', 'Inter', sans-serif;
    }
    /* 简化的样式以提高性能 */
    .stButton > button {
        width: 100%;
        margin-top: 1rem;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 主标题
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("📚 智能文本分析平台")
    st.markdown("专业的文学作品搜索、分析和可视化工具")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 尝试获取模型路径
    try:
        model_path = global_state.get_model_path()
        if model_path:
            st.success("✅ AI模型已就绪")
        else:
            st.warning("⚠️ 使用简化模式（功能可能受限）")
        logger.info(f"✅ 模型状态检查完成: {model_path}")
    except Exception as e:
        logger.error(f"❌ 模型检查失败: {e}")
        st.error("模型加载失败，请联系管理员")
        model_path = None

    # 主要配置区域
    with st.container():
        st.header("⚙️ 文档分析配置")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📄 上传PDF文件")
            uploaded_file = st.file_uploader(
                "选择要分析的PDF文件",
                type="pdf",
                help=f"支持PDF格式，最大文件大小: {CONFIG['MAX_FILE_SIZE'] // (1024*1024)}MB"
            )
            
            if uploaded_file:
                # 检查文件大小
                if len(uploaded_file.getvalue()) > CONFIG['MAX_FILE_SIZE']:
                    st.error(f"文件太大！最大支持 {CONFIG['MAX_FILE_SIZE'] // (1024*1024)}MB")
                    return
                
                # 显示文件信息
                file_size = len(uploaded_file.getvalue()) / 1024 / 1024
                st.info(f"📄 {uploaded_file.name} ({file_size:.1f}MB)")

        with col2:
            st.subheader("🔍 搜索关键词")
            query = st.text_input(
                "输入搜索关键词",
                placeholder="例如: 爱情, 死亡, 权力",
                help="输入您要搜索的关键词或短语"
            )

        # 高级配置（简化版）
        with st.expander("⚙️ 高级设置", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                use_keyword_expansion = st.checkbox("启用关键词扩展", value=True)
                min_results = st.slider("最少返回结果", 1, 10, 3)
                enable_retry = st.checkbox("启用智能重试", value=True)
            
            with col2:
                final_max_results = st.slider("最终结果数量", 3, 20, 10)
                similarity_threshold = st.slider("相似度阈值", 0.0, 0.5, 0.1, 0.05)
                reranker_type = st.selectbox("排序方法", 
                    ["enhanced_cross_encoder", "simple", "none"], 
                    index=0)
            
            with col3:
                auto_perform_analysis = st.checkbox("自动分析", value=True)
                perform_visualization = st.checkbox("生成可视化", value=True)
                chunk_size = st.slider("文本块大小", 500, 1500, 800)

        # 搜索按钮
        search_button = st.button("🚀 开始分析", type="primary", use_container_width=True)

    # 处理搜索请求
    if uploaded_file and query and search_button:
        process_search_request(
            uploaded_file, query, model_path, user_id,
            use_keyword_expansion, min_results, enable_retry,
            final_max_results, similarity_threshold, reranker_type,
            auto_perform_analysis, perform_visualization, chunk_size
        )

    # 显示结果（如果存在）
    display_cached_results()

def process_search_request(uploaded_file, query, model_path, user_id, 
                         use_keyword_expansion, min_results, enable_retry,
                         final_max_results, similarity_threshold, reranker_type,
                         auto_perform_analysis, perform_visualization, chunk_size):
    """处理搜索请求 - 优化版"""
    
    start_time = time.time()
    logger.info(f"用户 {user_id} 开始搜索: {query}")
    
    with st.spinner("正在处理您的请求..."):
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        
        try:
            # 1. 处理PDF文件
            status_placeholder.info("📄 正在处理PDF文件...")
            progress_bar.progress(10)
            
            file_bytes = uploaded_file.getvalue()
            file_hash = get_file_hash(file_bytes)
            
            # 尝试从缓存获取
            texts = cached_process_pdf(file_hash, chunk_size, 80, False)
            
            if texts is None:
                # 处理PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=CONFIG['TEMP_DIR']) as tmp_file:
                    tmp_file.write(file_bytes)
                    texts = process_pdf(tmp_file.name, chunk_size=chunk_size, chunk_overlap=80, remove_stopwords=False)
                    os.unlink(tmp_file.name)
                
                # 保存到缓存
                save_pdf_cache(file_hash, texts, chunk_size, 80, False)
                logger.info(f"PDF处理完成，提取了 {len(texts)} 个文本块")
            else:
                logger.info(f"从缓存获取PDF数据，{len(texts)} 个文本块")
            
            if not texts:
                st.error("无法从PDF中提取文本")
                return
            
            progress_bar.progress(30)
            
            # 2. 关键词扩展
            final_query = query
            if use_keyword_expansion:
                status_placeholder.info("🔍 正在扩展关键词...")
                try:
                    expander = KeywordExpander(method="semantic", document_type="literary")
                    expanded_keywords = expander.expand_keywords(
                        query.split(), max_synonyms_per_word=3, max_related_per_word=2
                    )
                    final_query = " ".join(expanded_keywords.keys())
                    st.session_state.expanded_keywords = expanded_keywords
                    logger.info(f"关键词扩展: {len(query.split())} → {len(expanded_keywords)}")
                except Exception as e:
                    logger.warning(f"关键词扩展失败: {e}")
                    final_query = query
            
            progress_bar.progress(50)
            
            # 3. 搜索
            status_placeholder.info("🔍 正在搜索相关内容...")
            try:
                if enable_retry:
                    results = perform_smart_search(final_query, texts, model_path, final_max_results, min_results)
                else:
                    results = standard_search(final_query, texts, model_path)
                    results = results[:final_max_results] if results else []
                
                logger.info(f"搜索完成，找到 {len(results)} 个结果")
            except Exception as e:
                logger.error(f"搜索失败: {e}")
                st.error(f"搜索失败: {e}")
                return
            
            progress_bar.progress(70)
            
            # 4. 结果处理和重排
            if results and reranker_type != "none":
                status_placeholder.info("📊 正在优化结果排序...")
                try:
                    reranker = create_reranker(reranker_type, keywords=final_query.split())
                    candidates = [{'content': r.get('text', ''), 'score': r.get('score', 0), 
                                 'page_num': r.get('page_num', 1), 'index': i} 
                                 for i, r in enumerate(results)]
                    reranked = reranker.rerank(candidates, k=final_max_results)
                    results = [{'text': c['content'], 'score': c.get('rerank_score', c.get('score', 0)),
                              'page_num': c.get('page_num', 1)} for c in reranked]
                    logger.info(f"重排完成，最终 {len(results)} 个结果")
                except Exception as e:
                    logger.warning(f"重排失败: {e}")
            
            progress_bar.progress(90)
            
            # 5. 保存结果
            st.session_state.results = results
            st.session_state.texts = texts
            st.session_state.query = query
            st.session_state.processing_time = time.time() - start_time
            
            # 6. 自动分析（如果启用）
            if auto_perform_analysis and results:
                status_placeholder.info("🎭 正在进行文学分析...")
                try:
                    perform_auto_analysis(texts)
                except Exception as e:
                    logger.warning(f"自动分析失败: {e}")
            
            progress_bar.progress(100)
            status_placeholder.success(f"✅ 处理完成！用时 {time.time() - start_time:.1f} 秒")
            
            logger.info(f"用户 {user_id} 搜索完成，用时 {time.time() - start_time:.1f} 秒")
            
        except Exception as e:
            logger.error(f"处理失败: {e}")
            st.error(f"处理失败: {e}")
            progress_bar.empty()
            status_placeholder.empty()

def perform_smart_search(query, texts, model_path, max_results, min_results):
    """执行智能搜索"""
    # 简化的搜索逻辑
    results = standard_search(query, texts, model_path)
    if len(results) < min_results and len(texts) > min_results:
        # 简单的补充机制
        import random
        additional_needed = min_results - len(results)
        remaining_texts = [texts[i] for i in range(len(texts)) if i not in [r.get('index', -1) for r in results]]
        if remaining_texts:
            additional = random.sample(remaining_texts, min(additional_needed, len(remaining_texts)))
            for i, text in enumerate(additional):
                results.append({
                    'text': text,
                    'score': 0.1,
                    'page_num': len(results) + i + 1,
                    'is_fallback': True
                })
    
    return results[:max_results]

def perform_auto_analysis(texts):
    """执行自动分析"""
    try:
        full_text_chunks = [{'content': text, 'page_num': i+1} for i, text in enumerate(texts)]
        
        # 简化的分析
        theme_analyzer = LiteraryThemeAnalyzer()
        theme_result = theme_analyzer.analyze_text_themes(full_text_chunks)
        
        literary_analyzer = LiteraryAnalyzer()
        comprehensive_results = literary_analyzer.generate_comprehensive_analysis(full_text_chunks)
        
        analysis_results = {
            'theme_analysis': theme_result,
            'character_analysis': comprehensive_results.get('characters', {}),
            'emotion_analysis': comprehensive_results.get('emotions', {}),
            'narrative_analysis': comprehensive_results.get('narrative', {}),
            'comprehensive_results': comprehensive_results
        }
        
        st.session_state.analysis_results = analysis_results
        logger.info("自动分析完成")
        
    except Exception as e:
        logger.error(f"自动分析失败: {e}")
        raise

def display_cached_results():
    """显示缓存的结果"""
    if 'results' in st.session_state and st.session_state.results:
        st.markdown("---")
        st.header("🎯 搜索结果")
        
        # 显示处理时间
        if 'processing_time' in st.session_state:
            st.info(f"⏱️ 处理用时: {st.session_state.processing_time:.1f} 秒")
        
        display_results(st.session_state.results)
        
        # 显示分析结果
        if 'analysis_results' in st.session_state:
            display_analysis_results()
        
        # 显示可视化
        if st.session_state.get('perform_visualization', False):
            display_visualizations()

def display_analysis_results():
    """显示分析结果"""
    st.markdown("---")
    st.header("🎭 文学分析结果")
    
    results = st.session_state.analysis_results
    
    if results.get('comprehensive_results'):
        st.subheader("📈 综合分析可视化")
        try:
            visualizer = AdvancedVisualizer(language='zh')
            fig = visualizer.plot_literary_analysis(results['comprehensive_results'])
            if fig:
                st.pyplot(fig)
                st.success("✅ 分析图表生成成功")
        except Exception as e:
            logger.error(f"分析可视化失败: {e}")
            st.error("分析可视化生成失败")

def display_visualizations():
    """显示可视化图表"""
    st.markdown("---")
    st.header("📊 数据可视化")
    
    try:
        visualizer = AdvancedVisualizer(language='zh')
        results = st.session_state.results
        keywords = st.session_state.get('query', '').split()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 页面分布图
            fig1 = visualizer.plot_page_distribution(results, keywords)
            if fig1:
                st.pyplot(fig1)
            
            # 词云图  
            fig3 = visualizer.plot_word_cloud(results, keywords, max_words=100)
            if fig3:
                st.pyplot(fig3)
        
        with col2:
            # 主题频率图
            fig2 = visualizer.plot_theme_frequency(results, keywords)
            if fig2:
                st.pyplot(fig2)
            
            # 共现热力图
            fig4 = visualizer.plot_cooccurrence_heatmap(results, keywords)
            if fig4:
                st.pyplot(fig4)
                
    except Exception as e:
        logger.error(f"可视化失败: {e}")
        st.error("可视化生成失败")

    # Enhanced CSS for modern, readable styling
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles - Clean background with Chinese font support */
    .stApp {
        font-family: 'PingFang SC', 'Microsoft YaHei', 'Inter', 'Arial Unicode MS', sans-serif;
        background-color: #f8f9fa;
    }
    
    /* Main content area with subtle styling and Chinese font support */
    .main .block-container {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        font-family: 'PingFang SC', 'Microsoft YaHei', 'Inter', 'Arial Unicode MS', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: #1f2937;
        margin-bottom: 2rem;
        position: relative;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        border-radius: 2px;
    }
    
    /* Configuration sections with clean styling */
    .config-section {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .config-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: #3b82f6;
    }
    
    /* Button styling - consistent and readable */
    .stButton > button {
        width: 100%;
        margin-top: 1rem;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px 0 rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    }
    
    /* Input field styling - better visibility */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        background: white;
        border: 2px solid #d1d5db;
        border-radius: 8px;
        color: #374151;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: #f8fafc;
        border: 2px dashed #d1d5db;
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #3b82f6;
        background: #eff6ff;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        color: white;
    }
    
    .streamlit-expanderContent {
        background: #f8fafc;
        border-radius: 0 0 8px 8px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Progress bar */
    .stProgress .progress-bar {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    
    /* Success/Error/Warning messages - better contrast */
    .stSuccess {
        background: #ecfdf5;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        color: #047857;
    }
    
    .stError {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        color: #dc2626;
    }
    
    .stWarning {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        color: #d97706;
    }
    
    .stInfo {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        color: #2563eb;
    }
    
    /* Custom result cards */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.06), 0 1px 2px -1px rgba(0, 0, 0, 0.06);
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: #3b82f6;
    }
    
    /* Score badge styling - better contrast */
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.25rem;
    }
    
    .score-high {
        background: #10b981;
        color: white;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    
    .score-medium {
        background: #f59e0b;
        color: white;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.3);
    }
    
    .score-low {
        background: #6b7280;
        color: white;
        box-shadow: 0 2px 8px rgba(107, 114, 128, 0.3);
    }
    
    /* Gradient text for headers */
    .gradient-text {
        background: linear-gradient(-45deg, #3b82f6, #8b5cf6);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 3s ease infinite;
        font-weight: 700;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Floating animation */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }
    
    .floating {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Text colors for better readability with Chinese font support */
    h1, h2, h3, h4, h5, h6 {
        color: #1f2937;
        font-family: 'PingFang SC', 'Microsoft YaHei', 'Inter', 'Arial Unicode MS', sans-serif !important;
    }
    
    p, span, div {
        color: #374151;
        font-family: 'PingFang SC', 'Microsoft YaHei', 'Inter', 'Arial Unicode MS', sans-serif !important;
    }
    
    /* Force Chinese font support for all elements */
    * {
        font-family: 'PingFang SC', 'Microsoft YaHei', 'Inter', 'Arial Unicode MS', sans-serif !important;
    }
    
    /* Checkbox and radio button styling */
    .stCheckbox > label {
        color: #374151;
        font-weight: 500;
        font-family: 'PingFang SC', 'Microsoft YaHei', 'Inter', 'Arial Unicode MS', sans-serif !important;
    }
    
    .stRadio > label {
        color: #374151;
        font-weight: 500;
        font-family: 'PingFang SC', 'Microsoft YaHei', 'Inter', 'Arial Unicode MS', sans-serif !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #3b82f6;
    }
    </style>
    """, unsafe_allow_html=True)
    print("✅ CSS样式加载完成")

    # Main header with clean styling
    st.markdown('<div class="main-header floating">', unsafe_allow_html=True)
    st.title("📚 引文搜索系统")
    st.markdown("智能文本搜索、分析和可视化的综合工具 - 专为教师和学生设计")
    st.markdown('</div>', unsafe_allow_html=True)
    print("✅ 页面标题显示完成")

    try:
        model_path = get_model_path()
        print(f"✅ 模型路径获取完成: {model_path}")
    except Exception as e:
        print(f"❌ 模型路径获取失败: {e}")
        st.error(f"模型路径获取失败: {e}")
        model_path = 'sentence-transformers/all-MiniLM-L6-v2'
    
    # 检查模型状态
    try:
        from modules.retriever import load_sentence_transformer_model
        test_model = load_sentence_transformer_model(model_path)
        if test_model is not None:
            # 检查是否是真正的SentenceTransformer模型
            if hasattr(test_model, 'encode') and hasattr(test_model, '_modules'):
                # 检查具体使用的模型路径
                local_paths = [
                    'final_model_data/sentence_transformer_model',
                    'models_cache/models--sentence-transformers--all-MiniLM-L6-v2'
                ]
                local_model_found = any(os.path.exists(path) for path in local_paths)
                
                if local_model_found:
                    st.success("✅ 本地语义模型加载成功! (使用缓存模型)")
                else:
                    st.success("✅ 在线语义模型加载成功!")
            else:
                st.warning("⚠️ 使用简单词汇模型 (语义模型不可用)")
        else:
            st.warning("⚠️ 语义模型加载失败，将使用简单词汇模型。功能可能受限。")
    except Exception as e:
        st.error(f"❌ 模型检查失败: {e}")
        st.info("💡 建议检查网络连接或重启应用程序")

    # --- Main Configuration Section ---
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.header("⚙️ 基本配置")
    
    # Create columns for the main input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # --- File Upload ---
        st.subheader("📄 上传PDF文件")
        uploaded_file = st.file_uploader("选择要搜索的PDF文件。", type="pdf", 
                                        help="上传您要搜索的PDF文档。")

    with col2:
        # --- Search Query ---
        st.subheader("🔍 输入搜索关键词")
        query = st.text_input("关键词:", placeholder="例如: 爱情, 死亡, 权力, 野心",
                             help="输入您要搜索的关键词或短语。")

    # --- 初始化默认配置变量 (确保在全局作用域内) ---
    # Core analysis settings
    perform_literary_analysis = True
    min_cooccurrence = 2
    perform_theme_analysis = True
    perform_visualization = True
    auto_perform_analysis = False
    
    # Text processing defaults
    chunk_size = 800
    chunk_overlap = 80
    remove_stopwords = False
    
    # Search configuration defaults
    use_keyword_expansion = True
    min_results = 3
    enable_retry = True
    use_hybrid_search = True
    
    # Keyword expansion defaults
    expander_method = "semantic"
    document_type = "literary"  
    max_synonyms = 3
    max_related_words = 2
    semantic_threshold = 0.75
    use_hierarchical = False
    
    # Hybrid search defaults
    fusion_method = "rrf"
    rrf_k = 50
    bm25_weight = 0.3
    embedding_weight = 0.7
    enable_parallel = True
    
    # Result processing defaults
    reranker_type = "enhanced_cross_encoder"
    initial_max_results = 20
    final_max_results = 10
    similarity_threshold = 0.1
    
    # Reranker defaults
    simple_reranker_threshold = 0.2
    cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_threshold = 0.2
    enable_ensemble = True
    enable_consistency_check = True
    enable_diversity = True
    enable_context_aware = True
    diversity_threshold = 0.7
    consistency_weight = 0.3
    
    # Visualization defaults  
    wordcloud_max_words = 100
    heatmap_top_n = 8
    plot_style = "whitegrid"
    figure_size = "medium"
    
    # System defaults
    cache_expire_days = 7

    # --- Advanced Configuration (moved above search button) ---
    st.markdown("---")
    with st.expander("⚙️ 高级搜索配置 (可选)", expanded=False):
        st.info("💡 以下参数已设置为适合教师和学生使用的推荐值。如需调整，请参考说明。")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🔍 搜索配置")
            use_keyword_expansion = st.checkbox("启用关键词扩展", value=True,
                                              help="🔍 **关键词扩展**: 自动添加相关词汇提高搜索效果。推荐启用。")
            
            min_results = st.slider("最少返回结果", 1, 10, 3,
                                   help="📊 **最少返回结果**: 质量保证的底线。即使相似度很低，也至少返回这么多结果。通常设为3个确保有内容查看。")
            
            enable_retry = st.checkbox("启用智能重试", value=True,
                                     help="🔄 **智能重试**: 结果不足时自动尝试其他搜索策略。强烈推荐启用。")
            
            if use_keyword_expansion:
                expander_method = st.selectbox("扩展方法", ["semantic", "wordnet"], index=0,
                                             format_func=lambda x: "语义扩展 (推荐)" if x == "semantic" else "词典扩展",
                                             help="📚 **扩展方法**: 语义扩展更智能，词典扩展更保守。")
                document_type = st.selectbox("文档类型", ["literary", "general"], index=0,
                                           format_func=lambda x: "文学作品 (推荐)" if x == "literary" else "一般文档",
                                           help="📖 **文档类型**: 文学作品针对诗歌、小说等优化。")
                max_synonyms = st.slider("同义词数量", 1, 10, 3,
                                       help="📝 **同义词数量**: 每个词的同义词个数。3个平衡效果和准确性。")
                max_related_words = st.slider("相关词数量", 1, 10, 2,
                                            help="🔗 **相关词数量**: 扩展的相关词个数。2个避免偏离主题。")
                semantic_threshold = st.slider("语义相似度", 0.1, 1.0, 0.75, 0.05,
                                             help="🎯 **语义相似度**: 扩展词的相似度要求。0.75确保高质量扩展。")
                use_hierarchical = st.checkbox("分层扩展", value=False,
                                             help="🌳 **分层扩展**: 多层次扩展关键词。可能增加无关结果。")
            else:
                expander_method = "semantic"
                document_type = "literary"
                max_synonyms = 3
                max_related_words = 2
                semantic_threshold = 0.75
                use_hierarchical = False

            st.markdown("**🔄 混合搜索**")
            use_hybrid_search = st.checkbox("启用混合搜索", value=True,
                                          help="🚀 **混合搜索**: 结合关键词和语义搜索。强烈推荐启用。")
            if use_hybrid_search:
                fusion_method = st.selectbox("融合方法", ["rrf", "weighted"], index=0,
                                           format_func=lambda x: "RRF融合 (推荐)" if x == "rrf" else "加权融合",
                                           help="⚖️ **融合方法**: RRF更平衡，加权融合可调性更强。")
                rrf_k = st.slider("RRF参数", 10, 100, 50,
                                help="🔧 **RRF参数**: 控制排名融合强度。50为平衡值。")
                bm25_weight = st.slider("关键词搜索权重", 0.0, 1.0, 0.3, 0.1,
                                      help="🔤 **关键词权重**: 传统关键词搜索的重要性。0.3适合文学分析。")
                embedding_weight = st.slider("语义搜索权重", 0.0, 1.0, 0.7, 0.1,
                                           help="🧠 **语义权重**: 智能语义理解的重要性。0.7适合深度分析。")
                enable_parallel = st.checkbox("并行处理", value=True,
                                             help="⚡ **并行处理**: 加速搜索过程。推荐启用。")
            else:
                fusion_method = "rrf"
                rrf_k = 50
                bm25_weight = 0.3
                embedding_weight = 0.7
                enable_parallel = True

        with col2:
            st.subheader("📊 结果处理")
            reranker_type = st.selectbox("结果排序方法", 
                                       ["enhanced_cross_encoder", "cross_encoder", "simple", "none"], 
                                       index=0,  # enhanced_cross_encoder as default
                                       format_func=lambda x: {
                                           "enhanced_cross_encoder": "智能增强排序 (推荐)",
                                           "cross_encoder": "交叉编码器排序",
                                           "simple": "简单排序",
                                           "none": "不排序"
                                       }.get(x, x),
                                       help="🎯 **排序方法**: 智能增强排序准确性最高，适合学术研究。")
            
            # 结果数量控制 - 分为两个阶段
            st.markdown("**📊 结果数量控制**")
            st.info("💡 **两阶段搜索**: 先大范围搜索候选结果，再精确重排选出最佳结果，提高搜索质量")
            col1, col2 = st.columns(2)
            
            with col1:
                initial_max_results = st.slider("初步搜索结果数", 10, 100, 20, 5,
                                               help="� **初步搜索**: 第一阶段返回的候选结果数量。更多结果可以提供更多选择给重排器。")
            
            with col2:
                final_max_results = st.slider("最终返回结果数", 3, 30, 10, 1,
                                             help="✅ **最终结果**: 重排和过滤后最终显示给用户的结果数量。这是您实际看到的结果数量。建议10个便于阅读。")
            
            similarity_threshold = st.slider("相似度阈值", 0.0, 0.5, 0.1, 0.05,
                                           help="📏 **相似度阈值**: 过滤低质量结果。0.1平衡数量和质量。")

            if reranker_type == "simple":
                simple_reranker_threshold = st.slider("简单排序阈值", 0.0, 1.0, 0.2, 0.05,
                                                     help="⚖️ **排序阈值**: 简单排序的过滤标准。")
            elif reranker_type == "cross_encoder":
                cross_encoder_model = st.text_input("交叉编码器模型", 
                                                   "cross-encoder/ms-marco-MiniLM-L-6-v2",
                                                   help="🤖 **模型名称**: 使用的AI模型。默认适合中英文。")
                cross_encoder_threshold = st.slider("交叉编码器阈值", 0.0, 1.0, 0.2, 0.05,
                                                   help="🎯 **编码器阈值**: 质量过滤标准。")
            elif reranker_type == "enhanced_cross_encoder":
                enable_ensemble = st.checkbox("启用集成模型", value=True,
                                             help="🤝 **集成模型**: 多个AI模型协作提高准确性。")
                enable_consistency_check = st.checkbox("一致性检查", value=True,
                                                      help="✅ **一致性检查**: 确保结果稳定可靠。")
                enable_diversity = st.checkbox("结果多样性", value=True,
                                             help="🌈 **多样性**: 避免相似结果重复。")
                enable_context_aware = st.checkbox("上下文感知", value=True,
                                                  help="🧠 **上下文感知**: 理解文本背景提高相关性。")
                # Additional enhanced options with Chinese descriptions
                diversity_threshold = st.slider("多样性阈值", 0.1, 1.0, 0.7, 0.1,
                                              help="🎨 **多样性阈值**: 控制结果差异程度。0.7确保适度差异。")
                consistency_weight = st.slider("一致性权重", 0.0, 1.0, 0.3, 0.1,
                                             help="⚖️ **一致性权重**: 稳定性在排序中的重要性。")
            
            # Set optimal default values for reranker parameters
            if reranker_type not in ["simple"]:
                simple_reranker_threshold = 0.2
            if reranker_type not in ["cross_encoder"]:
                cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                cross_encoder_threshold = 0.2
            if reranker_type not in ["enhanced_cross_encoder"]:
                enable_ensemble = True
                enable_consistency_check = True
                enable_diversity = True
                enable_context_aware = True
                diversity_threshold = 0.7
                consistency_weight = 0.3

        with col3:
            st.subheader("🎨 分析与可视化")
            auto_perform_analysis = st.checkbox("搜索后自动分析", value=False,
                                                  help="✅ **自动分析**: 搜索完成后立即自动执行下方选定的分析。如果禁用，则需要手动点击按钮开始分析。")
            perform_literary_analysis = st.checkbox("文学分析", value=True,
                                                   help="📚 **文学分析**: 分析人物关系、主题等。推荐文学作品启用。")
            if perform_literary_analysis:
                min_cooccurrence = st.slider("最小共现频率", 1, 10, 2,
                                           help="🔗 **共现频率**: 人物/概念同时出现的最少次数。2捕获有意义关系。")
            else:
                min_cooccurrence = 2
                
            perform_theme_analysis = st.checkbox("主题分析", value=True,
                                                help="🎭 **主题分析**: 识别文本主题和情感。有助于深度理解。")
            perform_visualization = st.checkbox("生成可视化图表", value=True,
                                              help="📊 **可视化**: 生成图表帮助理解分析结果。")
            
            # 保存可视化配置到 session_state
            st.session_state.perform_visualization = perform_visualization
            st.session_state.perform_theme_analysis = perform_theme_analysis
            
            # 保存文本处理设置到 session_state
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap  
            st.session_state.remove_stopwords = remove_stopwords
            
            if perform_visualization:
                wordcloud_max_words = st.slider("词云最大词数", 50, 300, 100,
                                               help="☁️ **词云词数**: 词云图显示的词语数量。100个便于观察。")
                heatmap_top_n = st.slider("热力图显示数量", 5, 20, 8,
                                        help="🔥 **热力图**: 热力图显示的主要元素数量。8个清晰易读。")
                # Additional visualization options with Chinese
                plot_style = st.selectbox("图表样式", ["whitegrid", "darkgrid", "white", "dark"], index=0,
                                        format_func=lambda x: {
                                            "whitegrid": "白色网格 (推荐)",
                                            "darkgrid": "深色网格", 
                                            "white": "纯白背景",
                                            "dark": "深色背景"
                                        }.get(x, x),
                                        help="🎨 **图表样式**: 可视化图表的外观风格。")
                figure_size = st.selectbox("图表大小", ["small", "medium", "large"], index=1,
                                         format_func=lambda x: {"small": "小", "medium": "中 (推荐)", "large": "大"}.get(x, x),
                                         help="📏 **图表大小**: 生成图表的尺寸。中等适合屏幕显示。")
                chart_language = st.selectbox("图表语言", ["zh", "en"], index=0,
                                            format_func=lambda x: {"zh": "中文", "en": "English"}.get(x, x),
                                            help="🌐 **图表语言**: 选择图表标题和标签的显示语言。")
            else:
                wordcloud_max_words = 100
                heatmap_top_n = 8
                plot_style = "whitegrid"
                figure_size = "medium"
                chart_language = "zh"
                
            # 保存可视化参数到 session_state
            st.session_state.wordcloud_max_words = wordcloud_max_words
            st.session_state.heatmap_top_n = heatmap_top_n
            st.session_state.plot_style = plot_style
            st.session_state.figure_size = figure_size
            st.session_state.chart_language = chart_language

            st.markdown("**⚙️ 系统设置**")
            cache_expire_days = st.slider("缓存保留天数", 1, 30, 7,
                                        help="💾 **缓存**: 保存处理结果的天数。7天平衡性能和存储。")
            
            # Text processing options with Chinese
            st.markdown("**📝 文本处理**")
            chunk_size = st.slider("文本块大小", 500, 2000, 800,
                                 help="📄 **文本块**: 处理文本的分段大小。800字适合中文分析。")
            chunk_overlap = st.slider("文本块重叠", 0, 200, 80,
                                    help="🔗 **重叠**: 文本块间的重叠字数。80字保持上下文连续性。")
            remove_stopwords = st.checkbox("移除停用词", value=False,
                                         help="🚫 **停用词**: 移除'的、了、在'等常用词。文学分析不建议启用。")

    # --- Search Button ---
    search_button = st.button("🚀 开始搜索", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Instructions and Status ---
    if not uploaded_file or not query:
        st.markdown("---")
        st.header("📝 使用指南")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📁 第一步: 上传PDF文件")
            st.markdown("""
            - 点击上方的"浏览文件"选择您的PDF文档
            - 支持的文件类型: 仅PDF
            - 文件将自动处理
            """)
        with col2:
            st.subheader("🔍 第二步: 输入搜索关键词")
            st.markdown("""
            - 在关键词字段中输入您的搜索词
            - 例如: "爱情", "死亡", "权力", "野心"
            - 多个关键词将自动扩展
            """)
        
        st.subheader("⚙️ 第三步: 配置设置 (可选)")
        st.markdown("""
        - 展开"高级搜索配置"来调整参数
        - 默认设置已针对大多数情况优化
        - 根据需要启用/禁用分析和可视化功能
        """)
        
        st.subheader("🚀 第四步: 点击'开始搜索'进行分析")
        
        # Show sample data info
        st.markdown("---")
        st.markdown("### 📚 可用示例数据")
        data_dir = "data"
        if os.path.exists(data_dir):
            sample_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
            if sample_files:
                st.success(f"在data目录中找到 {len(sample_files)} 个示例PDF文件:")
                for file in sample_files:
                    st.markdown(f"- `{file}`")
                st.info("您可以使用这些示例文件测试应用程序。")
            else:
                st.warning("在data目录中未找到示例PDF文件。")
        else:
            st.info("要使用示例数据进行测试，请将PDF文件放在'data'目录中。")

    # --- Main Content ---
    if uploaded_file and query and search_button:
        # Create progress container
        progress_container = st.container()
        
        try:
            with st.spinner("正在处理..."):
                with progress_container:
                    st.markdown("### 🔄 处理进度")
                    progress_placeholder = st.empty()
                    
                    def update_progress(message):
                        try:
                            with progress_placeholder.container():
                                st.info(f"ℹ️ {message}")
                        except Exception as e:
                            st.error(f"进度更新错误: {e}")
                            print(f"进度更新错误: {e}")
                    
                    # Process PDF
                    update_progress("📄 正在处理PDF文档...")
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            update_progress("📄 临时文件创建成功...")
                            # Pass text processing settings to process_pdf
                            # Use session state values if available, otherwise use current values
                            current_chunk_size = st.session_state.get('chunk_size', chunk_size)
                            current_chunk_overlap = st.session_state.get('chunk_overlap', chunk_overlap)
                            current_remove_stopwords = st.session_state.get('remove_stopwords', remove_stopwords)
                            
                            texts = process_pdf(tmp_file.name, 
                                              chunk_size=current_chunk_size, 
                                              chunk_overlap=current_chunk_overlap, 
                                              remove_stopwords=current_remove_stopwords)
                            update_progress(f"📄 PDF解析完成 - 使用设置: 块大小={current_chunk_size}, 重叠={current_chunk_overlap}, 移除停用词={current_remove_stopwords}")
                        os.unlink(tmp_file.name)
                        update_progress("📄 临时文件清理完成...")
                    except Exception as e:
                        st.error(f"PDF处理失败: {e}")
                        traceback.print_exc()
                        return

                    if not texts:
                        st.error("无法从PDF中提取文本。")
                        return

                    update_progress(f"✅ PDF处理成功 - 提取了 {len(texts)} 个文本块")

                    # Keyword Expansion
                    final_query = query
                    if use_keyword_expansion:
                        update_progress(f"🔍 正在扩展关键词: '{query}'...")
                        try:
                            expander = KeywordExpander(method=expander_method, document_type=document_type)
                            update_progress(f"🔍 KeywordExpander初始化成功...")
                            expanded_keywords = expander.expand_keywords(query.split(), max_synonyms_per_word=max_synonyms, max_related_per_word=max_related_words, semantic_threshold=semantic_threshold, use_hierarchical=use_hierarchical)
                            final_query = " ".join(expanded_keywords.keys())
                            st.session_state.expanded_keywords = expanded_keywords
                            update_progress(f"✅ 关键词扩展完成: {len(query.split())} → {len(expanded_keywords)} 个关键词")
                        except Exception as e:
                            update_progress(f"❌ 关键词扩展失败: {e}")
                            traceback.print_exc()
                            final_query = query  # 使用原始查询
                    
                    # Search with retry mechanism
                    if enable_retry:
                        # 先尝试主要搜索方法
                        if use_hybrid_search:
                            update_progress("🔄 初始化混合搜索引擎...")
                            from modules.retriever import EmbeddingRetriever, BM25Retriever
                            from modules.hybrid_search import create_hybrid_search_engine

                            embedding_retriever = EmbeddingRetriever(model_path)
                            bm25_retriever = BM25Retriever()
                            hybrid_search_engine = create_hybrid_search_engine(
                                embedding_retriever=embedding_retriever,
                                bm25_retriever=bm25_retriever,
                                fusion_method=fusion_method,
                                rrf_k=rrf_k,
                                weights={"bm25": bm25_weight, "embedding": embedding_weight},
                                enable_parallel=enable_parallel
                            )
                            
                            # 计算目标搜索结果数量，确保至少能满足最终需求
                            target_search_results = max(initial_max_results, final_max_results)
                            
                            update_progress(f"🚀 开始混合搜索，使用 {len(final_query.split())} 个关键词...")
                            results = hybrid_search_engine.search([{'content': t} for t in texts], final_query.split(), k=target_search_results, min_results=min_results)
                        else:
                            update_progress("🔍 执行标准语义搜索...")
                            results = standard_search(final_query, texts, model_path)

                        update_progress(f"🎯 初始搜索完成 - 找到 {len(results)} 个候选结果 (目标: {initial_max_results})")
                        
                        # 检查是否需要重试
                        filtered_results = [r for r in results if r.get('score', 0.0) >= similarity_threshold]
                        
                        # 计算目标结果数量：确保不少于min_results，但如果用户设置了更大的final_max_results，则使用更大的值
                        target_results = max(min_results, final_max_results)
                        
                        if len(filtered_results) < target_results:
                            update_progress(f"🔄 结果不足({len(filtered_results)}<{target_results})，启动智能重试...")
                            retry_results = retry_search_with_fallbacks(
                                query=final_query,
                                texts=texts,
                                model_path=model_path,
                                target_count=target_results,
                                similarity_threshold=similarity_threshold,
                                update_progress=update_progress
                            )
                            
                            # 合并主搜索和重试结果，去重
                            existing_indices = {r.get('index', -1) for r in filtered_results}
                            new_retry_results = [r for r in retry_results if r.get('index', -1) not in existing_indices]
                            
                            filtered_results.extend(new_retry_results)
                            update_progress(f"✅ 重试完成 - 总结果数: {len(filtered_results)}")
                        else:
                            update_progress(f"✅ 初始搜索已满足要求 - {len(filtered_results)} 个结果")
                            
                    else:
                        # 传统搜索方式（不使用重试）
                        if use_hybrid_search:
                            update_progress("🔄 初始化混合搜索引擎...")
                            from modules.retriever import EmbeddingRetriever, BM25Retriever
                            from modules.hybrid_search import create_hybrid_search_engine

                            embedding_retriever = EmbeddingRetriever(model_path)
                            bm25_retriever = BM25Retriever()
                            hybrid_search_engine = create_hybrid_search_engine(
                                embedding_retriever=embedding_retriever,
                                bm25_retriever=bm25_retriever,
                                fusion_method=fusion_method,
                                rrf_k=rrf_k,
                                weights={"bm25": bm25_weight, "embedding": embedding_weight},
                                enable_parallel=enable_parallel
                            )
                            
                            # 计算目标搜索结果数量，确保至少能满足最终需求
                            target_search_results = max(initial_max_results, final_max_results)
                            
                            update_progress(f"🚀 开始混合搜索，使用 {len(final_query.split())} 个关键词...")
                            results = hybrid_search_engine.search([{'content': t} for t in texts], final_query.split(), k=target_search_results, min_results=min_results)
                        else:
                            update_progress("🔍 执行标准语义搜索...")
                            results = standard_search(final_query, texts, model_path)

                        update_progress(f"🎯 搜索完成 - 找到 {len(results)} 个候选结果 (目标: {initial_max_results})")

                        # Filtering results
                        filtered_results = [r for r in results if r.get('score', 0.0) >= similarity_threshold]
                        
                        # 计算目标结果数量：确保不少于min_results，但如果用户设置了更大的final_max_results，则使用更大的值
                        target_results = max(min_results, final_max_results)
                        
                        # 传统的最少返回结果机制
                        if len(filtered_results) < target_results and len(results) > 0:
                            update_progress(f"🔧 结果不足({len(filtered_results)}<{target_results})，启动基础补充机制...")
                            # 按分数排序，取前target_results个
                            sorted_results = sorted(results, key=lambda x: x.get('score', 0.0), reverse=True)
                            filtered_results = sorted_results[:min(target_results, len(sorted_results))]
                            
                            # 如果还是不够，而且有文本，就随机选择一些
                            if len(filtered_results) < target_results and len(texts) >= target_results:
                                update_progress(f"🎲 随机补充结果以达到目标数量({target_results})...")
                                # 创建基础结果
                                import random
                                available_indices = list(range(len(texts)))
                                used_indices = {r.get('index', -1) for r in filtered_results}
                                remaining_indices = [i for i in available_indices if i not in used_indices]
                                
                                # 随机选择补充结果
                                needed = target_results - len(filtered_results)
                                if len(remaining_indices) >= needed:
                                    selected_indices = random.sample(remaining_indices, needed)
                                else:
                                    selected_indices = remaining_indices
                                if len(remaining_indices) >= needed:
                                    selected_indices = random.sample(remaining_indices, needed)
                                else:
                                    selected_indices = remaining_indices
                                
                                for idx in selected_indices:
                                    filtered_results.append({
                                        'text': texts[idx],
                                        'content': texts[idx],
                                        'score': 0.1,  # 低分数表示这是补充结果
                                        'page_num': idx + 1,
                                        'index': idx,
                                        'is_fallback': True  # 标记为补充结果
                                    })
                                
                                update_progress(f"📝 已补充 {len(selected_indices)} 个结果，总计 {len(filtered_results)} 个")
                
                # Show search statistics with score details
                if results:
                    scores = [r.get('score', 0.0) for r in results]
                    update_progress(f"📊 分数分析: 范围 {min(scores):.4f} - {max(scores):.4f}, 平均: {sum(scores)/len(scores):.4f}")
                    update_progress(f"✅ 过滤后 (阈值 {similarity_threshold:.3f}): {len(filtered_results)} 个结果")
                    if len(filtered_results) == 0 and len(results) > 0:
                        st.warning(f"⚠️ 所有结果都被过滤掉了。请尝试将相似度阈值降低到 {min(scores):.4f} 以下")

                # Reranking
                if reranker_type != "none" and filtered_results:
                    update_progress(f"🔄 应用{reranker_type.replace('_', ' ').title()}重新排序...")
                    try:
                        reranker_params = {"keywords": final_query.split()}
                        if reranker_type == "simple":
                            reranker_params["threshold"] = simple_reranker_threshold
                        elif reranker_type == "cross_encoder":
                            reranker_params["model_name"] = cross_encoder_model
                            reranker_params["threshold"] = cross_encoder_threshold
                        elif reranker_type == "enhanced_cross_encoder":
                            reranker_params["enable_ensemble"] = enable_ensemble
                            reranker_params["enable_consistency_check"] = enable_consistency_check
                            reranker_params["enable_diversity"] = enable_diversity
                            reranker_params["enable_context_aware"] = enable_context_aware

                        reranker = create_reranker(reranker_type, **reranker_params)
                        candidates = [{'content': r.get('text', r.get('content', '')), 'score': r.get('score', 0.0), 
                                     'page_num': r.get('page_num', 1), 'index': r.get('index', i)} 
                                     for i, r in enumerate(filtered_results)]
                        
                        update_progress(f"🔄 开始重排 {len(candidates)} 个候选结果...")
                        reranked_candidates = reranker.rerank(candidates, k=final_max_results)
                        update_progress(f"✅ 重排完成，获得 {len(reranked_candidates)} 个结果")
                        
                        final_results = [{'text': c['content'], 'score': c.get('rerank_score', c.get('score', 0)),
                                        'page_num': c.get('page_num', 1), 'index': c.get('index', i)} 
                                        for i, c in enumerate(reranked_candidates)]
                        update_progress(f"✅ 重新排序完成 - 从 {len(candidates)} 个候选中选出 {len(final_results)} 个最终结果")
                        
                    except Exception as e:
                        update_progress(f"❌ 重排失败: {str(e)}")
                        print(f"重排错误详情: {e}")
                        traceback.print_exc()
                        # 如果重排失败，使用原始结果
                        final_results = []
                        for r in sorted(filtered_results, key=lambda x: x.get('score', 0.0), reverse=True)[:final_max_results]:
                            final_results.append({
                                'text': r.get('text', r.get('content', '')),
                                'score': r.get('score', 0.0),
                                'page_num': r.get('page_num', 1),
                                'index': r.get('index', len(final_results))
                            })
                        update_progress(f"🔄 使用原始排序结果，共 {len(final_results)} 个")
                else:
                    # Ensure consistent format for results without reranking
                    final_results = []
                    for r in sorted(filtered_results, key=lambda x: x.get('score', 0.0), reverse=True)[:final_max_results]:
                        final_results.append({
                            'text': r.get('text', r.get('content', '')),
                            'score': r.get('score', 0.0),
                            'page_num': r.get('page_num', 1),
                            'index': r.get('index', len(final_results))
                        })
                
                # Normalize scores for better user experience (show 30-100 range)
                if final_results:
                    raw_scores = [r.get('score', 0.0) for r in final_results]
                    if max(raw_scores) > min(raw_scores):
                        # Normalize to 30-100 scale for better user experience
                        min_score, max_score = min(raw_scores), max(raw_scores)
                        for result in final_results:
                            raw_score = result.get('score', 0.0)
                            # Scale from original range to 30-100 range
                            normalized_score = 30 + ((raw_score - min_score) / (max_score - min_score)) * 70
                            result['score'] = normalized_score
                            # Store original raw score for debugging if needed
                            result['raw_score'] = raw_score
                    else:
                        # All scores are the same, set to high score
                        for result in final_results:
                            result['score'] = 85.0  # Set to a good score when all are equal
                            result['raw_score'] = result.get('score', 0.0)
                    
                    update_progress(f"🎯 分数归一化完成 - 已分配30-100分数范围")
                else:
                    # 如果没有结果，创建空列表
                    final_results = []
                    update_progress("ℹ️ 没有找到符合条件的结果")
                
                # 保存搜索结果到 session_state（这是关键！）
                st.session_state.results = final_results
                st.session_state.texts = texts
                st.session_state.query = query
                
                # 调试信息
                if final_results:
                    update_progress(f"🎯 已保存 {len(final_results)} 个搜索结果到 session_state")
                    print(f"DEBUG: final_results 长度: {len(final_results)}")
                    print(f"DEBUG: 第一个结果: {final_results[0] if final_results else '无'}")
                else:
                    update_progress("ℹ️ 没有搜索结果保存到 session_state")
                    print("DEBUG: final_results 为空")
                
                update_progress("✅ 搜索处理完成!")

                # 根据用户选择，决定是否立即执行分析
                if auto_perform_analysis:
                    update_progress("🤖 已启用自动分析，开始执行...")
                    try:
                        # 使用完整的文本块进行分析
                        full_text_chunks = [{'content': text, 'page_num': i+1} for i, text in enumerate(texts)]
                        analysis_results = {}

                        if perform_theme_analysis:
                            theme_analyzer = LiteraryThemeAnalyzer()
                            analysis_results['theme_analysis'] = theme_analyzer.analyze_text_themes(
                                full_text_chunks, progress_callback=lambda msg: update_progress(f"🎨 {msg}")
                            )
                        
                        if perform_literary_analysis:
                            literary_analyzer = LiteraryAnalyzer()
                            comprehensive_results = literary_analyzer.generate_comprehensive_analysis(
                                full_text_chunks, progress_callback=lambda msg: update_progress(f"📚 {msg}")
                            )
                            analysis_results['character_analysis'] = comprehensive_results['characters']
                            analysis_results['emotion_analysis'] = comprehensive_results['emotions']
                            analysis_results['narrative_analysis'] = comprehensive_results['narrative']

                        st.session_state.analysis_results = analysis_results
                        update_progress("✅ 自动分析完成!")

                    except Exception as e:
                        update_progress(f"❌ 自动分析失败: {e}")
                        traceback.print_exc()
                else:
                    # 清除旧的分析结果，等待用户手动触发
                    if 'analysis_results' in st.session_state:
                        del st.session_state['analysis_results']
                    update_progress("ℹ️ 自动分析已禁用。请在下方手动开始分析。")
                
                update_progress("🎉 所有处理完成! 请查看下方结果。")
                
        except Exception as e:
            st.error(f"❌ 处理过程中发生错误: {str(e)}")
            st.error(f"❌ 错误类型: {type(e).__name__}")
            st.code(traceback.format_exc())
            print(f"全局错误详情: {e}")
            traceback.print_exc()

    # --- Results, Analysis, and Visualization Display ---
    if 'results' in st.session_state:
        print(f"DEBUG: 在 session_state 中找到 results，长度: {len(st.session_state.results) if st.session_state.results else '空或None'}")
        
        # Enhanced results header
        st.markdown("---")
        st.header("🎯 搜索结果")
        
        display_results(st.session_state.results)
        
    # --- Analysis Section (Always available after search or when texts are loaded) ---
    if 'texts' in st.session_state:
        st.markdown("---")
        
        # Enhanced analysis header
        st.header("🎭 文学分析")
        st.info("📝 注意：文学分析是针对整个文档进行的，以提供更全面的见解。")

        # Enhanced analysis options with cool layout
        st.subheader("选择分析类型")
        
        # Create columns for analysis options
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_options = {
                'theme_analysis': st.checkbox("🎨 深入主题分析", value=True, help="分析文本的主题、情感和文学手法。"),
                'character_analysis': st.checkbox("👥 人物关系分析", value=True, help="分析人物共现、频率和关系网络。"),
            }
        
        with col2:
            analysis_options.update({
                'emotion_analysis': st.checkbox("💭 情感倾向分析", value=True, help="分析文本中的情感分布和强度。"),
                'narrative_analysis': st.checkbox("📖 叙事结构分析", value=True, help="分析场景分布和叙事节奏。")
            })

        if st.button("🚀 开始进行文学分析", key="start_analysis"):
            # 检查是否有文本数据
            if 'texts' not in st.session_state:
                st.error("❌ 没有找到文本数据。请先执行搜索。")
            else:
                with st.spinner("正在执行文学分析... 请稍候..."):
                    try:
                        analysis_results = {}
                        # 使用完整的文本块进行分析
                        full_text_chunks = [{'content': text, 'page_num': i+1} for i, text in enumerate(st.session_state.texts)]
                        
                        if analysis_options['theme_analysis']:
                            st.info("🎨 开始主题分析...")
                            theme_analyzer = LiteraryThemeAnalyzer()
                            theme_result = theme_analyzer.analyze_text_themes(
                                full_text_chunks, progress_callback=st.info
                            )
                            analysis_results['theme_analysis'] = theme_result
                            st.success(f"✅ 主题分析完成！结果类型: {type(theme_result)}")
                        
                        # LiteraryAnalyzer可以一次性处理多种分析
                        if analysis_options['character_analysis'] or analysis_options['emotion_analysis'] or analysis_options['narrative_analysis']:
                            st.info("👥 开始文学分析...")
                            literary_analyzer = LiteraryAnalyzer()
                            comprehensive_results = literary_analyzer.generate_comprehensive_analysis(
                                full_text_chunks, progress_callback=st.info
                            )
                            st.success(f"✅ 文学分析完成！结果类型: {type(comprehensive_results)}")
                            
                            if analysis_options['character_analysis']:
                                analysis_results['character_analysis'] = comprehensive_results['characters']
                            if analysis_options['emotion_analysis']:
                                analysis_results['emotion_analysis'] = comprehensive_results['emotions']
                            if analysis_options['narrative_analysis']:
                                analysis_results['narrative_analysis'] = comprehensive_results['narrative']
                            
                            # 存储完整的分析结果用于可视化
                            analysis_results['comprehensive_results'] = comprehensive_results

                        # 保存结果并显示摘要
                        st.session_state.analysis_results = analysis_results
                        
                        # 显示分析摘要
                        st.success("🎉 文学分析完成！")
                        st.info(f"📊 生成了 {len(analysis_results)} 种分析结果")
                        for key in analysis_results.keys():
                            if key == 'comprehensive_results':
                                continue  # 跳过综合结果的显示
                            result_data = analysis_results[key]
                            if result_data:
                                if isinstance(result_data, dict):
                                    st.write(f"✅ {key}: {len(result_data)} 个数据项")
                                elif isinstance(result_data, list):
                                    st.write(f"✅ {key}: {len(result_data)} 个条目")
                                else:
                                    st.write(f"✅ {key}: 已完成 (类型: {type(result_data)})")
                            else:
                                st.warning(f"⚠️ {key}: 无数据")

                    except Exception as e:
                        st.error(f"文学分析失败: {e}")
                        st.code(traceback.format_exc())

        # 显示分析结果
        if 'analysis_results' in st.session_state:
            st.header("📊 分析结果")
            results = st.session_state.analysis_results
            
            # 添加可视化部分
            if results.get('comprehensive_results'):
                st.subheader("📈 文学分析可视化")
                comprehensive_data = results['comprehensive_results']
                
                try:
                    # 使用用户选择的语言创建可视化器
                    chart_lang = st.session_state.get('chart_language', 'zh')
                    visualizer = AdvancedVisualizer(language=chart_lang)
                    
                    # 生成可视化图表
                    fig = visualizer.plot_literary_analysis(comprehensive_data)
                    if fig:
                        st.pyplot(fig)
                        st.success("✅ 可视化图表生成成功！")
                        
                        # 提供图表说明
                        st.info("""
                        📊 **图表说明**：
                        - **左上**: 主要人物出现频率，显示作品中最重要的角色
                        - **右上**: 主题分布饼图，展示文学作品的主要主题比例
                        - **左中**: 情感倾向分析，显示作品中积极、消极和中性情感的比例
                        - **右中**: 人物共现关系，展示哪些人物经常一起出现
                        - **左下**: 内容在各页面的分布，帮助理解叙事结构
                        - **右下**: 关键统计信息汇总
                        """)
                    else:
                        st.warning("⚠️ 可视化图表生成失败")
                        
                except Exception as e:
                    st.error(f"可视化生成失败: {e}")
                    st.code(traceback.format_exc())
            
            if results.get('theme_analysis'):
                with st.expander("🎨 深入主题分析结果", expanded=False):
                    st.write("**分析完成！以下是详细结果：**")
                    # 尝试更友好的显示方式
                    theme_data = results['theme_analysis']
                    if isinstance(theme_data, dict):
                        for key, value in theme_data.items():
                            st.subheader(f"📌 {key}")
                            if isinstance(value, (dict, list)):
                                st.json(value)
                            else:
                                st.write(value)
                    else:
                        st.json(theme_data)

            if results.get('character_analysis'):
                with st.expander("👥 人物关系分析结果", expanded=False):
                    st.write("**分析完成！以下是详细结果：**")
                    char_data = results['character_analysis']
                    if isinstance(char_data, dict):
                        for key, value in char_data.items():
                            st.subheader(f"👤 {key}")
                            if isinstance(value, (dict, list)):
                                st.json(value)
                            else:
                                st.write(value)
                    else:
                        st.json(char_data)

            if results.get('emotion_analysis'):
                with st.expander("💭 情感倾向分析结果", expanded=False):
                    st.write("**分析完成！以下是详细结果：**")
                    emotion_data = results['emotion_analysis']
                    if isinstance(emotion_data, dict):
                        for key, value in emotion_data.items():
                            st.subheader(f"😊 {key}")
                            if isinstance(value, (dict, list)):
                                st.json(value)
                            else:
                                st.write(value)
                    else:
                        st.json(emotion_data)

            if results.get('narrative_analysis'):
                with st.expander("📖 叙事结构分析结果", expanded=False):
                    st.write("**分析完成！以下是详细结果：**")
                    narrative_data = results['narrative_analysis']
                    if isinstance(narrative_data, dict):
                        for key, value in narrative_data.items():
                            st.subheader(f"📚 {key}")
                            if isinstance(value, (dict, list)):
                                st.json(value)
                            else:
                                st.write(value)
                    else:
                        st.json(narrative_data)
            
            # 添加调试信息
            with st.expander("🔧 调试信息 (开发用)", expanded=False):
                st.write("分析结果的数据类型和内容：")
                st.write(f"数据类型: {type(results)}")
                st.write(f"键值: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
                st.json(results)
    else:
        print("DEBUG: session_state 中没有找到 results 或 texts")

    # --- Enhanced visualizations section (Always check after results/analysis) ---
    if st.session_state.get('perform_visualization', False) and 'results' in st.session_state:
        st.markdown("---")
        st.header("📊 可视化图表")
        st.info("通过精美的图表深入理解搜索结果和文本模式")
        
        with st.spinner("🎨 正在生成精美的可视化图表..."):
            try:
                # 使用用户选择的语言创建可视化器
                chart_lang = st.session_state.get('chart_language', 'zh')
                visualizer = AdvancedVisualizer(language=chart_lang)
                
                # 安全地获取关键词
                expanded_keywords = st.session_state.get('expanded_keywords', {})
                query_keywords = st.session_state.get('query', '').split() if st.session_state.get('query') else []
                keywords_for_viz = list(expanded_keywords.keys()) or query_keywords
                
                # 如果没有关键词，跳过可视化
                if not keywords_for_viz:
                    st.warning("⚠️ 没有找到搜索关键词，无法生成可视化图表。")
                else:
                    results_for_viz = st.session_state.results.copy()
                    
                    # 为可视化添加found_keywords字段
                    search_keywords = keywords_for_viz
                    for result in results_for_viz:
                        if 'found_keywords' not in result:
                            # 简单匹配：检查哪些搜索关键词在结果文本中
                            text = result.get('text', '').lower()
                            found = []
                            for kw in search_keywords:
                                if kw.lower() in text:
                                    # 计算出现次数
                                    count = text.count(kw.lower())
                                    found.extend([kw] * count)
                            result['found_keywords'] = found

                    # 从 session_state 获取可视化参数
                    wordcloud_max_words = st.session_state.get('wordcloud_max_words', 100)
                    heatmap_top_n = st.session_state.get('heatmap_top_n', 8)

                    # Generate charts in a grid layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 生成图表并显示
                        fig1 = visualizer.plot_page_distribution(results_for_viz, keywords_for_viz)
                        if fig1:
                            st.pyplot(fig1)
                            
                        fig3 = visualizer.plot_word_cloud(results_for_viz, keywords_for_viz, max_words=wordcloud_max_words)
                        if fig3:
                            st.pyplot(fig3)
                    
                    with col2:
                        try:
                            print("DEBUG: 开始生成主题频率图")
                            fig2 = visualizer.plot_theme_frequency(results_for_viz, keywords_for_viz, top_n=heatmap_top_n)
                            if fig2:
                                st.pyplot(fig2)
                                print("DEBUG: 主题频率图显示成功")
                            else:
                                st.error("主题频率图生成失败")
                        except Exception as e:
                            print(f"DEBUG: 主题频率图生成错误: {e}")
                            st.error(f"主题频率图生成失败: {e}")
                            
                        try:
                            print("DEBUG: 开始生成共现热力图")
                            fig4 = visualizer.plot_cooccurrence_heatmap(results_for_viz, keywords_for_viz, top_n=heatmap_top_n)
                            if fig4:
                                st.pyplot(fig4)
                                print("DEBUG: 共现热力图显示成功")
                            else:
                                st.error("共现热力图生成失败")
                        except Exception as e:
                            print(f"DEBUG: 共现热力图生成错误: {e}")
                            import traceback
                            traceback.print_exc()
                            st.error(f"共现热力图生成失败: {e}")
                            # 显示错误详情
                            with st.expander("错误详情", expanded=False):
                                st.code(traceback.format_exc())
            
            except Exception as e:
                st.error(f"可视化生成失败: {e}")
                st.code(traceback.format_exc())

def retry_search_with_fallbacks(query, texts, model_path, target_count=3, similarity_threshold=0.3, update_progress=None):
    """
    智能重试搜索机制 - 当结果不足时尝试多种搜索策略
    
    Args:
        query: 搜索查询
        texts: 文本列表
        model_path: 模型路径
        target_count: 目标结果数量
        similarity_threshold: 相似度阈值
        update_progress: 进度更新函数
    
    Returns:
        搜索结果列表
    """
    def log_progress(msg):
        if update_progress:
            update_progress(msg)
        else:
            print(msg)
    
    all_results = []
    used_methods = []
    
    # 策略1: 标准语义搜索
    try:
        log_progress("🔄 尝试策略1: 标准语义搜索...")
        results = standard_search(query, texts, model_path)
        filtered_results = [r for r in results if r.get('score', 0.0) >= similarity_threshold]
        
        if len(filtered_results) >= target_count:
            log_progress(f"✅ 策略1成功: 找到 {len(filtered_results)} 个结果")
            return filtered_results[:target_count]
        
        all_results.extend(filtered_results)
        used_methods.append("语义搜索")
        log_progress(f"🔍 策略1部分成功: {len(filtered_results)} 个结果，继续尝试...")
        
    except Exception as e:
        log_progress(f"⚠️ 策略1失败: {e}")
    
    # 策略2: 降低相似度阈值的语义搜索
    if len(all_results) < target_count:
        try:
            log_progress("🔄 尝试策略2: 降低阈值的语义搜索...")
            lower_threshold = max(0.1, similarity_threshold * 0.5)
            results = standard_search(query, texts, model_path)
            filtered_results = [r for r in results if r.get('score', 0.0) >= lower_threshold]
            
            # 去重并合并结果
            existing_indices = {r.get('index', -1) for r in all_results}
            new_results = [r for r in filtered_results if r.get('index', -1) not in existing_indices]
            
            all_results.extend(new_results)
            used_methods.append("低阈值语义搜索")
            log_progress(f"🔍 策略2: 新增 {len(new_results)} 个结果，当前总计 {len(all_results)} 个")
            
        except Exception as e:
            log_progress(f"⚠️ 策略2失败: {e}")
    
    # 策略3: BM25关键词搜索
    if len(all_results) < target_count:
        try:
            log_progress("🔄 尝试策略3: BM25关键词搜索...")
            from modules.retriever import BM25Retriever
            
            bm25_retriever = BM25Retriever()
            bm25_retriever.keywords = query.split()
            
            text_chunks = [{'content': text, 'page_num': i+1, 'id': f'chunk_{i}'} for i, text in enumerate(texts)]
            bm25_results = bm25_retriever.retrieve(text_chunks, len(texts))
            
            # 转换格式并去重
            existing_indices = {r.get('index', -1) for r in all_results}
            new_results = []
            
            for result in bm25_results:
                if result.get('index', -1) not in existing_indices:
                    new_results.append({
                        'text': result.get('content', ''),
                        'content': result.get('content', ''),
                        'score': result.get('bm25_score', 0.0) * 10,  # 调整分数范围
                        'page_num': result.get('page_num', 1),
                        'index': result.get('index', -1),
                        'method': 'BM25'
                    })
            
            all_results.extend(new_results)
            used_methods.append("BM25搜索")
            log_progress(f"🔍 策略3: 新增 {len(new_results)} 个结果，当前总计 {len(all_results)} 个")
            
        except Exception as e:
            log_progress(f"⚠️ 策略3失败: {e}")
    
    # 策略4: 关键词扩展后重新搜索
    if len(all_results) < target_count:
        try:
            log_progress("🔄 尝试策略4: 关键词扩展搜索...")
            from modules.keyword_expander import KeywordExpander
            
            expander = KeywordExpander(method='wordnet', document_type='literary')
            expanded_keywords = expander.expand_keywords(query.split(), max_synonyms_per_word=2, max_related_per_word=1)
            expanded_query = " ".join(expanded_keywords.keys())
            
            results = standard_search(expanded_query, texts, model_path)
            lower_threshold = max(0.05, similarity_threshold * 0.3)
            filtered_results = [r for r in results if r.get('score', 0.0) >= lower_threshold]
            
            # 去重并合并结果
            existing_indices = {r.get('index', -1) for r in all_results}
            new_results = [r for r in filtered_results if r.get('index', -1) not in existing_indices]
            
            all_results.extend(new_results)
            used_methods.append("扩展关键词搜索")
            log_progress(f"🔍 策略4: 新增 {len(new_results)} 个结果，当前总计 {len(all_results)} 个")
            
        except Exception as e:
            log_progress(f"⚠️ 策略4失败: {e}")
    
    # 策略5: 简单文本匹配（最后手段）
    if len(all_results) < target_count:
        try:
            log_progress("🔄 尝试策略5: 简单文本匹配...")
            query_words = query.lower().split()
            existing_indices = {r.get('index', -1) for r in all_results}
            new_results = []
            
            for i, text in enumerate(texts):
                if i not in existing_indices:
                    text_lower = text.lower()
                    score = 0
                    for word in query_words:
                        score += text_lower.count(word)
                    
                    if score > 0:  # 只要包含任何查询词就算匹配
                        # 归一化分数
                        max_possible_score = len(query_words) * max(1, len(text.split()) // 20)
                        normalized_score = min(score / max_possible_score if max_possible_score > 0 else 0, 1.0)
                        
                        new_results.append({
                            'text': text,
                            'content': text,
                            'score': normalized_score * 100,  # 转换为百分制
                            'page_num': i + 1,
                            'index': i,
                            'method': '文本匹配'
                        })
            
            # 按分数排序并取最好的结果
            new_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            needed = target_count - len(all_results)
            all_results.extend(new_results[:needed])
            used_methods.append("文本匹配")
            log_progress(f"🔍 策略5: 新增 {len(new_results[:needed])} 个结果，当前总计 {len(all_results)} 个")
            
        except Exception as e:
            log_progress(f"⚠️ 策略5失败: {e}")
    
    # 最后的随机补充（如果还是不够）
    if len(all_results) < target_count and len(texts) > len(all_results):
        log_progress("🎲 最终策略: 随机补充...")
        import random
        
        existing_indices = {r.get('index', -1) for r in all_results}
        available_indices = [i for i in range(len(texts)) if i not in existing_indices]
        
        needed = target_count - len(all_results)
        if len(available_indices) >= needed:
            selected_indices = random.sample(available_indices, needed)
        else:
            selected_indices = available_indices
        
        for idx in selected_indices:
            all_results.append({
                'text': texts[idx],
                'content': texts[idx],
                'score': 0.1,
                'page_num': idx + 1,
                'index': idx,
                'is_fallback': True,
                'method': '随机补充'
            })
        
        used_methods.append("随机补充")
        log_progress(f"📝 随机补充: 新增 {len(selected_indices)} 个结果")
    
    # 排序结果（按分数降序）
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # 总结报告
    log_progress(f"🎯 重试搜索完成: 共 {len(all_results)} 个结果，使用策略: {', '.join(used_methods)}")
    
    return all_results

if __name__ == "__main__":
    try:
        # 确保必要的目录存在
        os.makedirs(CONFIG['UPLOAD_DIR'], exist_ok=True)
        os.makedirs(CONFIG['CACHE_DIR'], exist_ok=True) 
        os.makedirs(CONFIG['TEMP_DIR'], exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        logger.info("🚀 启动智能文本分析平台...")
        main()
    except Exception as e:
        logger.error(f"❌ 应用启动失败: {str(e)}")
        st.error(f"❌ 应用启动失败: {str(e)}")
        print(f"应用启动错误: {e}")
        traceback.print_exc()
