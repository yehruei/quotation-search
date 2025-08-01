#!/bin/bash

# 引文搜索应用启动脚本
echo "🚀 启动引文搜索应用..."

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo "❌ 虚拟环境不存在，正在创建..."
    python3 -m venv venv
    echo "✅ 虚拟环境创建完成"
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 检查并安装依赖
echo "📦 检查依赖包..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# 启动应用
echo "🎯 启动Streamlit应用..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0

echo "✅ 应用启动完成！"
