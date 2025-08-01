# 📚 智能文本分析平台 - 服务器部署指南

## 🚀 快速部署

### 服务器部署 (推荐)

1. **上传项目到服务器**
```bash
# 将项目文件传输到服务器
scp -r Quotation_Search user@server:/path/to/deployment/
cd /path/to/deployment/Quotation_Search
```

2. **安装Python环境**
```bash
# 确保Python 3.9+ 已安装
python3 --version

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

3. **启动应用**
```bash
# 使用启动脚本
chmod +x start_app.sh
./start_app.sh

# 或直接启动
source venv/bin/activate
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

4. **访问应用**
- 浏览器打开: http://your-server-ip:8501
- 健康检查: http://your-server-ip:8501/_stcore/health

## ⚙️ 系统要求

- **CPU**: 2核心以上
- **内存**: 4GB以上 (推荐8GB)
- **磁盘**: 10GB可用空间
- **Python**: 3.9+
- **操作系统**: Linux/macOS/Windows

## 📁 目录结构

```
Quotation_Search/
├── app.py                      # 主应用文件
├── modules/                    # 核心模块
├── data/                       # 示例数据
├── uploads/                    # 用户上传文件
├── cache/                      # 处理缓存
├── logs/                       # 应用日志
├── temp/                       # 临时文件
├── .streamlit/                 # Streamlit配置
├── requirements.txt            # 生产依赖
├── start_app.sh               # 启动脚本
└── DEPLOYMENT.md              # 部署文档
```

## 🔧 配置说明

### Streamlit 配置 (.streamlit/config.toml)
- **端口**: 8501
- **最大上传**: 50MB
- **安全设置**: 已优化生产环境
- **性能优化**: 启用缓存和优化

### 应用配置 (app.py)
- **最大并发用户**: 10
- **速率限制**: 每小时10次请求
- **缓存TTL**: 1小时
- **文件大小限制**: 50MB

## 🔒 安全设置

1. **CORS 保护**: 已禁用跨域请求
2. **XSRF 保护**: 已启用
3. **文件上传限制**: 仅支持PDF格式
4. **用户会话管理**: 独立会话隔离
5. **速率限制**: 防止滥用攻击

## 📊 监控和日志

### 日志位置
- **应用日志**: logs/app.log
- **系统日志**: 容器标准输出

### 健康检查
- **端点**: /_stcore/health
- **间隔**: 30秒
- **超时**: 10秒

### 系统监控
- CPU/内存使用率实时显示
- 用户会话数量监控
- 请求频率统计

## 🚀 性能优化

### 缓存策略
- **PDF处理缓存**: 基于文件哈希
- **模型缓存**: 一次加载全局共享
- **搜索结果缓存**: 1小时TTL

### 并发处理
- **线程池**: 最大10个并发任务
- **会话隔离**: 独立用户状态
- **资源限制**: 防止内存溢出

## 🔧 故障排除

### 常见问题

1. **内存不足**
   - 检查系统内存使用
   - 调整 `MAX_CONCURRENT_USERS` 配置
   - 重启容器释放内存

2. **模型加载失败**
   - 检查网络连接
   - 验证模型文件完整性
   - 查看日志错误信息

3. **PDF处理错误**
   - 检查文件格式和大小
   - 验证上传目录权限
   - 清理临时文件

### 日志查看
```bash
# 应用日志
tail -f logs/app.log

# 系统服务日志 (systemd)
sudo journalctl -u quotation-search -f

# Supervisor日志
sudo tail -f logs/supervisor.log
```

## 📈 生产环境优化

### 使用进程管理器 (推荐)

1. **使用Supervisor**
```bash
# 安装supervisor
sudo apt-get install supervisor  # Ubuntu/Debian
sudo yum install supervisor      # CentOS/RHEL

# 创建配置文件 /etc/supervisor/conf.d/quotation-search.conf
[program:quotation-search]
directory=/path/to/Quotation_Search
command=/path/to/Quotation_Search/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
user=www-data
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/path/to/Quotation_Search/logs/supervisor.log

# 启动服务
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start quotation-search
```

2. **使用systemd**
```bash
# 创建服务文件 /etc/systemd/system/quotation-search.service
[Unit]
Description=Quotation Search App
After=network.target

[Service]
Type=exec
User=www-data
WorkingDirectory=/path/to/Quotation_Search
ExecStart=/path/to/Quotation_Search/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target

# 启动服务
sudo systemctl daemon-reload
sudo systemctl enable quotation-search
sudo systemctl start quotation-search
sudo systemctl status quotation-search
```

### 反向代理配置 (可选)

**Nginx配置示例**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_buffering off;
    }
}
```

### 环境变量配置
```bash
# 可选的环境变量
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export MAX_CONCURRENT_USERS=20
export RATE_LIMIT_PER_HOUR=50
```

## 🛠️ 维护

### 定期清理
```bash
# 清理过期缓存
find cache/ -name "*.cache" -mtime +7 -delete

# 清理临时文件
find temp/ -type f -mtime +1 -delete

# 清理日志文件 (保留7天)
find logs/ -name "*.log" -mtime +7 -delete
```

### 备份建议
- 定期备份用户上传文件 (uploads/)
- 备份应用配置文件
- 监控磁盘空间使用

## 📞 支持

如遇问题，请检查：
1. 系统日志和应用日志
2. 健康检查端点状态
3. 系统资源使用情况
4. 网络连接状态

---

**部署完成！** 🎉 您的智能文本分析平台已准备就绪，可以为多用户提供服务。