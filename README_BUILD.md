# Quotation Search - 应用打包说明

本项目使用 **Nuitka + subprocess 调用 Streamlit** 的方案来打包成独立的可执行应用。

## 🚀 一键打包（推荐）

### Windows用户
```bash
# 双击运行
build.bat
```

### Linux/macOS用户
```bash
# 运行脚本
./build.sh
```

## 🔧 手动打包

### 1. 环境准备
```bash
python setup_build.py
```

### 2. 开始打包
```bash
python build_app.py
```

### 3. 运行应用
打包完成后，运行生成的可执行文件：
- Windows: `main.exe`
- Linux/macOS: `./main`

## ✨ 打包方案特点

- ✅ 使用 Nuitka 打包，性能优秀，启动快速
- ✅ 通过 subprocess 启动 Streamlit，避免兼容性问题
- ✅ 自动打开浏览器，用户体验良好
- ✅ 单文件可执行，便于分发
- ✅ 无需安装Python环境即可运行

## 文件说明

### 核心文件

- `main.py` - 主启动器，负责启动 Streamlit 服务器
- `app.py` - Streamlit 应用主体（已简化）
- `build_app.py` - Nuitka 打包脚本
- `setup_build.py` - 构建环境设置脚本

### 工作流程

1. `main.py` 作为入口点被 Nuitka 打包
2. 运行时，`main.py` 通过 subprocess 启动 `streamlit run app.py`
3. 等待 Streamlit 服务器启动后自动打开浏览器
4. 用户通过浏览器使用应用

## 构建选项

Nuitka 构建使用以下选项：

- `--standalone` - 创建独立应用
- `--onefile` - 单文件模式
- `--enable-plugin=anti-bloat` - 减小文件大小
- `--include-data-dir` - 包含数据目录
- `--include-data-file` - 包含必要文件

## 注意事项

1. 确保 `final_model_data` 目录包含必要的模型文件
2. 首次运行可能需要较长时间启动
3. 应用运行时会占用 8501 端口
4. 关闭控制台窗口会停止应用

## 故障排除

### 构建失败
- 检查 Python 版本（需要 3.8+）
- 确保所有依赖已安装
- 检查磁盘空间是否充足

### 运行失败
- 检查 8501 端口是否被占用
- 确保防火墙允许本地连接
- 查看控制台错误信息

## 自定义配置

如需修改端口或其他设置，编辑 `main.py` 中的相关参数：

```python
# 修改端口
port = 8501

# 修改超时时间
timeout = 30
```
