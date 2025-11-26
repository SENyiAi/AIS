# AIS - AI Image Super-Resolution

<div align="center">

**简体中文** | [English](#english)

一站式 AI 图像超分辨率工具，集成多个顶级超分引擎，提供简洁的 Web 界面。

</div>

## ✨ 特性

- 🎨 **多引擎支持**: Real-CUGAN、Real-ESRGAN、Waifu2x
- 🖥️ **Web 界面**: 基于 Gradio 的现代化界面
- 🌍 **多语言**: 支持简体中文和英文
- ⚡ **GPU 加速**: 基于 Vulkan，支持 NVIDIA/AMD/Intel 显卡
- 📦 **开箱即用**: 下载即用，无需配置环境
<div align="center">
  <img src="https://github.com/user-attachments/assets/a39b3aee-86ba-43a2-aa26-0ed6acfd83a3" height="200" alt="界面预览" />
  <img src="https://github.com/user-attachments/assets/b24ddf12-7cc3-4c7c-82aa-386b998ed269" height="200" alt="噪点去除" />
</div>

## 🚀 快速开始

### 方式一：下载发行版（推荐）

前往 [Releases](https://github.com/SENyiAi/AIS/releases) 下载最新版本：

| 版本 | 说明 | 适用场景 |
|------|------|----------|
| **Full (完整版)** | 内置 Python 3.12 + Gradio，开箱即用 | 推荐大多数用户 |
| **Lite (精简版)** | 仅包含核心文件，需自行安装 Python | 已有 Python 环境的用户 |

**完整版使用步骤：**
1. 下载 `AIS-vX.X.X-Full.zip`
2. 解压缩到任意目录（路径不要有中文）
3. 双击 `启动.bat` 运行
4. 浏览器访问 http://127.0.0.1:7860

**精简版使用步骤：**
1. 确保已安装 Python 3.10+
2. 下载 `AIS-vX.X.X-Lite.zip` 并解压
3. 运行 `pip install -r requirements.txt`
4. 双击 `启动.bat` 运行

### 方式二：从源码运行

```bash
# 克隆仓库
git clone https://github.com/SENyiAi/AIS.git
cd AIS

# 安装依赖
pip install -r requirements.txt

# 下载模型文件（放入 模型/ 目录）
# - realcugan-ncnn-vulkan
# - realesrgan-ncnn-vulkan
# - waifu2x-ncnn-vulkan

# 运行
python AIS_WebUI.py
```

## 📖 引擎介绍

| 引擎 | 特点 | 推荐场景 |
|------|------|----------|
| **Real-CUGAN** | 专为动漫设计，细节保留出色 | 动漫截图、插画 |
| **Real-ESRGAN** | 通用性强，效果稳定 | 照片、混合内容 |
| **Waifu2x** | 经典算法，速度快 | 快速预览、批量处理 |

## 🛠️ 高级参数

所有引擎都支持以下高级参数：

- **TTA 模式**: 8倍时间换取更好效果
- **Tile 大小**: 控制显存占用
- **GPU 选择**: 多显卡选择
- **线程数**: 加载:处理:保存
- **输出格式**: PNG/JPG/WebP

## 📁 目录结构

```
AIS/
├── AIS_WebUI.py      # WebUI 主程序
├── AIS.py            # 命令行版本
├── i18n.py           # i18n模块
├── 模型/             # 超分引擎
├── 前置/             # Python 嵌入版
├── 输出/             # 处理结果
└── 数据/             # 配置和日志
```

## 📄 许可证

MIT License

## 🙏 致谢

- [Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [Waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan)
- [Gradio](https://gradio.app/)

---

<a name="english"></a>

## English

A one-stop AI image super-resolution tool integrating multiple top-tier upscaling engines with a clean Web UI.

### Features

- 🎨 **Multi-Engine**: Real-CUGAN, Real-ESRGAN, Waifu2x
- 🖥️ **Web UI**: Modern Gradio-based interface
- 🌍 **i18n**: Chinese and English support
- ⚡ **GPU Accelerated**: Vulkan-based, supports NVIDIA/AMD/Intel
- 📦 **Portable**: Download and run, no setup required

### Quick Start

1. Download from [Releases](https://github.com/SENyiAi/AIS/releases)
2. Extract to any directory
3. Run `启动.bat`
4. Open http://127.0.0.1:7860

### License

MIT License
