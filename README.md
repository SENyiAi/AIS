# AIS - AI Image Super-Resolution

<div align="center">

**简体中文** | [English](#english)

AI 图像超分辨率工具，集成多个超分引擎，提供 Web 界面。

</div>

## 特性

- 多引擎支持: Real-CUGAN、Real-ESRGAN、Waifu2x、Anime4KCPP
- Web 界面: 基于 Gradio
- 图像超分: 单图/批量处理
- 动图支持: GIF/WebP 超分处理
- 视频超分: 支持 MP4/AVI/MKV 等格式，逐帧处理或原生处理
- 多语言: 简体中文和英文
- GPU 加速: Vulkan (NVIDIA/AMD/Intel)
- 开箱即用: Full版本内置环境
- 实时预览: 滑动条对比效果
- 自定义预设: 保存常用参数
- 断点续传: 视频处理支持中断恢复
<div align="center">
  <img src="https://github.com/user-attachments/assets/a39b3aee-86ba-43a2-aa26-0ed6acfd83a3" height="200" alt="界面预览" />
  <img src="https://github.com/user-attachments/assets/b24ddf12-7cc3-4c7c-82aa-386b998ed269" height="200" alt="噪点去除" />
</div>

## 快速开始

### 方式一：下载发行版

前往 [Releases](https://github.com/SENyiAi/AIS/releases) 下载最新版本。

| 版本 | 说明 |
|------|------|
| Full | 内置 Python 环境，开箱即用 |
| Lite | 需自行安装 Python 3.10+ |

**Full版使用：**
1. 解压到任意目录（路径避免中文）
2. 双击 `启动.bat`
3. 访问 http://127.0.0.1:7860

**Lite版使用：**
1. 解压文件
2. 双击 `安装依赖.bat`（自动创建虚拟环境，国内默认清华源）
3. 双击 `启动.bat`

### 方式二：从源码运行

```bash
git clone https://github.com/SENyiAi/AIS.git
cd AIS
python -m venv venv
source venv/bin/activate  # Linux/Mac, Windows用venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python AIS_WebUI.py

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
| Real-CUGAN | 专为动漫设计 | 动漫截图、插画 |
| Real-ESRGAN | 通用性强 | 照片、混合内容 |
| Waifu2x | 经典算法 | 快速处理 |
| Anime4KCPP | 极速处理 | 视频、GIF |

## 功能说明

### 图像超分
- 支持单张图片或批量处理
- 多种超分引擎可选
- 自定义放大倍数和降噪强度
- 实时预览对比效果

### 动图超分

支持 GIF 动态图超分辨率处理。

输出格式：
- WebP (推荐): 24-bit 真彩色，文件更小
- GIF: 256 色，兼容性好

处理方式：逐帧超分后重组，可选 FFmpeg 合成。

### 视频超分 (Beta)

支持视频文件超分辨率处理，两种模式：

**Anime4K 原生模式 (极速)**
- 直接处理视频，无需逐帧提取
- 速度最快，适合长视频
- 自动保留音轨和字幕
- 支持 CUDA/OpenCL 加速

**逐帧处理模式 (高质量)**
- 支持所有超分引擎
- 逐帧提取、处理、重组
- 支持断点续传
- 可自定义编码参数

功能特性：
- 输入格式: MP4/AVI/MKV/MOV/FLV 等
- 输出格式: MP4/WebM/MKV
- 视频编码: H.264/H.265/VP9
- 保留音轨: 支持多音轨
- 保留字幕: 内嵌/外挂字幕
- 断点续传: 中断后可继续处理
- 质量控制: CRF/比特率/编码速度

## 高级参数

- TTA 模式: 提升画质
- Tile 大小: 控制显存占用
- GPU 选择: 多显卡支持
- 线程数: 加载:处理:保存
- 输出格式: PNG/JPG/WebP

## 目录结构

```
AIS/
├── AIS_WebUI.py      # WebUI 主程序
├── AIS.py            # 命令行版本
├── 模型/             # 超分引擎
├── 前置/             # Python + FFmpeg (Full版)
├── 输出/             # 处理结果
└── 数据/             # 配置、预设、日志
```

## 许可证

GPL-3.0 license

## 致谢

- [Real-CUGAN](https://github.com/bilibili/ailab) - Bilibili
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Xintao Wang
- [Waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan) - nihui
- [Anime4KCPP](https://github.com/TianZerL/Anime4KCPP) - TianZerL
- [FFmpeg](https://ffmpeg.org/)
- [Gradio](https://gradio.app/)

---

<a name="english"></a>

## English

A one-stop AI image super-resolution tool integrating multiple top-tier upscaling engines with a clean Web UI.

### Features

- 🎨 **Multi-Engine**: Real-CUGAN, Real-ESRGAN, Waifu2x, Anime4KCPP
- 🖥️ **Web UI**: Modern Gradio-based interface
- 🎬 **GIF/WebP Animation**: Super-resolution for animated images
- 🌍 **i18n**: Chinese and English support
- ⚡ **GPU Accelerated**: Vulkan-based, supports NVIDIA/AMD/Intel
- 📦 **Portable**: Download and run, no setup required
- ⭐ **Custom Presets**: Save and reuse your favorite settings

### Quick Start

1. Download from [Releases](https://github.com/SENyiAi/AIS/releases)
2. Extract to any directory
3. Run `启动.bat`
4. Open http://127.0.0.1:7860

### License

MIT License
