# AIS - AI Image Super-Resolution

<div align="center">

**简体中文** | [English](#english)

AI 图像超分辨率工具，集成多个超分引擎，提供 Web 界面。

</div>

## 特性

- 多引擎支持: Real-CUGAN、Real-ESRGAN、Waifu2x、Anime4KCPP
- Web 界面: 基于 Gradio
- 动图支持: GIF/WebP 超分处理
- 多语言: 简体中文和英文
- GPU 加速: Vulkan (NVIDIA/AMD/Intel)
- 开箱即用: Full版本内置环境
- 实时预览: 滑动条对比效果
- 自定义预设: 保存常用参数
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

## 动图超分

支持 GIF 动态图超分辨率处理。

输出格式：
- WebP (推荐): 24-bit 真彩色，文件更小
- GIF: 256 色，兼容性好

处理方式：逐帧超分后重组，可选 FFmpeg 合成。

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
