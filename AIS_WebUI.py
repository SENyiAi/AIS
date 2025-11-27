import sys
import subprocess
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime

# 基础路径配置
BASE_DIR = Path(__file__).parent.absolute()

# 确保程序根目录在 sys.path 中（用于导入 i18n 等本地模块）
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

PREREQ_DIR = BASE_DIR / "前置"
MODEL_DIR = BASE_DIR / "模型"
OUTPUT_DIR = BASE_DIR / "输出"
OUTPUT_DIR.mkdir(exist_ok=True)

# 设置本地Python库路径 (支持多个版本)
for python_dir_name in ["python-3.12.10-embed-amd64", "python-3.12.7-embed-amd64", "python-3.14.0-embed-amd64"]:
    LOCAL_PYTHON_DIR = PREREQ_DIR / python_dir_name
    if LOCAL_PYTHON_DIR.exists():
        LOCAL_LIB_PATH = LOCAL_PYTHON_DIR / "Lib" / "site-packages"
        if LOCAL_LIB_PATH.exists() and str(LOCAL_LIB_PATH) not in sys.path:
            sys.path.insert(0, str(LOCAL_LIB_PATH))
        break

# Gradio自动安装
def install_gradio() -> bool:
    """检查并安装Gradio"""
    try:
        import gradio
        return True
    except ImportError:
        print("[提示] Gradio未安装, 正在尝试安装...")
        
        # 优先使用本地whl安装
        local_whl = PREREQ_DIR / "Grodio" / "gradio-6.0.1-py3-none-any.whl"
        if local_whl.exists():
            print(f"[安装] 使用本地软件包: {local_whl}")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    str(local_whl), "--quiet"
                ])
                print("[完成] Gradio安装成功")
                return True
            except subprocess.CalledProcessError as e:
                print(f"[警告] 本地安装失败: {e}")
        
        # 尝试在线安装
        print("[安装] 尝试在线安装Gradio...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "gradio", "--quiet"
            ])
            print("[完成] Gradio在线安装成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[错误] Gradio安装失败: {e}")
            return False

# 安装Gradio
if not install_gradio():
    print("[错误] 无法安装Gradio, 程序退出")
    sys.exit(1)

import gradio as gr
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import logging
import io
import numpy as np
import tempfile
import uuid
import time

# 导入国际化模块
from i18n import t, get_choices, get_current_lang, set_lang, load_lang_config, LANGUAGES

# 数据文件夹 - 存放所有配置和日志
DATA_DIR = BASE_DIR / "数据"
DATA_DIR.mkdir(exist_ok=True)

# 日志系统
LOG_FILE = DATA_DIR / "ais_log.txt"
LOG_BUFFER = io.StringIO()

def setup_logging():
    """设置日志系统"""
    # 清空旧日志
    if LOG_FILE.exists():
        try:
            LOG_FILE.unlink()
        except (OSError, PermissionError):
            pass  # 文件可能被占用，忽略
    
    logger = logging.getLogger('AIS')
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

def log_info(msg: str):
    """记录日志"""
    logger.info(msg)

# 临时文件目录（用于存放剪贴板图片）
TEMP_DIR = BASE_DIR / "临时"
TEMP_DIR.mkdir(exist_ok=True)

def preprocess_image_input(image_input) -> Optional[str]:
    """预处理图片输入，支持多种输入类型
    
    参数:
        image_input: 可以是文件路径(str), PIL.Image, numpy数组, 或None
    
    返回:
        图片文件路径(str)，如果输入无效则返回None
    """
    if image_input is None:
        return None
    
    # 如果已经是文件路径字符串
    if isinstance(image_input, str):
        if Path(image_input).exists():
            # 检查是否为GIF并记录
            if image_input.lower().endswith('.gif'):
                log_info(f"[信息] 检测到GIF文件: {Path(image_input).name}")
                if is_animated_gif(image_input):
                    log_info(f"[信息] 这是动态GIF，将逐帧处理")
            return image_input
        log_info(f"[警告] 文件不存在: {image_input}")
        return None
    
    # 如果是 Path 对象
    if isinstance(image_input, Path):
        if image_input.exists():
            return str(image_input)
        return None
    
    # 如果是 numpy 数组（剪贴板粘贴可能是这种格式）
    # 注意：从剪贴板粘贴的GIF会丢失动画信息，只保留第一帧
    if isinstance(image_input, np.ndarray):
        try:
            # 转换为 PIL Image
            if image_input.ndim == 2:
                # 灰度图
                pil_image = Image.fromarray(image_input, mode='L')
            elif image_input.ndim == 3:
                if image_input.shape[2] == 4:
                    # RGBA
                    pil_image = Image.fromarray(image_input, mode='RGBA')
                else:
                    # RGB
                    pil_image = Image.fromarray(image_input, mode='RGB')
            else:
                log_info(f"[警告] 不支持的数组维度: {image_input.ndim}")
                return None
            
            # 保存到临时文件
            temp_filename = f"clipboard_{int(time.time() * 1000)}.png"
            temp_path = TEMP_DIR / temp_filename
            pil_image.save(temp_path, format='PNG')
            log_info(f"[信息] 从剪贴板保存图片: {temp_path.name}")
            log_info(f"[提示] 如需处理动态GIF，请使用文件上传功能")
            return str(temp_path)
        except Exception as e:
            log_info(f"[错误] 处理numpy数组失败: {e}")
            return None
    
    # 如果是 PIL Image
    if isinstance(image_input, Image.Image):
        try:
            temp_filename = f"clipboard_{uuid.uuid4().hex[:8]}.png"
            temp_path = TEMP_DIR / temp_filename
            image_input.save(temp_path, format='PNG')
            log_info(f"[信息] 从剪贴板保存图片: {temp_path.name}")
            return str(temp_path)
        except Exception as e:
            log_info(f"[错误] 保存PIL图片失败: {e}")
            return None
    
    # 未知类型
    log_info(f"[警告] 未知的图片输入类型: {type(image_input)}")
    return None

def cleanup_temp_files():
    """清理临时文件夹中的旧文件"""
    try:
        for temp_file in TEMP_DIR.glob("clipboard_*.png"):
            try:
                temp_file.unlink()
            except (OSError, PermissionError):
                pass  # 文件可能正在使用
    except Exception:
        pass


# ============================================================
# 缩略图系统 - 用于图库显示
# ============================================================

THUMBNAIL_DIR = BASE_DIR / "缩略图"
THUMBNAIL_DIR.mkdir(exist_ok=True)
THUMBNAIL_SIZE = (300, 300)  # 缩略图尺寸


def get_thumbnail_path(image_path: str) -> Path:
    """获取图片对应的缩略图路径"""
    original = Path(image_path)
    # 使用原文件名 + 修改时间哈希作为缩略图名，确保更新后重新生成
    mtime = original.stat().st_mtime if original.exists() else 0
    thumb_name = f"{original.stem}_{int(mtime)}_thumb.jpg"
    return THUMBNAIL_DIR / thumb_name


def create_thumbnail(image_path: str) -> Optional[str]:
    """为图片创建缩略图"""
    try:
        original = Path(image_path)
        if not original.exists():
            return None
        
        thumb_path = get_thumbnail_path(image_path)
        
        # 如果缩略图已存在且更新，直接返回
        if thumb_path.exists():
            return str(thumb_path)
        
        # 创建缩略图
        with Image.open(original) as img:
            # 处理GIF：只取第一帧
            if hasattr(img, 'n_frames') and img.n_frames > 1:
                img.seek(0)
            
            # 转换为RGB（处理RGBA等情况）
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 计算缩略图尺寸（保持比例）
            img.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            
            # 保存为JPEG（体积小）
            img.save(thumb_path, 'JPEG', quality=85, optimize=True)
        
        return str(thumb_path)
    except Exception as e:
        log_info(f"[警告] 创建缩略图失败: {e}")
        return None


def get_gallery_thumbnails() -> List[str]:
    """获取图库缩略图列表"""
    if not OUTPUT_DIR.exists():
        return []
    
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
    thumbnails = []
    
    try:
        images = []
        for f in OUTPUT_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in extensions:
                images.append(str(f))
        
        # 按修改时间倒序排列
        images.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
        
        # 为每张图片创建/获取缩略图
        for img_path in images:
            thumb = create_thumbnail(img_path)
            if thumb:
                thumbnails.append(thumb)
            else:
                # 缩略图创建失败，使用原图路径（但不推荐）
                thumbnails.append(img_path)
    except (OSError, PermissionError) as e:
        log_info(f"读取图库失败: {e}")
    
    return thumbnails


def cleanup_old_thumbnails():
    """清理过期的缩略图"""
    try:
        valid_stems = set()
        for f in OUTPUT_DIR.iterdir():
            if f.is_file():
                valid_stems.add(f.stem)
        
        for thumb in THUMBNAIL_DIR.glob("*_thumb.jpg"):
            # 提取原文件名部分
            parts = thumb.stem.rsplit('_', 2)
            if len(parts) >= 2:
                original_stem = parts[0]
                if original_stem not in valid_stems:
                    try:
                        thumb.unlink()
                    except (OSError, PermissionError):
                        pass
    except Exception:
        pass


# ============================================================
# GIF 处理功能
# ============================================================

def is_gif(image_path: str) -> bool:
    """检查图片是否为GIF"""
    return Path(image_path).suffix.lower() == '.gif'


def is_animated_gif(image_path: str) -> bool:
    """检查是否为动态GIF"""
    try:
        with Image.open(image_path) as img:
            return getattr(img, 'n_frames', 1) > 1
    except Exception:
        return False


def extract_gif_frames_ffmpeg(gif_path: str) -> Tuple[List[str], float, Optional[str]]:
    """使用FFmpeg提取GIF帧（更好的颜色和透明度处理）
    
    返回: (帧文件路径列表, 帧率, 错误信息)
    """
    try:
        frames_dir = TEMP_DIR / f"gif_frames_{uuid.uuid4().hex[:8]}"
        frames_dir.mkdir(exist_ok=True)
        
        # 使用ffprobe获取帧率
        fps = 10.0  # 默认帧率
        if FFPROBE_EXE.exists():
            try:
                probe_cmd = [
                    str(FFPROBE_EXE),
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_streams",
                    gif_path
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    import json
                    info = json.loads(result.stdout)
                    for stream in info.get('streams', []):
                        if 'r_frame_rate' in stream:
                            fps_str = stream['r_frame_rate']
                            if '/' in fps_str:
                                num, den = fps_str.split('/')
                                fps = float(num) / float(den) if float(den) != 0 else 10.0
                            else:
                                fps = float(fps_str)
                            break
            except Exception:
                pass
        
        # 使用FFmpeg提取帧
        if FFMPEG_EXE.exists():
            output_pattern = str(frames_dir / "frame_%04d.png")
            cmd = [
                str(FFMPEG_EXE),
                "-i", gif_path,
                "-vsync", "0",  # 保持原始帧率
                output_pattern
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            
            if result.returncode == 0:
                frame_paths = sorted([str(p) for p in frames_dir.glob("frame_*.png")])
                if frame_paths:
                    log_info(f"[FFmpeg] 提取了 {len(frame_paths)} 帧, 帧率: {fps:.2f}fps")
                    return frame_paths, fps, None
        
        # 如果FFmpeg不可用或失败，回退到PIL方法
        return extract_gif_frames_pil(gif_path)
    except Exception as e:
        log_info(f"[FFmpeg] 提取GIF帧失败: {e}, 回退到PIL")
        return extract_gif_frames_pil(gif_path)


def extract_gif_frames_pil(gif_path: str) -> Tuple[List[str], float, Optional[str]]:
    """使用PIL提取GIF帧（备用方法）
    
    正确处理GIF的disposal方法和透明度，确保每帧都是完整的合成图像
    
    返回: (帧文件路径列表, 帧率, 错误信息)
    """
    try:
        frames_dir = TEMP_DIR / f"gif_frames_{uuid.uuid4().hex[:8]}"
        frames_dir.mkdir(exist_ok=True)
        
        frame_paths = []
        total_duration = 0
        
        with Image.open(gif_path) as img:
            n_frames = getattr(img, 'n_frames', 1)
            
            # 获取GIF的逻辑屏幕大小
            size = img.size
            
            # 创建画布 - 用于累积帧内容
            # 背景色：优先使用GIF的背景色，否则使用白色
            bg_color = img.info.get('background', 0)
            if img.mode == 'P' and img.palette:
                try:
                    palette = img.palette.getdata()[1]
                    bg_r = palette[bg_color * 3] if bg_color * 3 < len(palette) else 255
                    bg_g = palette[bg_color * 3 + 1] if bg_color * 3 + 1 < len(palette) else 255
                    bg_b = palette[bg_color * 3 + 2] if bg_color * 3 + 2 < len(palette) else 255
                    bg_rgba = (bg_r, bg_g, bg_b, 255)
                except:
                    bg_rgba = (255, 255, 255, 255)
            else:
                bg_rgba = (255, 255, 255, 255)
            
            # 初始画布
            canvas = Image.new('RGBA', size, bg_rgba)
            last_disposal = 0
            last_frame_region = None
            prev_canvas: Optional[Image.Image] = None  # 用于disposal=3
            
            for i in range(n_frames):
                img.seek(i)
                
                # 获取帧持续时间（毫秒）
                duration = img.info.get('duration', 100)
                if duration <= 0:
                    duration = 100
                total_duration += duration
                
                # 获取disposal方法
                # 0: 不处理, 1: 保留, 2: 恢复到背景色, 3: 恢复到上一帧
                disposal = img.info.get('disposal', 0)
                
                # 获取当前帧的位置信息（用于局部帧）
                # tile格式: [('gif', (x0, y0, x1, y1), offset, ...)]
                try:
                    if hasattr(img, 'tile') and img.tile:
                        tile = img.tile[0]
                        if len(tile) > 1:
                            frame_box = tile[1]  # (x0, y0, x1, y1)
                        else:
                            frame_box = (0, 0, size[0], size[1])
                    else:
                        frame_box = (0, 0, size[0], size[1])
                except:
                    frame_box = (0, 0, size[0], size[1])
                
                # 处理上一帧的disposal（在绘制当前帧之前）
                if i > 0:
                    if last_disposal == 2 and last_frame_region:
                        # 恢复到背景色
                        bg_region = Image.new('RGBA', 
                            (last_frame_region[2] - last_frame_region[0], 
                             last_frame_region[3] - last_frame_region[1]), 
                            bg_rgba)
                        canvas.paste(bg_region, (last_frame_region[0], last_frame_region[1]))
                    elif last_disposal == 3 and prev_canvas is not None:
                        # 恢复到上一帧 - 使用保存的画布副本
                        canvas = prev_canvas.copy()
                
                # 保存当前画布状态（用于disposal=3）
                current_canvas_backup = canvas.copy()
                
                # 转换当前帧为RGBA
                # 需要保留调色板并正确处理透明度
                frame_rgba = img.convert('RGBA')
                
                # 将当前帧合成到画布上
                # 使用paste而不是alpha_composite来正确处理透明区域
                canvas.paste(frame_rgba, mask=frame_rgba.split()[3])
                
                # 保存完整合成帧
                frame_path = frames_dir / f"frame_{i:04d}.png"
                canvas.save(frame_path, 'PNG')
                frame_paths.append(str(frame_path))
                
                # 保存状态供下一帧使用
                last_disposal = disposal
                last_frame_region = frame_box
                prev_canvas = current_canvas_backup
        
        # 计算平均帧率
        fps = 1000.0 * n_frames / total_duration if total_duration > 0 else 10.0
        
        return frame_paths, fps, None
    except Exception as e:
        import traceback
        log_info(f"[GIF] 帧提取错误: {traceback.format_exc()}")
        return [], 10.0, str(e)


def extract_gif_frames(gif_path: str) -> Tuple[List[str], List[int], Optional[str]]:
    """提取GIF帧 - 兼容旧接口
    
    正确处理GIF的disposal方法和透明度，确保每帧都是完整的合成图像
    
    返回: (帧文件路径列表, 每帧持续时间列表, 错误信息)
    """
    try:
        frames_dir = TEMP_DIR / f"gif_frames_{uuid.uuid4().hex[:8]}"
        frames_dir.mkdir(exist_ok=True)
        
        frame_paths = []
        durations = []
        
        with Image.open(gif_path) as img:
            n_frames = getattr(img, 'n_frames', 1)
            
            # 获取GIF的逻辑屏幕大小
            size = img.size
            
            # 获取背景色
            bg_color = img.info.get('background', 0)
            if img.mode == 'P' and img.palette:
                try:
                    palette = img.palette.getdata()[1]
                    bg_r = palette[bg_color * 3] if bg_color * 3 < len(palette) else 255
                    bg_g = palette[bg_color * 3 + 1] if bg_color * 3 + 1 < len(palette) else 255
                    bg_b = palette[bg_color * 3 + 2] if bg_color * 3 + 2 < len(palette) else 255
                    bg_rgba = (bg_r, bg_g, bg_b, 255)
                except:
                    bg_rgba = (255, 255, 255, 255)
            else:
                bg_rgba = (255, 255, 255, 255)
            
            # 初始画布
            canvas = Image.new('RGBA', size, bg_rgba)
            last_disposal = 0
            last_frame_region = None
            prev_canvas: Optional[Image.Image] = None
            
            for i in range(n_frames):
                img.seek(i)
                
                # 获取帧持续时间（毫秒）
                duration = img.info.get('duration', 100)
                if duration <= 0:
                    duration = 100
                durations.append(duration)
                
                # 获取disposal方法
                disposal = img.info.get('disposal', 0)
                
                # 获取帧位置
                try:
                    if hasattr(img, 'tile') and img.tile:
                        tile = img.tile[0]
                        if len(tile) > 1:
                            frame_box = tile[1]
                        else:
                            frame_box = (0, 0, size[0], size[1])
                    else:
                        frame_box = (0, 0, size[0], size[1])
                except:
                    frame_box = (0, 0, size[0], size[1])
                
                # 处理上一帧的disposal
                if i > 0:
                    if last_disposal == 2 and last_frame_region:
                        bg_region = Image.new('RGBA', 
                            (last_frame_region[2] - last_frame_region[0], 
                             last_frame_region[3] - last_frame_region[1]), 
                            bg_rgba)
                        canvas.paste(bg_region, (last_frame_region[0], last_frame_region[1]))
                    elif last_disposal == 3 and prev_canvas is not None:
                        canvas = prev_canvas.copy()
                
                # 保存画布状态
                current_canvas_backup = canvas.copy()
                
                # 转换当前帧为RGBA
                frame_rgba = img.convert('RGBA')
                
                # 合成到画布
                canvas.paste(frame_rgba, mask=frame_rgba.split()[3])
                
                # 保存完整合成帧
                frame_path = frames_dir / f"frame_{i:04d}.png"
                canvas.save(frame_path, 'PNG')
                frame_paths.append(str(frame_path))
                
                # 更新状态
                last_disposal = disposal
                last_frame_region = frame_box
                prev_canvas = current_canvas_backup
        
        return frame_paths, durations, None
    except Exception as e:
        import traceback
        log_info(f"[GIF] 帧提取错误: {traceback.format_exc()}")
        return [], [], str(e)


def reassemble_gif_ffmpeg(frame_paths: List[str], fps: float, output_path: str) -> Tuple[bool, str]:
    """使用FFmpeg重新组装GIF（更好的颜色和压缩）
    
    参数:
        frame_paths: 处理后的帧文件路径列表
        fps: 帧率
        output_path: 输出GIF路径
    
    返回: (成功标志, 消息)
    """
    try:
        if not frame_paths:
            return False, "没有帧可组装"
        
        if not FFMPEG_EXE.exists():
            # 回退到PIL方法，需要转换fps为durations
            duration_ms = int(1000 / fps) if fps > 0 else 100
            durations = [duration_ms] * len(frame_paths)
            return reassemble_gif(frame_paths, durations, output_path)
        
        # 获取帧目录
        frames_dir = Path(frame_paths[0]).parent
        
        # 使用FFmpeg合成GIF - 两步法获得最佳质量
        # 先创建全局调色板（从所有帧采样）
        palette_path = frames_dir / "palette.png"
        
        # 生成调色板 - 使用full模式分析所有帧
        # stats_mode=full: 分析所有帧的颜色获得全局最优调色板
        palette_cmd = [
            str(FFMPEG_EXE),
            "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "processed_%04d.png"),
            "-vf", "palettegen=max_colors=256:stats_mode=full:reserve_transparent=0",
            str(palette_path)
        ]
        
        result = subprocess.run(palette_cmd, capture_output=True, timeout=120)
        
        if result.returncode == 0 and palette_path.exists():
            # 使用调色板生成高质量GIF
            # sierra2_4a: Sierra-2-4A 抖动算法，比 floyd_steinberg 产生更少的图案
            # 也可以尝试: sierra2, sierra3, none (无抖动)
            gif_cmd = [
                str(FFMPEG_EXE),
                "-y",
                "-framerate", str(fps),
                "-i", str(frames_dir / "processed_%04d.png"),
                "-i", str(palette_path),
                "-lavfi", "paletteuse=dither=sierra2_4a:diff_mode=rectangle:new=1",
                "-loop", "0",
                output_path
            ]
        else:
            # 直接生成GIF（无调色板优化）
            gif_cmd = [
                str(FFMPEG_EXE),
                "-y",
                "-framerate", str(fps),
                "-i", str(frames_dir / "processed_%04d.png"),
                "-loop", "0",
                output_path
            ]
        
        result = subprocess.run(gif_cmd, capture_output=True, timeout=180)
        
        if result.returncode == 0 and Path(output_path).exists():
            log_info(f"[FFmpeg] 成功组装 {len(frame_paths)} 帧GIF")
            return True, f"GIF组装完成: {len(frame_paths)}帧 (FFmpeg)"
        else:
            error = result.stderr.decode('utf-8', errors='ignore')[:200]
            log_info(f"[FFmpeg] GIF组装失败: {error}")
            # 回退到PIL
            duration_ms = int(1000 / fps) if fps > 0 else 100
            durations = [duration_ms] * len(frame_paths)
            return reassemble_gif(frame_paths, durations, output_path)
            
    except Exception as e:
        log_info(f"[FFmpeg] GIF组装异常: {e}, 回退到PIL")
        duration_ms = int(1000 / fps) if fps > 0 else 100
        durations = [duration_ms] * len(frame_paths)
        return reassemble_gif(frame_paths, durations, output_path)


def reassemble_gif(frame_paths: List[str], durations: List[int], output_path: str) -> Tuple[bool, str]:
    """重新组装GIF（PIL方法）
    
    使用全局调色板确保所有帧颜色一致，避免色带/色块问题
    
    参数:
        frame_paths: 处理后的帧文件路径列表
        durations: 每帧持续时间（毫秒）
        output_path: 输出GIF路径
    
    返回: (成功标志, 消息)
    """
    try:
        if not frame_paths:
            return False, "没有帧可组装"
        
        # 第一步：加载所有帧为RGB
        rgb_frames = []
        for path in frame_paths:
            if Path(path).exists():
                frame = Image.open(path).convert('RGBA')
                background = Image.new('RGBA', frame.size, (255, 255, 255, 255))
                composite = Image.alpha_composite(background, frame)
                rgb_frames.append(composite.convert('RGB'))
        
        if not rgb_frames:
            return False, "无法加载处理后的帧"
        
        # 第二步：创建全局调色板
        # 方法：将所有帧的像素合并，然后生成一个统一的调色板
        # 为了效率，我们从采样帧中提取调色板
        sample_size = min(10, len(rgb_frames))  # 最多采样10帧
        step = max(1, len(rgb_frames) // sample_size)
        
        # 创建一个大图来收集颜色样本
        sample_width = rgb_frames[0].width
        sample_height = rgb_frames[0].height
        
        # 将采样帧垂直拼接
        combined_height = sample_height * sample_size
        combined = Image.new('RGB', (sample_width, combined_height))
        
        for i, idx in enumerate(range(0, len(rgb_frames), step)):
            if i >= sample_size:
                break
            combined.paste(rgb_frames[idx], (0, i * sample_height))
        
        # 从合并图像生成全局调色板
        global_palette_img = combined.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
        global_palette = global_palette_img.getpalette()
        
        # 第三步：使用全局调色板转换所有帧
        frames = []
        for rgb_frame in rgb_frames:
            # 使用全局调色板量化，启用抖动以减少色带
            p_frame = rgb_frame.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG, palette=global_palette_img)
            frames.append(p_frame)
        
        # 确保durations列表长度正确
        while len(durations) < len(frames):
            durations.append(100)
        
        # 保存为GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations[:len(frames)],
            loop=0,
            disposal=2,
            optimize=False
        )
        
        log_info(f"[GIF] 成功组装 {len(frames)} 帧到 {Path(output_path).name}")
        return True, f"GIF组装完成: {len(frames)}帧"
    except Exception as e:
        log_info(f"[GIF] 组装失败: {e}")
        return False, f"GIF组装失败: {e}"


def reassemble_webp(frame_paths: List[str], durations: List[int], output_path: str) -> Tuple[bool, str]:
    """组装为WebP动图（支持24-bit颜色，无色带问题）
    
    参数:
        frame_paths: 处理后的帧文件路径列表
        durations: 每帧持续时间（毫秒）
        output_path: 输出WebP路径
    
    返回: (成功标志, 消息)
    """
    try:
        if not frame_paths:
            return False, "没有帧可组装"
        
        frames = []
        target_size = None
        
        for path in frame_paths:
            if Path(path).exists():
                # 打开图像
                frame = Image.open(path)
                
                # 确定目标尺寸（使用第一帧的尺寸）
                if target_size is None:
                    target_size = frame.size
                
                # 转换为RGB模式（WebP动图不需要透明通道，避免闪烁问题）
                # 如果原图有透明背景，用白色填充
                if frame.mode == 'RGBA':
                    # 创建白色背景
                    background = Image.new('RGB', frame.size, (255, 255, 255))
                    # 使用 alpha 通道作为 mask 进行合成
                    background.paste(frame, mask=frame.split()[3])
                    frame = background
                elif frame.mode != 'RGB':
                    frame = frame.convert('RGB')
                
                # 确保尺寸一致（防止闪烁）
                if frame.size != target_size:
                    frame = frame.resize(target_size, Image.Resampling.LANCZOS)
                
                frames.append(frame)
        
        if not frames:
            return False, "无法加载处理后的帧"
        
        # 确保durations列表长度正确
        while len(durations) < len(frames):
            durations.append(100)
        
        # 保存为WebP动图
        # 使用 RGB 模式，quality=95 提供高质量的有损压缩
        # 有损压缩在视觉上几乎无损，但避免了无损模式可能的兼容性问题
        frames[0].save(
            output_path,
            format='WEBP',
            save_all=True,
            append_images=frames[1:],
            duration=durations[:len(frames)],
            loop=0,
            quality=95, 
            method=6  
        )
        
        log_info(f"[WebP] 成功组装 {len(frames)} 帧到 {Path(output_path).name}")
        return True, f"WebP动图组装完成: {len(frames)}帧"
    except Exception as e:
        log_info(f"[WebP] 组装失败: {e}")
        return False, f"WebP组装失败: {e}"


def reassemble_webp_ffmpeg(frame_paths: List[str], fps: float, output_path: str, quality: int = 95) -> Tuple[bool, str]:
    """使用FFmpeg组装WebP动图
    
    参数:
        frame_paths: 处理后的帧文件路径列表
        fps: 帧率
        output_path: 输出WebP路径
        quality: 质量 (0-100, 100为最佳)
    
    返回: (成功标志, 消息)
    """
    try:
        if not frame_paths:
            return False, "没有帧可组装"
        
        if not FFMPEG_EXE.exists():
            # 回退到PIL方法
            duration_ms = int(1000 / fps) if fps > 0 else 100
            durations = [duration_ms] * len(frame_paths)
            return reassemble_webp(frame_paths, durations, output_path)
        
        frames_dir = Path(frame_paths[0]).parent
        
        # 使用FFmpeg生成高质量WebP动图
        # -preset picture: 针对静态图片优化
        # -compression_level 6: 最佳压缩
        # -quality 95: 高质量
        cmd = [
            str(FFMPEG_EXE),
            "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "processed_%04d.png"),
            "-c:v", "libwebp",
            "-quality", str(quality),
            "-compression_level", "6",
            "-preset", "picture",
            "-loop", "0",
            "-pix_fmt", "yuv420p",  # 标准像素格式，兼容性好
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        
        if result.returncode == 0 and Path(output_path).exists():
            log_info(f"[FFmpeg] 成功组装 {len(frame_paths)} 帧WebP动图")
            return True, f"WebP动图组装完成: {len(frame_paths)}帧 (FFmpeg)"
        else:
            error = result.stderr.decode('utf-8', errors='ignore')[:200]
            log_info(f"[FFmpeg] WebP组装失败: {error}")
            # 回退到PIL
            duration_ms = int(1000 / fps) if fps > 0 else 100
            durations = [duration_ms] * len(frame_paths)
            return reassemble_webp(frame_paths, durations, output_path)
            
    except Exception as e:
        log_info(f"[FFmpeg] WebP组装异常: {e}")
        duration_ms = int(1000 / fps) if fps > 0 else 100
        durations = [duration_ms] * len(frame_paths)
        return reassemble_webp(frame_paths, durations, output_path)


def cleanup_gif_temp(frames_dir: Path):
    """清理GIF临时文件"""
    try:
        import shutil
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
    except Exception:
        pass


# 全局状态
SHARE_URL: Optional[str] = None

# 引擎配置 - 使用模型文件夹
ENGINES: Dict[str, Dict[str, Any]] = {
    "cugan": {
        "dir": MODEL_DIR / "realcugan-ncnn-vulkan-20220728-windows",
        "exe": "realcugan-ncnn-vulkan.exe",
        "models": {
            "SE": "models-se",
            "Pro": "models-pro"
        }
    },
    "esrgan": {
        "dir": MODEL_DIR / "realesrgan-ncnn-vulkan-20220424-windows",
        "exe": "realesrgan-ncnn-vulkan.exe",
        "models_dir": "models",
        "models": {
            2: "realesr-animevideov3-x2",
            3: "realesr-animevideov3-x3",
            4: "realesrgan-x4plus-anime"
        }
    },
    "waifu2x": {
        "dir": MODEL_DIR / "waifu2x-ncnn-vulkan-20250915-windows",
        "exe": "waifu2x-ncnn-vulkan.exe",
        "models_dir": "models-cunet"
    },
    "anime4k": {
        "dir": MODEL_DIR / "Anime4KCPP-CLI-v3.0.0-x64-MSVC",
        "exe": "ac_cli.exe",
        "models": [
            "acnet-gan",       # GAN 增强模型 (默认, 质量更好)
            "acnet",           # 标准 CNN 模型 (速度更快)
        ],
        "processors": ["cpu", "opencl", "cuda"]  # 支持的处理器
    }
}

# FFmpeg 配置
FFMPEG_DIR = BASE_DIR / "前置" / "ffmpeg-8.0.1-essentials_build" / "bin"
FFMPEG_EXE = FFMPEG_DIR / "ffmpeg.exe"
FFPROBE_EXE = FFMPEG_DIR / "ffprobe.exe"

# 预设配置
# 预设配置 - 使用 key 而非直接文本，便于 i18n
PRESET_KEYS = ["preset_universal", "preset_repair", "preset_wallpaper", "preset_soft", "preset_anime4k"]

def get_presets() -> Dict[str, Dict[str, Any]]:
    """获取当前语言的预设配置"""
    return {
        t("preset_universal"): {
            "engine": "cugan",
            "params": {"scale": 2, "denoise": 0, "model": "Pro"},
            "desc": t("preset_universal_desc")
        },
        t("preset_repair"): {
            "engine": "esrgan",
            "params": {"scale": 4},
            "desc": t("preset_repair_desc")
        },
        t("preset_wallpaper"): {
            "engine": "cugan",
            "params": {"scale": 4, "denoise": -1, "model": "SE"},
            "desc": t("preset_wallpaper_desc")
        },
        t("preset_soft"): {
            "engine": "waifu2x",
            "params": {"scale": 2, "denoise": 3},
            "desc": t("preset_soft_desc")
        },
        t("preset_anime4k"): {
            "engine": "anime4k",
            "params": {"scale": 2, "model_name": "acnet-gan", "processor": "cuda"},
            "desc": t("preset_anime4k_desc")
        }
    }

# 兼容旧代码
PRESETS: Dict[str, Dict[str, Any]] = {
    "通用增强": {
        "engine": "cugan",
        "params": {"scale": 2, "denoise": 0, "model": "Pro"},
        "desc": "Real-CUGAN Pro 2x 保守降噪, 适合大多数场景"
    },
    "烂图修复": {
        "engine": "esrgan",
        "params": {"scale": 4},
        "desc": "Real-ESRGAN 4x, 强力修复低质量图片"
    },
    "壁纸制作": {
        "engine": "cugan",
        "params": {"scale": 4, "denoise": -1, "model": "SE"},
        "desc": "Real-CUGAN SE 4x 无降噪, 保留细节制作高清壁纸"
    },
    "极致柔化": {
        "engine": "waifu2x",
        "params": {"scale": 2, "denoise": 3},
        "desc": "Waifu2x 2x 强力降噪, 画面柔和细腻"
    },
    "动漫快处理": {
        "engine": "anime4k",
        "params": {"scale": 2, "model_name": "acnet-gan", "processor": "cuda"},
        "desc": "Anime4K 2x 快速处理, 适合动图与视频这类动画帧较多的文件"
    },
    "快速超分": {
        "engine": "anime4k",
        "params": {"scale": 2, "model_name": "acnet-gan", "processor": "cuda"},
        "desc": "Anime4K 2x 快速处理, 适合动图与视频这类动画帧较多的文件"
    }
}

def get_unique_path(filename: str) -> Path:
    """生成唯一的输出文件路径, 避免覆盖已有文件"""
    target = OUTPUT_DIR / filename
    if not target.exists():
        return target
    
    stem, suffix = target.stem, target.suffix
    counter = 1
    while target.exists():
        target = OUTPUT_DIR / f"{stem}_{counter}{suffix}"
        counter += 1
    return target


def write_ais_metadata(image_path: Path, metadata: Dict[str, Any]) -> bool:
    """将AIS元数据写入PNG图片"""
    # 仅对 PNG 格式写入元数据
    if image_path.suffix.lower() != '.png':
        return True  # 非 PNG 格式跳过，不视为失败
    
    try:
        with Image.open(image_path) as img:
            pnginfo = PngInfo()
            
            # 构建AIS元数据字符串
            ais_data = {
                "AIS": "AI Image Super-Resolution",
                "version": "1.0",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **metadata
            }
            
            # 写入元数据
            pnginfo.add_text("AIS", json.dumps(ais_data, ensure_ascii=False))
            pnginfo.add_text("AIS_Engine", str(metadata.get("engine", "unknown")))
            pnginfo.add_text("AIS_Scale", str(metadata.get("scale", 0)))
            if "denoise" in metadata:
                pnginfo.add_text("AIS_Denoise", str(metadata.get("denoise", 0)))
            if "model" in metadata:
                pnginfo.add_text("AIS_Model", str(metadata.get("model", "")))
            
            # 保存带元数据的图片
            img.save(image_path, pnginfo=pnginfo)
        return True
    except (IOError, OSError) as e:
        log_info(f"写入元数据失败: {e}")
        return False


def read_ais_metadata(image_path: str) -> Optional[Dict[str, Any]]:
    """从PNG图片读取AIS元数据"""
    try:
        img = Image.open(image_path)
        if hasattr(img, 'info') and 'AIS' in img.info:
            return json.loads(img.info['AIS'])
        return None
    except Exception:
        return None


def format_metadata_display(metadata: Optional[Dict[str, Any]]) -> str:
    """格式化元数据用于显示"""
    if not metadata:
        return "无AIS元数据"
    
    lines = [
        "=== AIS 超分信息 ===",
        f"处理时间: {metadata.get('timestamp', '未知')}",
        f"引擎: {metadata.get('engine', '未知').upper()}",
        f"放大倍率: {metadata.get('scale', '未知')}x",
    ]
    
    if 'denoise' in metadata:
        lines.append(f"降噪等级: {metadata.get('denoise')}")
    if 'model' in metadata:
        lines.append(f"模型: {metadata.get('model')}")
    if 'source_file' in metadata:
        lines.append(f"原文件: {metadata.get('source_file')}")
    
    return "\n".join(lines)


def run_command(cmd: List[str], cwd: Path) -> Tuple[bool, str, str]:
    """执行命令并返回结果
    返回: (成功标志, 消息, 处理日志)
    """
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        # 合并stdout和stderr作为日志
        log_output = ""
        if result.stdout:
            log_output += result.stdout.strip()
        if result.stderr:
            if log_output:
                log_output += "\n"
            log_output += result.stderr.strip()
        return True, "成功", log_output
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else str(e)
        log_output = ""
        if e.stdout:
            log_output += e.stdout.strip()
        if e.stderr:
            if log_output:
                log_output += "\n"
            log_output += e.stderr.strip()
        return False, error_msg, log_output
    except FileNotFoundError:
        return False, "引擎可执行文件未找到", ""
    except Exception as e:
        return False, str(e), ""


def check_engines() -> Dict[str, bool]:
    """检查各引擎是否可用"""
    status = {
        name: (config["dir"] / config["exe"]).exists()
        for name, config in ENGINES.items()
    }
    # 添加FFmpeg状态
    status["ffmpeg"] = FFMPEG_EXE.exists()
    return status


def build_cugan_command(input_path: Path, output_path: Path, 
                        scale: int, denoise: int, model: str,
                        tile_size: int = 0, syncgap: int = 3,
                        gpu_id: int = -2, threads: str = "1:2:2",
                        tta_mode: bool = False, output_format: str = "png") -> Tuple[List[str], Path]:
    """构建 Real-CUGAN 命令
    
    参数:
        scale: 放大倍率 (1/2/3/4)
        denoise: 降噪等级 (-1/0/1/2/3)
        model: 模型版本 (SE/Pro)
        tile_size: Tile大小 (>=32, 0=自动)
        syncgap: 同步模式 (0/1/2/3)
        gpu_id: GPU设备 (-1=CPU, -2=自动, 0/1/2...=指定GPU)
        threads: 线程数 (load:proc:save)
        tta_mode: TTA模式增强 (8倍时间换取更好效果)
        output_format: 输出格式 (png/jpg/webp)
    """
    config = ENGINES["cugan"]
    model_dir = config["dir"] / config["models"][model]
    cmd = [
        str(config["dir"] / config["exe"]),
        "-i", str(input_path),
        "-o", str(output_path),
        "-s", str(scale),
        "-n", str(denoise),
        "-m", str(model_dir),
        "-t", str(tile_size),
        "-c", str(syncgap),
        "-j", threads,
        "-f", output_format
    ]
    if gpu_id != -2:  # -2表示自动
        cmd.extend(["-g", str(gpu_id)])
    if tta_mode:
        cmd.append("-x")
    return cmd, config["dir"]


def build_esrgan_command(input_path: Path, output_path: Path, 
                         scale: int, model_name: str = "auto",
                         tile_size: int = 0, gpu_id: int = -2,
                         threads: str = "1:2:2", tta_mode: bool = False,
                         output_format: str = "png") -> Tuple[List[str], Path]:
    """构建 Real-ESRGAN 命令
    
    参数:
        scale: 放大倍率 (2/3/4)
        model_name: 模型名称 (auto/realesr-animevideov3/realesrgan-x4plus/realesrgan-x4plus-anime)
        tile_size: Tile大小 (>=32, 0=自动)
        gpu_id: GPU设备 (-1=CPU, -2=自动, 0/1/2...=指定GPU)
        threads: 线程数 (load:proc:save)
        tta_mode: TTA模式增强
        output_format: 输出格式 (png/jpg/webp)
    """
    config = ENGINES["esrgan"]
    # 模型选择: auto时根据倍率自动选择
    if model_name == "auto":
        model_name = config["models"].get(scale, config["models"][4])
    cmd = [
        str(config["dir"] / config["exe"]),
        "-i", str(input_path),
        "-o", str(output_path),
        "-s", str(scale),
        "-n", model_name,
        "-m", str(config["dir"] / config["models_dir"]),
        "-t", str(tile_size),
        "-j", threads,
        "-f", output_format
    ]
    if gpu_id != -2:
        cmd.extend(["-g", str(gpu_id)])
    if tta_mode:
        cmd.append("-x")
    return cmd, config["dir"]


def build_waifu2x_command(input_path: Path, output_path: Path,
                          scale: int, denoise: int, model_type: str = "cunet",
                          tile_size: int = 0, gpu_id: int = -2,
                          threads: str = "1:2:2", tta_mode: bool = False,
                          output_format: str = "png") -> Tuple[List[str], Path]:
    """构建 Waifu2x 命令
    
    参数:
        scale: 放大倍率 (1/2/4/8/16/32)
        denoise: 降噪等级 (-1/0/1/2/3)
        model_type: 模型类型 (cunet/upconv_7_anime_style_art_rgb/upconv_7_photo)
        tile_size: Tile大小 (>=32, 0=自动)
        gpu_id: GPU设备 (-1=CPU, -2=自动, 0/1/2...=指定GPU)
        threads: 线程数 (load:proc:save)
        tta_mode: TTA模式增强
        output_format: 输出格式 (png/jpg/webp)
    """
    config = ENGINES["waifu2x"]
    model_dir_name = f"models-{model_type}"
    cmd = [
        str(config["dir"] / config["exe"]),
        "-i", str(input_path),
        "-o", str(output_path),
        "-s", str(scale),
        "-n", str(denoise),
        "-m", str(config["dir"] / model_dir_name),
        "-t", str(tile_size),
        "-j", threads,
        "-f", output_format
    ]
    if gpu_id != -2:
        cmd.extend(["-g", str(gpu_id)])
    if tta_mode:
        cmd.append("-x")
    return cmd, config["dir"]


def build_anime4k_command(input_path: Path, output_path: Path,
                          scale: float = 2.0, model: str = "acnet-gan",
                          processor: str = "opencl", device: int = 0) -> Tuple[List[str], Path]:
    """构建 Anime4KCPP 命令
    
    参数:
        scale: 放大倍率 (支持小数，如 1.5, 2, 2.5, 3, 4)
        model: 模型名称 (acnet/acnet-gan)
        processor: 处理器类型 (cpu/opencl/cuda)
        device: 设备索引 (0, 1, 2...)
    """
    config = ENGINES["anime4k"]
    cmd = [
        str(config["dir"] / config["exe"]),
        "-i", str(input_path),
        "-o", str(output_path),
        "-f", str(scale),          # factor 放大倍率
        "-m", model,               # 模型
        "-p", processor,           # 处理器
        "-d", str(device)          # 设备索引
    ]
    return cmd, config["dir"]


def process_image(input_path: str, engine: str, **params) -> Tuple[Optional[str], str]:
    """处理单张图片
    
    通用参数:
        tile_size: Tile大小 (0=自动)
        gpu_id: GPU设备 (-2=自动, -1=CPU, 0/1/2...=指定GPU)
        threads: 线程数 (load:proc:save)
        tta_mode: TTA模式
        output_format: 输出格式 (png/jpg/webp)
    
    返回:
        (输出路径, 状态消息) - 成功时输出路径为文件路径，失败时为 None
    """
    if not input_path:
        return None, "[错误] 请先上传图片"
    
    input_file = Path(input_path)
    if not input_file.exists():
        return None, "[错误] 输入文件不存在"
    
    # 验证文件是否为有效图片
    try:
        with Image.open(input_file) as img:
            img.verify()
    except Exception:
        return None, "[错误] 无效的图片文件"
    
    engine_status = check_engines()
    if not engine_status.get(engine, False):
        return None, f"[错误] {engine} 引擎不可用"
    
    # 通用高级参数
    tile_size = params.get("tile_size", 0)
    gpu_id = params.get("gpu_id", -2)
    threads = params.get("threads", "1:2:2")
    tta_mode = params.get("tta_mode", False)
    output_format = params.get("output_format", "png")
    
    # 准备元数据
    metadata: Dict[str, Any] = {
        "engine": engine,
        "source_file": input_file.name
    }
    
    # 根据引擎构建命令
    if engine == "cugan":
        scale = params.get("scale", 2)
        denoise = params.get("denoise", 0)
        model = params.get("model", "Pro")
        syncgap = params.get("syncgap", 3)
        out_name = f"{input_file.stem}_CUGAN_{model}_{scale}x_n{denoise}.{output_format}"
        out_path = get_unique_path(out_name)
        cmd, cwd = build_cugan_command(
            input_file, out_path, scale, denoise, model,
            tile_size=tile_size, syncgap=syncgap, gpu_id=gpu_id,
            threads=threads, tta_mode=tta_mode, output_format=output_format
        )
        metadata.update({"scale": scale, "denoise": denoise, "model": model, "tta": tta_mode})
        
    elif engine == "esrgan":
        scale = params.get("scale", 4)
        model_name = params.get("model_name", "auto")
        out_name = f"{input_file.stem}_ESRGAN_{scale}x.{output_format}"
        out_path = get_unique_path(out_name)
        cmd, cwd = build_esrgan_command(
            input_file, out_path, scale, model_name=model_name,
            tile_size=tile_size, gpu_id=gpu_id, threads=threads,
            tta_mode=tta_mode, output_format=output_format
        )
        metadata.update({"scale": scale, "model": model_name, "tta": tta_mode})
        
    elif engine == "waifu2x":
        scale = params.get("scale", 2)
        denoise = params.get("denoise", 1)
        model_type = params.get("model_type", "cunet")
        out_name = f"{input_file.stem}_Waifu_{scale}x_n{denoise}.{output_format}"
        out_path = get_unique_path(out_name)
        cmd, cwd = build_waifu2x_command(
            input_file, out_path, scale, denoise, model_type=model_type,
            tile_size=tile_size, gpu_id=gpu_id, threads=threads,
            tta_mode=tta_mode, output_format=output_format
        )
        metadata.update({"scale": scale, "denoise": denoise, "model": model_type, "tta": tta_mode})
    
    elif engine == "anime4k":
        scale = params.get("scale", 2.0)
        model = params.get("model_name", params.get("model", "acnet-gan"))
        processor = params.get("processor", "opencl")
        device = params.get("device", 0)
        out_name = f"{input_file.stem}_Anime4K_{model}_{scale}x.{output_format}"
        out_path = get_unique_path(out_name)
        cmd, cwd = build_anime4k_command(
            input_file, out_path, scale=scale, model=model,
            processor=processor, device=device
        )
        metadata.update({"scale": scale, "model": model, "processor": processor})
        
    else:
        return None, f"[错误] 未知引擎: {engine}"
    
    # 执行处理
    success, msg, log = run_command(cmd, cwd)
    
    if success and out_path.exists():
        # 写入AIS元数据
        write_ais_metadata(out_path, metadata)
        # 构建带日志的完成消息
        result_msg = f"[完成] 保存至: {out_path.name}"
        if log:
            result_msg += f"\n--- 处理日志 ---\n{log[:500]}"  # 限制日志长度
        return str(out_path), result_msg
    else:
        # 构建带日志的错误消息
        error_msg = f"[失败] {msg}"
        if log:
            error_msg += f"\n--- 处理日志 ---\n{log[:500]}"
        return None, error_msg

def process_with_preset(input_image, 
                        preset_name: str,
                        gif_output_format: str = "gif") -> Tuple[Optional[str], Optional[Tuple], Optional[str], Optional[str], str]:
    """使用预设处理图片（支持内置预设和固定的自定义预设）
    返回: (处理结果, 对比元组, 原图路径, 结果路径, 状态消息)
    """
    # 预处理图片输入（支持剪贴板粘贴）
    input_path = preprocess_image_input(input_image)
    if input_path is None:
        return None, None, None, None, "[错误] 请先上传图片"
    
    # 获取预设配置（支持内置和自定义预设）
    preset = get_preset_config(preset_name)
    if not preset:
        # 尝试从旧版 PRESETS 获取
        preset = PRESETS.get(preset_name)
    
    if not preset:
        return None, None, None, None, f"[错误] 未知预设: {preset_name}"
    
    engine = preset.get("engine", "cugan")
    params = preset.get("params", {})
    
    # 检查是否为GIF，如果是则使用GIF处理
    if is_animated_gif(input_path):
        output_path, result_msg = process_gif_image(input_path, engine, gif_output_format=gif_output_format, **params)
    else:
        output_path, result_msg = process_image(input_path, engine, **params)
    
    if output_path:
        return output_path, (input_path, output_path), input_path, output_path, result_msg
    return None, None, None, None, result_msg


def process_all_presets(input_image) -> Tuple[List[Optional[str]], str]:
    """执行所有预设并返回结果"""
    # 预处理图片输入（支持剪贴板粘贴）
    input_path = preprocess_image_input(input_image)
    if input_path is None:
        return [None] * 4, "[错误] 请先上传图片"
    
    results: List[Optional[str]] = []
    messages: List[str] = []
    
    for preset_name, preset in PRESETS.items():
        messages.append(f"正在执行: {preset_name}")
        output_path, msg = process_image(
            input_path,
            preset["engine"],
            **preset["params"]
        )
        
        if output_path:
            results.append(output_path)
            messages.append(f"  [OK] {preset_name} 完成")
        else:
            results.append(None)
            messages.append(f"  [X] {preset_name} 失败: {msg}")
    
    # 确保返回4个结果
    while len(results) < 4:
        results.append(None)
    
    return results, "\n".join(messages)


def process_custom(input_image, engine: str, 
                   cugan_model: str, cugan_scale: int, cugan_denoise: str,
                   esrgan_scale: int,
                   waifu_scale: int, waifu_denoise: int) -> Tuple[Optional[str], Optional[Tuple], Optional[str], Optional[str], str]:
    """自定义模式处理
    返回: (处理结果, 对比元组, 原图路径, 结果路径, 状态消息)
    """
    # 预处理图片输入（支持剪贴板粘贴）
    input_path = preprocess_image_input(input_image)
    if input_path is None:
        return None, None, None, None, "[错误] 请先上传图片"
    
    output: Optional[str] = None
    msg: str = ""
    
    if engine == "Real-CUGAN":
        model = "Pro" if "Pro" in cugan_model else "SE"
        denoise_map = {"无降噪": -1, "保守降噪": 0, "强力降噪": 3}
        denoise = denoise_map.get(cugan_denoise, 0)
        output, msg = process_image(
            input_path, "cugan", 
            scale=int(cugan_scale), denoise=denoise, model=model
        )
    
    elif engine == "Real-ESRGAN":
        output, msg = process_image(
            input_path, "esrgan", 
            scale=int(esrgan_scale)
        )
    
    elif engine == "Waifu2x":
        output, msg = process_image(
            input_path, "waifu2x",
            scale=int(waifu_scale), denoise=int(waifu_denoise)
        )
    else:
        return None, None, None, None, "[错误] 请选择引擎"
    
    if output:
        return output, (input_path, output), input_path, output, msg
    return None, None, None, None, msg


# ============================================================
# 配置管理 (JSON格式，存放在数据文件夹)
# ============================================================

CONFIG_FILE = DATA_DIR / "config.json"

def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    if CONFIG_FILE.exists():
        try:
            content = CONFIG_FILE.read_text(encoding='utf-8')
            return json.loads(content)
        except Exception:
            pass
    return {}


def save_config(config: Dict[str, Any]) -> bool:
    """保存配置文件"""
    try:
        content = json.dumps(config, ensure_ascii=False, indent=2)
        CONFIG_FILE.write_text(content, encoding='utf-8')
        return True
    except Exception as e:
        print(f"[警告] 无法保存配置: {e}")
        return False


def load_share_config() -> bool:
    """从配置文件加载分享设置"""
    config = load_config()
    return config.get("share_enabled", False)


def save_share_config(enabled: bool) -> None:
    """保存分享设置到配置文件"""
    config = load_config()
    config["share_enabled"] = enabled
    save_config(config)


# ============================================================
# 自定义预设管理
# ============================================================

CUSTOM_PRESETS_FILE = DATA_DIR / "presets.json"


def load_custom_presets() -> Dict[str, Dict[str, Any]]:
    """加载用户自定义预设"""
    if CUSTOM_PRESETS_FILE.exists():
        try:
            content = CUSTOM_PRESETS_FILE.read_text(encoding='utf-8')
            return json.loads(content)
        except Exception:
            pass
    return {}


def save_custom_presets(presets: Dict[str, Dict[str, Any]]) -> bool:
    """保存用户自定义预设"""
    try:
        content = json.dumps(presets, ensure_ascii=False, indent=2)
        CUSTOM_PRESETS_FILE.write_text(content, encoding='utf-8')
        return True
    except Exception as e:
        print(f"[警告] 保存预设失败: {e}")
        return False


def get_custom_preset_names() -> List[str]:
    """获取所有自定义预设名称"""
    presets = load_custom_presets()
    return list(presets.keys())


def add_custom_preset(name: str, engine: str, params: Dict[str, Any]) -> Tuple[bool, str]:
    """添加自定义预设"""
    if not name or not name.strip():
        return False, "预设名称不能为空"
    
    name = name.strip()
    presets = load_custom_presets()
    
    presets[name] = {
        "engine": engine,
        "params": params,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if save_custom_presets(presets):
        return True, f"[完成] 预设 '{name}' 已保存"
    return False, "[错误] 保存失败"


def delete_custom_preset(name: str) -> Tuple[bool, str]:
    """删除自定义预设"""
    presets = load_custom_presets()
    if name in presets:
        del presets[name]
        if save_custom_presets(presets):
            return True, f"[完成] 预设 '{name}' 已删除"
    return False, f"[错误] 预设 '{name}' 不存在"


def rename_custom_preset(old_name: str, new_name: str) -> Tuple[bool, str]:
    """重命名自定义预设"""
    if not new_name or not new_name.strip():
        return False, "新名称不能为空"
    
    new_name = new_name.strip()
    presets = load_custom_presets()
    
    if old_name not in presets:
        return False, f"[错误] 预设 '{old_name}' 不存在"
    
    if new_name in presets:
        return False, f"[错误] 预设 '{new_name}' 已存在"
    
    presets[new_name] = presets.pop(old_name)
    if save_custom_presets(presets):
        return True, f"[完成] 已重命名为 '{new_name}'"
    return False, "[错误] 保存失败"


# ============================================================
# 固定预设功能 - 将自定义预设固定到首页
# ============================================================

def get_pinned_presets() -> List[str]:
    """获取固定到首页的预设列表"""
    config = load_config()
    return config.get("pinned_presets", [])


def set_pinned_presets(preset_names: List[str]) -> bool:
    """设置固定到首页的预设列表"""
    config = load_config()
    config["pinned_presets"] = preset_names
    return save_config(config)


def pin_preset(name: str) -> Tuple[bool, str]:
    """将预设固定到首页"""
    pinned = get_pinned_presets()
    if name in pinned:
        return False, f"预设 '{name}' 已经固定"
    
    # 检查预设是否存在
    presets = load_custom_presets()
    if name not in presets:
        return False, f"预设 '{name}' 不存在"
    
    pinned.append(name)
    if set_pinned_presets(pinned):
        return True, f"[完成] 已将 '{name}' 固定到首页"
    return False, "[错误] 保存失败"


def unpin_preset(name: str) -> Tuple[bool, str]:
    """取消预设固定"""
    pinned = get_pinned_presets()
    if name not in pinned:
        return False, f"预设 '{name}' 未固定"
    
    pinned.remove(name)
    if set_pinned_presets(pinned):
        return True, f"[完成] 已取消 '{name}' 的固定"
    return False, "[错误] 保存失败"


def get_all_preset_choices() -> List[str]:
    """获取所有预设选项（内置预设 + 固定的自定义预设）"""
    # 内置预设
    builtin = list(get_presets().keys())
    
    # 获取固定的自定义预设
    pinned = get_pinned_presets()
    custom_presets = load_custom_presets()
    
    # 只添加存在的固定预设
    for name in pinned:
        if name in custom_presets and name not in builtin:
            builtin.append(f"⭐ {name}")  # 用星号标记自定义预设
    
    return builtin


def get_author_preset_choices() -> List[str]:
    """获取作者预设列表"""
    return list(get_presets().keys())


def get_user_preset_choices() -> List[str]:
    """获取用户固定的预设列表"""
    pinned = get_pinned_presets()
    custom_presets = load_custom_presets()
    
    # 只返回存在的固定预设
    valid_presets = []
    for name in pinned:
        if name in custom_presets:
            valid_presets.append(name)
    
    return valid_presets


def get_preset_config(preset_name: str) -> Optional[Dict[str, Any]]:
    """获取预设配置（支持内置和自定义预设）"""
    # 检查 None 或空字符串
    if not preset_name:
        return None
    
    # 处理带星号的自定义预设名称（只处理一次）
    if preset_name.startswith("⭐ "):
        preset_name = preset_name[2:]
    
    # 先检查自定义预设（用户预设优先）
    custom = load_custom_presets()
    if preset_name in custom:
        preset_data = custom[preset_name]
        # 转换为内置预设格式
        if "all_params" in preset_data:
            # 新格式预设 - 使用默认引擎参数
            params = preset_data["all_params"]
            # 使用cugan作为默认
            cugan_p = params.get("cugan", {})
            return {
                "engine": "cugan",
                "params": {
                    "scale": cugan_p.get("scale", 2),
                    "denoise": cugan_p.get("denoise", 0),
                    "model": cugan_p.get("model", "Pro")
                },
                "desc": f"自定义预设: {preset_name}"
            }
        else:
            return preset_data
    
    # 检查内置预设
    builtin = get_presets()
    if preset_name in builtin:
        return builtin[preset_name]
    
    # 检查旧版内置预设
    if preset_name in PRESETS:
        return PRESETS[preset_name]
    
    return None


# ============================================================
# GIF 超分处理
# ============================================================

def process_gif_image(input_path: str, engine: str, gif_output_format: str = "gif", **params) -> Tuple[Optional[str], str]:
    """处理GIF图片 - 逐帧超分后重组
    
    参数:
        input_path: 输入GIF路径
        engine: 超分引擎
        gif_output_format: 动图输出格式 (gif/webp)
            - gif: 传统GIF格式，256色限制，兼容性最好
            - webp: WebP动图，支持24-bit颜色，无色带问题，文件更小
        **params: 引擎参数
    
    返回: (输出路径, 状态消息)
    """
    if not is_animated_gif(input_path):
        # 不是动态GIF，使用普通处理
        return process_image(input_path, engine, **params)
    
    log_info(f"[GIF] 开始处理动态GIF: {Path(input_path).name}，输出格式: {gif_output_format.upper()}")
    
    # 始终使用PIL提取帧（保留每帧精确时间）
    frame_paths, durations, error = extract_gif_frames(input_path)
    
    if error or not frame_paths:
        return None, f"[错误] GIF帧提取失败: {error}"
    
    total_frames = len(frame_paths)
    log_info(f"[GIF] 共 {total_frames} 帧，每帧时间: {durations[:5]}... ms")
    
    # 获取帧所在目录
    frames_dir = Path(frame_paths[0]).parent
    # 创建处理后帧的输出目录
    processed_dir = TEMP_DIR / f"gif_processed_{uuid.uuid4().hex[:8]}"
    processed_dir.mkdir(exist_ok=True)
    processed_frames = []
    
    try:
        for i, frame_path in enumerate(frame_paths):
            log_info(f"[GIF] 处理帧 {i+1}/{total_frames}")
            
            # 处理单帧
            output_path, msg = process_image(frame_path, engine, **params)
            
            if output_path and Path(output_path).exists():
                # 将处理后的帧复制到临时目录，保持顺序
                ordered_frame_path = processed_dir / f"processed_{i:04d}.png"
                try:
                    import shutil
                    shutil.copy2(output_path, ordered_frame_path)
                    processed_frames.append(str(ordered_frame_path))
                    # 删除输出目录中的临时帧文件
                    Path(output_path).unlink()
                except Exception as e:
                    log_info(f"[GIF] 帧 {i+1} 复制失败: {e}")
                    processed_frames.append(frame_path)
            else:
                # 帧处理失败，使用原帧
                log_info(f"[GIF] 帧 {i+1} 处理失败，使用原帧")
                processed_frames.append(frame_path)
        
        # 根据输出格式选择组装方法
        input_file = Path(input_path)
        
        # 计算帧率（用于FFmpeg）
        avg_duration = sum(durations) / len(durations) if durations else 100
        fps = 1000.0 / avg_duration if avg_duration > 0 else 10.0
        
        if gif_output_format.lower() == "webp":
            # WebP 动图 - 24-bit 颜色，无色带
            out_name = f"{input_file.stem}_{engine.upper()}_animated.webp"
            out_path = get_unique_path(out_name)
            
            if FFMPEG_EXE.exists():
                success, assemble_msg = reassemble_webp_ffmpeg(processed_frames, fps, str(out_path))
            else:
                success, assemble_msg = reassemble_webp(processed_frames, durations, str(out_path))
            
            format_name = "WebP动图"
        else:
            # GIF 格式 - 256色，兼容性好
            out_name = f"{input_file.stem}_{engine.upper()}_animated.gif"
            out_path = get_unique_path(out_name)
            
            if FFMPEG_EXE.exists():
                success, assemble_msg = reassemble_gif_ffmpeg(processed_frames, fps, str(out_path))
            else:
                success, assemble_msg = reassemble_gif(processed_frames, durations, str(out_path))
            
            format_name = "GIF"
        
        if success:
            log_info(f"[{format_name}] 处理完成: {out_path.name}")
            return str(out_path), f"[完成] {format_name}处理完成，共{total_frames}帧\n保存至: {out_path.name}"
        else:
            return None, f"[错误] {assemble_msg}"
    
    finally:
        # 清理临时文件
        cleanup_gif_temp(frames_dir)
        cleanup_gif_temp(processed_dir)


# ============================================================
# 图库功能
# ============================================================

def get_gallery_images() -> List[str]:
    """获取输出目录中的所有图片路径"""
    if not OUTPUT_DIR.exists():
        return []
    
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
    images = []
    
    try:
        for f in OUTPUT_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in extensions:
                images.append(str(f))
        
        # 按修改时间倒序排列(最新的在前)
        images.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    except (OSError, PermissionError) as e:
        log_info(f"读取图库失败: {e}")
        return []
    
    return images


# 图库索引 - 用于缩略图和原图的映射
_gallery_image_map: Dict[str, str] = {}


def get_gallery_with_thumbnails() -> Tuple[List[str], Dict[str, str]]:
    """获取图库缩略图列表和映射关系
    
    返回: (缩略图路径列表, {缩略图路径: 原图路径} 映射)
    """
    global _gallery_image_map
    _gallery_image_map = {}
    
    if not OUTPUT_DIR.exists():
        return [], {}
    
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
    thumbnails = []
    
    try:
        images = []
        for f in OUTPUT_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in extensions:
                images.append(str(f))
        
        # 按修改时间倒序排列
        images.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
        
        # 为每张图片创建缩略图
        for img_path in images:
            thumb = create_thumbnail(img_path)
            if thumb:
                thumbnails.append(thumb)
                _gallery_image_map[thumb] = img_path
            else:
                # 缩略图创建失败时使用原图
                thumbnails.append(img_path)
                _gallery_image_map[img_path] = img_path
    except (OSError, PermissionError) as e:
        log_info(f"读取图库失败: {e}")
    
    return thumbnails, _gallery_image_map


def get_original_from_thumbnail(thumb_path: str) -> str:
    """根据缩略图路径获取原图路径"""
    return _gallery_image_map.get(thumb_path, thumb_path)


def get_image_info(image_path: Optional[str]) -> Tuple[Optional[str], str]:
    """获取图片信息和元数据"""
    if not image_path:
        return None, "请选择一张图片"
    
    path = Path(image_path)
    if not path.exists():
        return None, "图片不存在"
    
    # 基本文件信息
    stat = path.stat()
    size_mb = stat.st_size / (1024 * 1024)
    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    
    info_lines = [
        "=== 文件信息 ===",
        f"文件名: {path.name}",
        f"大小: {size_mb:.2f} MB",
        f"修改时间: {mtime}",
    ]
    
    # 尝试获取图片尺寸
    try:
        with Image.open(path) as img:
            info_lines.append(f"尺寸: {img.width} x {img.height}")
            info_lines.append(f"格式: {img.format}")
    except Exception:
        pass
    
    info_lines.append("")
    
    # 读取AIS元数据
    metadata = read_ais_metadata(image_path)
    if metadata:
        info_lines.append(format_metadata_display(metadata))
    else:
        info_lines.append("=== AIS 超分信息 ===")
        info_lines.append("非AIS处理的图片")
    
    return image_path, "\n".join(info_lines)


def delete_gallery_image(image_path: Optional[str]) -> Tuple[List[str], str]:
    """删除图库中的图片（仅允许删除输出目录中的文件）"""
    if not image_path:
        return get_gallery_images(), "请先选择要删除的图片"
    
    path = Path(image_path).resolve()
    
    # 安全检查：确保只能删除输出目录中的文件
    try:
        path.relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        return get_gallery_images(), "[错误] 只能删除输出目录中的文件"
    
    if path.exists():
        try:
            path.unlink()
            return get_gallery_images(), f"[完成] 已删除: {path.name}"
        except (OSError, PermissionError) as e:
            return get_gallery_images(), f"[错误] 删除失败: {e}"
    
    return get_gallery_images(), "图片不存在"


def refresh_share_url() -> str:
    """刷新公开链接显示 - 从日志文件读取"""
    global SHARE_URL
    if SHARE_URL:
        return SHARE_URL
    
    # 从日志文件读取公开链接
    if LOG_FILE.exists():
        try:
            import re
            content = LOG_FILE.read_text(encoding='utf-8')
            match = re.search(r'https://[a-zA-Z0-9-]+\.gradio\.live', content)
            if match:
                url = match.group(0)
                SHARE_URL = url
                return url
        except (IOError, OSError, UnicodeDecodeError):
            pass
    
    # 尝试从专用文件读取
    url = load_share_url_from_file()
    if url:
        return url
    
    if load_share_config():
        return t("generating") if get_current_lang() == "en" else "公开链接正在生成中, 请稍后刷新..."
    return t("not_enabled") if get_current_lang() == "en" else "未启用公开链接"


# ============================================================
# Gradio UI
# ============================================================

def get_engine_status_text() -> str:
    """获取引擎状态文本"""
    engine_status = check_engines()
    # 引擎显示名称映射
    display_names = {
        "cugan": "Real-CUGAN",
        "esrgan": "Real-ESRGAN", 
        "waifu2x": "Waifu2x",
        "anime4k": "Anime4K",
        "ffmpeg": "FFmpeg"
    }
    status_list = []
    for name, available in engine_status.items():
        icon = "✓" if available else "✗"
        display_name = display_names.get(name, name.upper())
        status_list.append(f"{icon} {display_name}")
    return " | ".join(status_list)


# 剪贴板粘贴 JavaScript 代码
# 使用 head 参数在 launch() 时注入，这是 Gradio 6.0 的正确方式
CLIPBOARD_PASTE_JS = """
<script>
(function() {
    console.log('[AIS] 剪贴板粘贴脚本开始加载...');
    
    function initPasteHandler() {
        console.log('[AIS] 正在初始化剪贴板粘贴功能...');
        
        // 检查是否已经初始化
        if (window._aisPasteInitialized) {
            console.log('[AIS] 粘贴功能已初始化，跳过');
            return;
        }
        window._aisPasteInitialized = true;
        
        // 监听全局粘贴事件
        document.addEventListener('paste', async function(e) {
            console.log('[AIS] 检测到粘贴事件');
            
            const items = e.clipboardData?.items;
            if (!items) {
                console.log('[AIS] 剪贴板无数据');
                return;
            }
            
            // 查找图片
            let imageFile = null;
            for (const item of items) {
                if (item.type.startsWith('image/')) {
                    imageFile = item.getAsFile();
                    break;
                }
            }
            
            if (!imageFile) {
                console.log('[AIS] 剪贴板中没有图片');
                return;
            }
            
            console.log('[AIS] 检测到剪贴板图片:', imageFile.type, imageFile.size, 'bytes');
            
            // 查找所有图片上传的 input 元素
            const fileInputs = document.querySelectorAll('input[type="file"]');
            console.log('[AIS] 找到', fileInputs.length, '个文件输入框');
            
            // 找到第一个接受图片的、可见的输入框
            let targetInput = null;
            for (const input of fileInputs) {
                const accept = input.getAttribute('accept') || '';
                if (accept.includes('image') || accept === '*/*' || accept === '') {
                    // 检查是否可见（父元素没有被隐藏）
                    const parent = input.closest('[data-testid="image"]') || input.parentElement;
                    if (parent && parent.offsetParent !== null) {
                        targetInput = input;
                        console.log('[AIS] 找到目标输入框');
                        break;
                    }
                }
            }
            
            if (!targetInput && fileInputs.length > 0) {
                // 使用第一个文件输入
                targetInput = fileInputs[0];
                console.log('[AIS] 使用第一个输入框作为备选');
            }
            
            if (!targetInput) {
                console.log('[AIS] 未找到可用的上传输入框');
                return;
            }
            
            try {
                // 创建 DataTransfer 并添加文件
                const dt = new DataTransfer();
                const newFile = new File([imageFile], 'clipboard_' + Date.now() + '.png', {type: imageFile.type});
                dt.items.add(newFile);
                
                // 设置到 input
                targetInput.files = dt.files;
                
                // 触发各种可能的事件
                targetInput.dispatchEvent(new Event('input', { bubbles: true }));
                targetInput.dispatchEvent(new Event('change', { bubbles: true }));
                
                // 显示成功提示
                const hint = document.createElement('div');
                hint.style.cssText = 'position:fixed;bottom:20px;right:20px;background:rgba(0,0,0,0.8);color:white;padding:10px 20px;border-radius:8px;z-index:9999;';
                hint.textContent = '✓ 图片已粘贴';
                document.body.appendChild(hint);
                setTimeout(() => hint.remove(), 2000);
                
                console.log('[AIS] 图片粘贴成功!');
                e.preventDefault();
            } catch (err) {
                console.error('[AIS] 粘贴失败:', err);
            }
        });
        
        console.log('[AIS] 剪贴板粘贴功能已启用 - 在页面任意位置按 Ctrl+V 粘贴图片');
    }
    
    // 使用多种方式确保初始化
    if (document.readyState === 'complete') {
        initPasteHandler();
    } else {
        window.addEventListener('load', initPasteHandler);
    }
    
    // 额外延迟确保 Gradio 组件加载完成
    setTimeout(initPasteHandler, 2000);
})();
</script>

<script>
// 图片加载优化 - 添加懒加载和错误重试
(function() {
    console.log('[AIS] 图片加载优化脚本加载中...');
    
    function initImageOptimization() {
        // 为所有图片添加懒加载
        const images = document.querySelectorAll('img');
        images.forEach(img => {
            if (!img.loading) {
                img.loading = 'lazy';
            }
        });
        
        // 监听图片加载错误并自动重试
        document.addEventListener('error', function(e) {
            if (e.target.tagName === 'IMG') {
                const img = e.target;
                const retryCount = parseInt(img.dataset.retryCount || '0');
                
                if (retryCount < 3) {
                    console.log('[AIS] 图片加载失败，重试中...', img.src);
                    img.dataset.retryCount = retryCount + 1;
                    
                    // 延迟重试
                    setTimeout(() => {
                        const originalSrc = img.src;
                        img.src = '';
                        img.src = originalSrc + (originalSrc.includes('?') ? '&' : '?') + 'retry=' + Date.now();
                    }, 1000 * (retryCount + 1));
                }
            }
        }, true);
        
        console.log('[AIS] 图片加载优化已启用');
    }
    
    // 使用 MutationObserver 监听新添加的图片
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeType === 1) {
                    const images = node.querySelectorAll ? node.querySelectorAll('img') : [];
                    images.forEach(img => {
                        if (!img.loading) {
                            img.loading = 'lazy';
                        }
                    });
                }
            });
        });
    });
    
    if (document.readyState === 'complete') {
        initImageOptimization();
        observer.observe(document.body, { childList: true, subtree: true });
    } else {
        window.addEventListener('load', function() {
            initImageOptimization();
            observer.observe(document.body, { childList: true, subtree: true });
        });
    }
})();
</script>
"""


def create_ui() -> gr.Blocks:
    """创建 Gradio 界面"""
    
    # 加载语言配置
    load_lang_config()
    
    engine_status = check_engines()
    status_text = get_engine_status_text()
    current_presets = get_presets()
    
    with gr.Blocks(title="AIS") as app:
        # 使用HTML隐藏底栏 + 自定义样式 (JavaScript 已移至 launch() 的 head 参数)
        gr.HTML("""
        <style>
        footer {display: none !important;}
        .gradio-container footer {display: none !important;}
        
        /* 仅针对图片预览顶部工具栏的图标按钮 (下载/全屏/分享) */
        .icon-button {
            min-width: 44px !important;
            min-height: 44px !important;
            padding: 10px !important;
        }
        .icon-button svg {
            width: 22px !important;
            height: 22px !important;
        }
        
        /* 引擎选项卡样式 */
        .engine-tabs .tab-nav button {
            font-size: 16px !important;
            padding: 12px 16px !important;
        }
        
        /* 缩略图图库样式 */
        .thumbnail-gallery img {
            object-fit: cover !important;
        }
        </style>
        """)
        
        # 标题栏 (语言设置已移至设置标签页)
        gr.Markdown(f"""
        # {t("app_title")}
        **[GitHub](https://github.com/SENyiAi/AIS)** | {t("app_subtitle")}: {status_text}
        """)
        
        # 获取预设选项（分类显示）
        author_preset_choices = get_author_preset_choices()
        user_preset_choices = get_user_preset_choices()
        
        with gr.Tabs():
            # 快速处理标签页
            with gr.Tab(t("tab_quick")):
                gr.Markdown(t("quick_desc"))
                
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        quick_input = gr.Image(
                            label=t("upload_image"),
                            type="filepath",
                            sources=["upload", "clipboard"],
                            height=280
                        )
                        
                        # 作者预设
                        gr.Markdown(f"**{t('author_presets')}**")
                        author_preset_radio = gr.Radio(
                            choices=author_preset_choices,
                            value=author_preset_choices[0] if author_preset_choices else None,
                            label=None,
                            show_label=False
                        )
                        
                        # 用户预设
                        gr.Markdown(f"**{t('user_presets')}**")
                        user_preset_radio = gr.Radio(
                            choices=user_preset_choices if user_preset_choices else [],
                            value=None,
                            label=None,
                            show_label=False
                        )
                        
                        # 刷新预设列表按钮
                        refresh_quick_presets_btn = gr.Button("🔄 刷新预设列表", variant="secondary", size="sm")
                        
                        # 当前选中预设的描述
                        quick_preset_desc = gr.Markdown(
                            value=f"[{t('msg_info').strip('[]')}] {list(current_presets.values())[0]['desc']}"
                        )
                        
                        # 动图输出格式选择
                        with gr.Accordion(t("gif_output_format"), open=False):
                            gr.Markdown(t("gif_format_info"))
                            quick_gif_output_format = gr.Radio(
                                choices=get_choices("gif_format"),
                                value="webp",
                                label=t("gif_output_format"),
                                show_label=False
                            )
                        
                        with gr.Row():
                            quick_btn = gr.Button(t("start_process"), variant="primary", scale=2)
                            quick_all_btn = gr.Button(t("process_all"), variant="secondary", scale=1)
                        
                        quick_status = gr.Textbox(label=t("status"), lines=6, interactive=False)
                        
                    with gr.Column(scale=2, min_width=500):
                        gr.Markdown(f"### {t('compare_title')}")
                        quick_compare = gr.ImageSlider(
                            label=t("compare_label"),
                            type="filepath"
                        )
                        with gr.Accordion(t("click_zoom"), open=False):
                            gr.Markdown(t("click_zoom_tip"))
                            with gr.Row():
                                zoom_original = gr.Image(
                                    label=t("original"),
                                    type="filepath",
                                    height=500,
                                    interactive=False
                                )
                                zoom_result = gr.Image(
                                    label=t("result"),
                                    type="filepath",
                                    height=500,
                                    interactive=False
                                )
                        with gr.Accordion(t("view_result"), open=False):
                            quick_output = gr.Image(
                                label=t("result_download"),
                                type="filepath",
                                height=400
                            )
                
                # 刷新首页预设列表的函数
                def refresh_quick_preset_choices():
                    """刷新首页预设选择列表"""
                    author_choices = get_author_preset_choices()
                    user_choices = get_user_preset_choices()
                    return (
                        gr.update(choices=author_choices, value=author_choices[0] if author_choices else None),
                        gr.update(choices=user_choices, value=None)
                    )
                
                refresh_quick_presets_btn.click(
                    fn=refresh_quick_preset_choices,
                    inputs=None,
                    outputs=[author_preset_radio, user_preset_radio]
                )
                
                # 全部预设结果对比 - 增强版
                with gr.Accordion(t("all_preset_compare"), open=False):
                    gr.Markdown(t("all_preset_desc"))
                    
                    # 存储所有结果的状态
                    all_results_state = gr.State(value={})
                    
                    preset_names = list(current_presets.keys())
                    with gr.Row():
                        all_result_1 = gr.Image(label=preset_names[0] if len(preset_names) > 0 else "1", type="filepath", height=220)
                        all_result_2 = gr.Image(label=preset_names[1] if len(preset_names) > 1 else "2", type="filepath", height=220)
                    with gr.Row():
                        all_result_3 = gr.Image(label=preset_names[2] if len(preset_names) > 2 else "3", type="filepath", height=220)
                        all_result_4 = gr.Image(label=preset_names[3] if len(preset_names) > 3 else "4", type="filepath", height=220)
                    
                    gr.Markdown(f"### {t('free_compare')}")
                    compare_choices = get_choices("compare_sources")
                    with gr.Row():
                        compare_left_choice = gr.Dropdown(
                            choices=compare_choices,
                            value=compare_choices[0] if compare_choices else "",
                            label=t("left_source")
                        )
                        compare_right_choice = gr.Dropdown(
                            choices=compare_choices,
                            value=compare_choices[1] if len(compare_choices) > 1 else "",
                            label=t("right_source")
                        )
                    
                    compare_slider = gr.ImageSlider(
                        label="<- " + t("left_source") + " | " + t("right_source") + " ->",
                        type="filepath"
                    )
                
                # 当前选中的预设（用于在两个Radio间切换）
                current_selected_preset = gr.State(value=author_preset_choices[0] if author_preset_choices else None)
                
                def update_preset_desc(preset_name: str) -> str:
                    """更新预设描述（支持内置和自定义预设）"""
                    if not preset_name:
                        return ""
                    # 先尝试从 get_preset_config 获取
                    preset = get_preset_config(preset_name)
                    if preset:
                        return f"[{t('msg_info').strip('[]')}] {preset.get('desc', '自定义预设')}"
                    
                    # 回退到内置预设
                    presets = get_presets()
                    preset = presets.get(preset_name, PRESETS.get(preset_name, {}))
                    return f"[{t('msg_info').strip('[]')}] {preset.get('desc', '')}"
                
                def on_author_preset_change(author_choice, user_choice):
                    """作者预设选择变化时，清空用户预设选择"""
                    if author_choice:
                        return author_choice, update_preset_desc(author_choice), gr.update(value=None)
                    # 如果 author_choice 为 None（被清空），保持当前用户预设选择
                    if user_choice:
                        return user_choice, update_preset_desc(user_choice), gr.update()
                    return gr.update(), "", gr.update()
                
                def on_user_preset_change(user_choice, author_choice):
                    """用户预设选择变化时，清空作者预设选择"""
                    if user_choice:
                        return user_choice, update_preset_desc(user_choice), gr.update(value=None)
                    # 如果 user_choice 为 None（被清空），保持当前作者预设选择
                    if author_choice:
                        return author_choice, update_preset_desc(author_choice), gr.update()
                    return gr.update(), "", gr.update()
                
                author_preset_radio.change(
                    fn=on_author_preset_change,
                    inputs=[author_preset_radio, user_preset_radio],
                    outputs=[current_selected_preset, quick_preset_desc, user_preset_radio]
                )
                
                user_preset_radio.change(
                    fn=on_user_preset_change,
                    inputs=[user_preset_radio, author_preset_radio],
                    outputs=[current_selected_preset, quick_preset_desc, author_preset_radio]
                )
                
                quick_btn.click(
                    fn=process_with_preset,
                    inputs=[quick_input, current_selected_preset, quick_gif_output_format],
                    outputs=[quick_output, quick_compare, zoom_original, zoom_result, quick_status]
                )
                
                def run_all_and_update(input_image):
                    """执行所有预设并返回结果"""
                    # 预处理图片输入（支持剪贴板粘贴）
                    input_path = preprocess_image_input(input_image)
                    if input_path is None:
                        return None, None, None, None, {}, None, "[错误] 请先上传图片"
                    
                    results, msg = process_all_presets(input_path)
                    # 构建结果字典用于对比选择
                    results_dict = {
                        "原图": input_path,
                        "通用增强": results[0],
                        "烂图修复": results[1],
                        "壁纸制作": results[2],
                        "极致柔化": results[3]
                    }
                    # 默认对比: 原图 vs 通用增强
                    default_compare = (input_path, results[0]) if results[0] else None
                    return results[0], results[1], results[2], results[3], results_dict, default_compare, msg
                
                quick_all_btn.click(
                    fn=run_all_and_update,
                    inputs=[quick_input],
                    outputs=[all_result_1, all_result_2, all_result_3, all_result_4, 
                             all_results_state, compare_slider, quick_status]
                )
                
                def update_compare_slider(left_choice, right_choice, results_dict):
                    """更新对比滑块"""
                    if not results_dict:
                        return None
                    left_img = results_dict.get(left_choice)
                    right_img = results_dict.get(right_choice)
                    if left_img and right_img:
                        return (left_img, right_img)
                    return None
                
                compare_left_choice.change(
                    fn=update_compare_slider,
                    inputs=[compare_left_choice, compare_right_choice, all_results_state],
                    outputs=[compare_slider]
                )
                
                compare_right_choice.change(
                    fn=update_compare_slider,
                    inputs=[compare_left_choice, compare_right_choice, all_results_state],
                    outputs=[compare_slider]
                )
            
            # 自定义模式标签页 - 使用Tabs切换引擎
            with gr.Tab(t("tab_custom")):
                gr.Markdown(t("custom_desc"))
                
                with gr.Row():
                    # 左侧：上传图片
                    with gr.Column(scale=1, min_width=280):
                        custom_input = gr.Image(
                            label=t("upload_image"),
                            type="filepath",
                            sources=["upload", "clipboard"],
                            height=220
                        )
                        
                        # 动图输出格式选项
                        gif_output_format = gr.Radio(
                            choices=get_choices("gif_format"),
                            value="webp",
                            label=t("gif_output_format"),
                            info=t("gif_output_format_info")
                        )
                        
                        # 使用Tabs切换不同引擎 - 每个引擎独立完整
                        with gr.Tabs(elem_classes=["engine-tabs"]) as engine_tabs:
                            # Real-CUGAN 标签页
                            with gr.Tab("Real-CUGAN", id="cugan"):
                                cugan_model = gr.Radio(
                                    choices=get_choices("cugan_model"),
                                    value=get_choices("cugan_model")[1] if len(get_choices("cugan_model")) > 1 else "",
                                    label=t("model_version"),
                                    info=t("model_version_info")
                                )
                                cugan_scale = gr.Slider(
                                    minimum=2, maximum=4, step=1, value=2,
                                    label=t("scale_ratio")
                                )
                                cugan_denoise = gr.Radio(
                                    choices=get_choices("cugan_denoise"),
                                    value=get_choices("cugan_denoise")[1] if len(get_choices("cugan_denoise")) > 1 else "",
                                    label=t("denoise_level"),
                                    info=t("denoise_level_info")
                                )
                                
                                with gr.Accordion(t("advanced_options"), open=False):
                                    cugan_syncgap = gr.Slider(
                                        minimum=0, maximum=3, step=1, value=3,
                                        label=t("syncgap_mode"),
                                        info=t("syncgap_info")
                                    )
                                    cugan_tile = gr.Slider(
                                        minimum=0, maximum=512, step=32, value=0,
                                        label=t("tile_size"),
                                        info=t("tile_info")
                                    )
                                    cugan_tta = gr.Checkbox(
                                        value=False,
                                        label=t("tta_mode"),
                                        info=t("tta_info")
                                    )
                                    cugan_gpu = gr.Dropdown(
                                        choices=get_choices("gpu"),
                                        value=-2,
                                        label=t("gpu_select")
                                    )
                                    cugan_threads = gr.Textbox(
                                        value="1:2:2",
                                        label=t("threads"),
                                        info=t("threads_info")
                                    )
                                    cugan_format = gr.Radio(
                                        choices=get_choices("format"),
                                        value="png",
                                        label=t("output_format")
                                    )
                                
                                cugan_btn = gr.Button("🚀 " + t("start_process"), variant="primary", elem_classes=["mobile-friendly-btn"])
                            
                            # Real-ESRGAN 标签页
                            with gr.Tab("Real-ESRGAN", id="esrgan"):
                                esrgan_model = gr.Dropdown(
                                    choices=get_choices("esrgan_model"),
                                    value="auto",
                                    label=t("esrgan_model_select"),
                                    info=t("esrgan_model_info")
                                )
                                esrgan_scale = gr.Radio(
                                    choices=[2, 3, 4],
                                    value=4,
                                    label=t("scale_ratio")
                                )
                                
                                with gr.Accordion(t("advanced_options"), open=False):
                                    esrgan_tile = gr.Slider(
                                        minimum=0, maximum=512, step=32, value=0,
                                        label=t("tile_size"),
                                        info=t("tile_info")
                                    )
                                    esrgan_tta = gr.Checkbox(
                                        value=False,
                                        label=t("tta_mode"),
                                        info=t("tta_info")
                                    )
                                    esrgan_gpu = gr.Dropdown(
                                        choices=get_choices("gpu"),
                                        value=-2,
                                        label=t("gpu_select")
                                    )
                                    esrgan_threads = gr.Textbox(
                                        value="1:2:2",
                                        label=t("threads")
                                    )
                                    esrgan_format = gr.Radio(
                                        choices=get_choices("format"),
                                        value="png",
                                        label=t("output_format")
                                    )
                                
                                esrgan_btn = gr.Button("🚀 " + t("start_process"), variant="primary", elem_classes=["mobile-friendly-btn"])
                            
                            # Waifu2x 标签页
                            with gr.Tab("Waifu2x", id="waifu2x"):
                                waifu_model = gr.Dropdown(
                                    choices=get_choices("waifu_model"),
                                    value="cunet",
                                    label=t("waifu_model_select"),
                                    info=t("waifu_model_info")
                                )
                                waifu_scale = gr.Slider(
                                    minimum=1, maximum=32, step=1, value=2,
                                    label=t("scale_ratio"),
                                    info=t("waifu_scale_info")
                                )
                                waifu_denoise = gr.Slider(
                                    minimum=-1, maximum=3, step=1, value=1,
                                    label=t("denoise_level"),
                                    info=t("waifu_denoise_info")
                                )
                                
                                with gr.Accordion(t("advanced_options"), open=False):
                                    waifu_tile = gr.Slider(
                                        minimum=0, maximum=512, step=32, value=0,
                                        label=t("tile_size"),
                                        info=t("tile_info")
                                    )
                                    waifu_tta = gr.Checkbox(
                                        value=False,
                                        label=t("tta_mode"),
                                        info=t("tta_info")
                                    )
                                    waifu_gpu = gr.Dropdown(
                                        choices=get_choices("gpu"),
                                        value=-2,
                                        label=t("gpu_select")
                                    )
                                    waifu_threads = gr.Textbox(
                                        value="1:2:2",
                                        label=t("threads")
                                    )
                                    waifu_format = gr.Radio(
                                        choices=get_choices("format"),
                                        value="png",
                                        label=t("output_format")
                                    )
                                
                                waifu_btn = gr.Button("🚀 " + t("start_process"), variant="primary", elem_classes=["mobile-friendly-btn"])
                            
                            # Anime4K 标签页
                            with gr.Tab("Anime4K", id="anime4k"):
                                anime4k_model = gr.Dropdown(
                                    choices=get_choices("anime4k_model"),
                                    value="acnet-gan",
                                    label=t("anime4k_model_select"),
                                    info=t("anime4k_model_info")
                                )
                                anime4k_factor = gr.Slider(
                                    minimum=2, maximum=4, step=1, value=2,
                                    label=t("anime4k_factor"),
                                    info=t("anime4k_factor_info")
                                )
                                
                                with gr.Accordion(t("advanced_options"), open=False):
                                    anime4k_processor = gr.Dropdown(
                                        choices=get_choices("anime4k_processor"),
                                        value="cuda",
                                        label=t("anime4k_processor"),
                                        info=t("anime4k_processor_info")
                                    )
                                    anime4k_device = gr.Number(
                                        value=0,
                                        label=t("anime4k_device"),
                                        info=t("anime4k_device_info"),
                                        precision=0
                                    )
                                    anime4k_format = gr.Radio(
                                        choices=get_choices("format"),
                                        value="png",
                                        label=t("output_format")
                                    )
                                
                                anime4k_btn = gr.Button("🚀 " + t("start_process"), variant="primary", elem_classes=["mobile-friendly-btn"])
                        
                        custom_status = gr.Textbox(label=t("status"), lines=4, interactive=False)
                        
                        # 预设管理
                        with gr.Accordion(t("preset_manage"), open=False):
                            preset_name_input = gr.Textbox(label=t("preset_name"), placeholder=t("preset_name_placeholder"), lines=1)
                            current_engine_state = gr.State(value="cugan")
                            
                            with gr.Row():
                                save_preset_btn = gr.Button(t("save"), variant="primary", scale=1)
                                load_preset_btn = gr.Button(t("load"), variant="secondary", scale=1)
                            
                            saved_presets_dropdown = gr.Dropdown(
                                choices=get_custom_preset_names(),
                                label=t("saved_presets"),
                                interactive=True
                            )
                            
                            with gr.Row():
                                rename_preset_btn = gr.Button(t("rename"), scale=1)
                                delete_preset_btn = gr.Button(t("delete"), variant="stop", scale=1)
                            
                            new_preset_name = gr.Textbox(label=t("new_name"), placeholder=t("new_name_placeholder"), lines=1)
                            preset_manage_status = gr.Textbox(label=t("operation_status"), interactive=False, lines=1)
                    
                    # 右侧：结果展示
                    with gr.Column(scale=2, min_width=400):
                        gr.Markdown(f"### {t('process_result')}")
                        custom_output = gr.Image(
                            label=t("result_preview"),
                            type="filepath",
                            height=350,
                            interactive=False
                        )
                        
                        # 自定义预览按钮（移动端友好）
                        with gr.Row(elem_classes=["preview-btn-row"]):
                            custom_download_btn = gr.Button(t("download"), elem_classes=["mobile-friendly-btn"])
                            custom_fullscreen_btn = gr.Button(t("zoom"), elem_classes=["mobile-friendly-btn"])
                        
                        gr.Markdown(f"### {t('effect_compare')}")
                        custom_compare = gr.ImageSlider(
                            label=t("compare_label"),
                            type="filepath"
                        )
                
                # 统一处理函数 - 自动检测GIF并分派
                def process_smart(input_path: str, engine: str, gif_fmt: str = "webp", **params) -> Tuple[Optional[str], str]:
                    """智能处理函数 - 自动检测GIF并使用相应处理逻辑"""
                    if is_animated_gif(input_path):
                        return process_gif_image(input_path, engine, gif_output_format=gif_fmt, **params)
                    else:
                        return process_image(input_path, engine, **params)
                
                # 各引擎处理函数 - 支持完整参数和GIF
                def process_cugan(img, gif_fmt, model, scale, denoise, syncgap, tile, tta, gpu, threads, fmt):
                    if img is None:
                        return None, None, "[错误] 请先上传图片"
                    model_key = "Pro" if "Pro" in model else "SE"
                    denoise_map = {"无降噪": -1, "保守降噪": 0, "强力降噪": 3}
                    output, msg = process_smart(
                        img, "cugan", gif_fmt,
                        scale=int(scale),
                        denoise=denoise_map.get(denoise, 0),
                        model=model_key,
                        syncgap=int(syncgap),
                        tile_size=int(tile),
                        tta_mode=tta,
                        gpu_id=int(gpu) if gpu is not None else -2,
                        threads=threads,
                        output_format=fmt
                    )
                    if output:
                        return output, (img, output), msg
                    return None, None, msg
                
                def process_esrgan(img, gif_fmt, model_name, scale, tile, tta, gpu, threads, fmt):
                    if img is None:
                        return None, None, "[错误] 请先上传图片"
                    output, msg = process_smart(
                        img, "esrgan", gif_fmt,
                        scale=int(scale),
                        model_name=model_name,
                        tile_size=int(tile),
                        tta_mode=tta,
                        gpu_id=int(gpu) if gpu is not None else -2,
                        threads=threads,
                        output_format=fmt
                    )
                    if output:
                        return output, (img, output), msg
                    return None, None, msg
                
                def process_waifu(img, gif_fmt, model_type, scale, denoise, tile, tta, gpu, threads, fmt):
                    if img is None:
                        return None, None, "[错误] 请先上传图片"
                    output, msg = process_smart(
                        img, "waifu2x", gif_fmt,
                        scale=int(scale),
                        denoise=int(denoise),
                        model_type=model_type,
                        tile_size=int(tile),
                        tta_mode=tta,
                        gpu_id=int(gpu) if gpu is not None else -2,
                        threads=threads,
                        output_format=fmt
                    )
                    if output:
                        return output, (img, output), msg
                    return None, None, msg
                
                def process_anime4k(img, gif_fmt, model, factor, processor, device, fmt):
                    if img is None:
                        return None, None, "[错误] 请先上传图片"
                    output, msg = process_smart(
                        img, "anime4k", gif_fmt,
                        scale=int(factor),
                        model_name=model,
                        processor=processor,
                        device=int(device) if device else 0,
                        output_format=fmt
                    )
                    if output:
                        return output, (img, output), msg
                    return None, None, msg
                
                cugan_btn.click(
                    fn=process_cugan,
                    inputs=[custom_input, gif_output_format, cugan_model, cugan_scale, cugan_denoise,
                            cugan_syncgap, cugan_tile, cugan_tta, cugan_gpu, cugan_threads, cugan_format],
                    outputs=[custom_output, custom_compare, custom_status]
                )
                
                esrgan_btn.click(
                    fn=process_esrgan,
                    inputs=[custom_input, gif_output_format, esrgan_model, esrgan_scale,
                            esrgan_tile, esrgan_tta, esrgan_gpu, esrgan_threads, esrgan_format],
                    outputs=[custom_output, custom_compare, custom_status]
                )
                
                waifu_btn.click(
                    fn=process_waifu,
                    inputs=[custom_input, gif_output_format, waifu_model, waifu_scale, waifu_denoise,
                            waifu_tile, waifu_tta, waifu_gpu, waifu_threads, waifu_format],
                    outputs=[custom_output, custom_compare, custom_status]
                )
                
                anime4k_btn.click(
                    fn=process_anime4k,
                    inputs=[custom_input, gif_output_format, anime4k_model, anime4k_factor, anime4k_processor, anime4k_device, anime4k_format],
                    outputs=[custom_output, custom_compare, custom_status]
                )
                
                # 下载和放大功能
                def open_image_folder(img_path):
                    if img_path:
                        import subprocess
                        subprocess.run(['explorer', '/select,', img_path])
                        return "[提示] 已在文件管理器中打开"
                    return "[提示] 暂无图片"
                
                custom_download_btn.click(
                    fn=open_image_folder,
                    inputs=[custom_output],
                    outputs=[custom_status]
                )
                
                # 预设管理函数
                def save_current_preset_v2(name, c_model, c_scale, c_denoise, 
                                          c_syncgap, c_tile, c_tta, c_gpu, c_threads, c_format,
                                          e_model, e_scale, e_tile, e_tta, e_gpu, e_threads, e_format,
                                          w_model, w_scale, w_denoise, w_tile, w_tta, w_gpu, w_threads, w_format):
                    """保存当前参数为预设 - 保存所有引擎的全部参数"""
                    if not name or not name.strip():
                        return "预设名称不能为空", gr.update()
                    
                    # 统一处理名称
                    name = name.strip()
                    
                    # 保存所有引擎的完整参数
                    all_params = {
                        "cugan": {
                            "model": "Pro" if "Pro" in str(c_model) else "SE",
                            "scale": int(c_scale),
                            "denoise": {"无降噪": -1, "保守降噪": 0, "强力降噪": 3}.get(c_denoise, 0),
                            "syncgap": int(c_syncgap),
                            "tile": int(c_tile),
                            "tta": bool(c_tta),
                            "gpu": int(c_gpu) if c_gpu is not None else -2,
                            "threads": str(c_threads),
                            "format": str(c_format)
                        },
                        "esrgan": {
                            "model": str(e_model),
                            "scale": int(e_scale),
                            "tile": int(e_tile),
                            "tta": bool(e_tta),
                            "gpu": int(e_gpu) if e_gpu is not None else -2,
                            "threads": str(e_threads),
                            "format": str(e_format)
                        },
                        "waifu2x": {
                            "model": str(w_model),
                            "scale": int(w_scale),
                            "denoise": int(w_denoise),
                            "tile": int(w_tile),
                            "tta": bool(w_tta),
                            "gpu": int(w_gpu) if w_gpu is not None else -2,
                            "threads": str(w_threads),
                            "format": str(w_format)
                        }
                    }
                    
                    presets = load_custom_presets()
                    presets[name] = {
                        "all_params": all_params,
                        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    if save_custom_presets(presets):
                        # 返回更新后的下拉框，确保 value 与 choices 中的项匹配
                        new_choices = get_custom_preset_names()
                        return f"[完成] 预设 '{name}' 已保存", gr.update(choices=new_choices, value=name)
                    return "[错误] 保存失败", gr.update()

                
                save_preset_btn.click(
                    fn=save_current_preset_v2,
                    inputs=[preset_name_input, 
                            cugan_model, cugan_scale, cugan_denoise, cugan_syncgap, cugan_tile, cugan_tta, cugan_gpu, cugan_threads, cugan_format,
                            esrgan_model, esrgan_scale, esrgan_tile, esrgan_tta, esrgan_gpu, esrgan_threads, esrgan_format,
                            waifu_model, waifu_scale, waifu_denoise, waifu_tile, waifu_tta, waifu_gpu, waifu_threads, waifu_format],
                    outputs=[preset_manage_status, saved_presets_dropdown]
                )
                
                def load_selected_preset_v2(preset_name):
                    """加载预设 - 恢复所有引擎的全部参数"""
                    # 返回值数量: cugan(6) + esrgan(6) + waifu(6) + status(1) = 19
                    default_return = [gr.update()] * 18 + ["请先选择预设"]
                    
                    if not preset_name:
                        return default_return
                    
                    # 统一处理名称
                    preset_name = preset_name.strip() if isinstance(preset_name, str) else preset_name
                    
                    presets = load_custom_presets()
                    print(f"[调试] 尝试加载预设: '{preset_name}', 可用预设: {list(presets.keys())}")
                    
                    if preset_name not in presets:
                        return [gr.update()] * 18 + [f"预设 '{preset_name}' 不存在，可用: {list(presets.keys())}"]
                    
                    preset = presets[preset_name]
                    
                    if "all_params" in preset:
                        params = preset["all_params"]
                        cugan_p = params.get("cugan", {})
                        esrgan_p = params.get("esrgan", {})
                        waifu_p = params.get("waifu2x", {})
                        
                        denoise_map = {-1: "无降噪", 0: "保守降噪", 3: "强力降噪"}
                        
                        return [
                            # CUGAN 参数 (6个)
                            gr.update(value=f"{cugan_p.get('model', 'Pro')} ({'专业版' if cugan_p.get('model') == 'Pro' else '标准版'})"),
                            gr.update(value=cugan_p.get("scale", 2)),
                            gr.update(value=denoise_map.get(cugan_p.get("denoise", 0), "保守降噪")),
                            gr.update(value=cugan_p.get("syncgap", 3)),
                            gr.update(value=cugan_p.get("tile", 0)),
                            gr.update(value=cugan_p.get("tta", False)),
                            # ESRGAN 参数 (6个)
                            gr.update(value=esrgan_p.get("model", "auto")),
                            gr.update(value=esrgan_p.get("scale", 4)),
                            gr.update(value=esrgan_p.get("tile", 0)),
                            gr.update(value=esrgan_p.get("tta", False)),
                            gr.update(value=esrgan_p.get("gpu", -2)),
                            gr.update(value=esrgan_p.get("threads", "1:2:2")),
                            # WAIFU2X 参数 (6个)
                            gr.update(value=waifu_p.get("model", "cunet")),
                            gr.update(value=waifu_p.get("scale", 2)),
                            gr.update(value=waifu_p.get("denoise", 1)),
                            gr.update(value=waifu_p.get("tile", 0)),
                            gr.update(value=waifu_p.get("tta", False)),
                            gr.update(value=waifu_p.get("gpu", -2)),
                            # 状态
                            f"[加载] {preset_name}"
                        ]
                    else:
                        # 旧格式兼容
                        return [gr.update()] * 18 + [f"[加载] {preset_name} (旧格式)"]
                
                load_preset_btn.click(
                    fn=load_selected_preset_v2,
                    inputs=[saved_presets_dropdown],
                    outputs=[cugan_model, cugan_scale, cugan_denoise, cugan_syncgap, cugan_tile, cugan_tta,
                             esrgan_model, esrgan_scale, esrgan_tile, esrgan_tta, esrgan_gpu, esrgan_threads,
                             waifu_model, waifu_scale, waifu_denoise, waifu_tile, waifu_tta, waifu_gpu,
                             preset_manage_status]
                )
                
                def rename_selected_preset(old_name, new_name):
                    success, msg = rename_custom_preset(old_name, new_name)
                    return msg, gr.update(choices=get_custom_preset_names(), value=new_name if success else old_name)
                
                rename_preset_btn.click(
                    fn=rename_selected_preset,
                    inputs=[saved_presets_dropdown, new_preset_name],
                    outputs=[preset_manage_status, saved_presets_dropdown]
                )
                
                def delete_selected_preset(name):
                    success, msg = delete_custom_preset(name)
                    return msg, gr.update(choices=get_custom_preset_names(), value=None)
                
                delete_preset_btn.click(
                    fn=delete_selected_preset,
                    inputs=[saved_presets_dropdown],
                    outputs=[preset_manage_status, saved_presets_dropdown]
                )
            
            # 图库标签页
            with gr.Tab(t("tab_gallery")):
                gr.Markdown(f"### {t('gallery_title')}")
                gr.Markdown(t("gallery_desc") + f" **({t('click_load_original')})**")
                
                # 初始化时获取缩略图
                initial_thumbs, thumb_map = get_gallery_with_thumbnails()
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # 图库组件 - 使用缩略图显示
                        gallery = gr.Gallery(
                            label=t("image_list"),
                            value=initial_thumbs,
                            columns=5,
                            rows=3,
                            height="auto",
                            object_fit="cover",
                            allow_preview=False,  # 禁用内置预览，使用自定义预览
                            show_label=False,
                            elem_classes=["thumbnail-gallery"]
                        )
                        
                        with gr.Row():
                            refresh_btn = gr.Button(t("refresh_gallery"), variant="secondary")
                            delete_btn = gr.Button(t("delete_selected"), variant="stop")
                        
                        gallery_status = gr.Textbox(
                            label=t("operation_status"),
                            value=t("click_view_detail"),
                            interactive=False,
                            lines=1
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown(f"### {t('image_preview')}")
                        # 点击缩略图后在此加载原图
                        preview_image = gr.Image(
                            label=t("preview") + f" ({t('click_load_original')})",
                            type="filepath",
                            height=350,
                            interactive=False
                        )
                        
                        gr.Markdown(f"### {t('image_info')}")
                        image_info = gr.Textbox(
                            label=t("detail_info"),
                            value=t("select_show"),
                            lines=12,
                            interactive=False
                        )
                
                # 用于存储当前选中的原图路径
                selected_image_path = gr.State(value=None)
                
                def on_gallery_select(evt: gr.SelectData):
                    """图库选择事件 - 从缩略图加载原图"""
                    thumbs, mapping = get_gallery_with_thumbnails()
                    if evt.index < len(thumbs):
                        thumb_path = thumbs[evt.index]
                        # 获取原图路径
                        original_path = get_original_from_thumbnail(thumb_path)
                        # 加载原图信息
                        preview, info = get_image_info(original_path)
                        return original_path, preview, info
                    return None, None, t("select_show")
                
                gallery.select(
                    fn=on_gallery_select,
                    inputs=None,
                    outputs=[selected_image_path, preview_image, image_info]
                )
                
                def refresh_gallery():
                    """刷新图库 - 重新生成缩略图"""
                    cleanup_old_thumbnails()  # 清理旧缩略图
                    thumbs, _ = get_gallery_with_thumbnails()
                    return thumbs, t("msg_refresh_count").format(count=len(thumbs))
                
                refresh_btn.click(
                    fn=refresh_gallery,
                    inputs=None,
                    outputs=[gallery, gallery_status]
                )
                
                def delete_selected(img_path):
                    """删除选中的图片"""
                    if not img_path:
                        thumbs, _ = get_gallery_with_thumbnails()
                        return thumbs, t("msg_select_delete"), None, None, t("select_show")
                    
                    # 删除原图
                    images, msg = delete_gallery_image(img_path)
                    # 刷新缩略图列表
                    thumbs, _ = get_gallery_with_thumbnails()
                    return thumbs, msg, None, None, t("msg_image_deleted")
                
                delete_btn.click(
                    fn=delete_selected,
                    inputs=[selected_image_path],
                    outputs=[gallery, gallery_status, selected_image_path, preview_image, image_info]
                )
            
            # 设置标签页
            with gr.Tab(t("tab_settings")):
                # === 语言设置 ===
                gr.Markdown(f"## {t('language_settings')}")
                gr.Markdown(t("language_desc"))
                
                with gr.Row():
                    lang_selector = gr.Radio(
                        choices=["简体中文", "English"],
                        value="简体中文" if get_current_lang() == "zh-CN" else "English",
                        label=t("language"),
                        interactive=True
                    )
                    lang_save_status = gr.Textbox(
                        label="",
                        value="",
                        interactive=False,
                        lines=1,
                        visible=False
                    )
                
                gr.Markdown("---")
                
                # === 固定预设设置 ===
                gr.Markdown(f"## {t('pinned_presets')}")
                gr.Markdown(t("pinned_presets_desc"))
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # 可固定的预设列表（自定义预设）
                        available_presets = get_custom_preset_names()
                        pinned_list = get_pinned_presets()
                        
                        preset_to_pin = gr.Dropdown(
                            choices=available_presets,
                            value=available_presets[0] if available_presets else None,
                            label=t("saved_presets"),
                            interactive=True
                        )
                        
                        # 刷新预设列表按钮
                        refresh_preset_list_btn = gr.Button("🔄 刷新预设列表", variant="secondary", size="sm")
                        
                        with gr.Row():
                            pin_btn = gr.Button(t("pin_preset"), variant="primary", scale=1)
                            unpin_btn = gr.Button(t("unpin_preset"), variant="secondary", scale=1)
                        
                        pinned_status = gr.Textbox(
                            label=t("operation_status"),
                            value=t("pinned_count").format(count=len(pinned_list)),
                            interactive=False,
                            lines=1
                        )
                    
                    with gr.Column(scale=1):
                        # 显示当前固定的预设
                        pinned_display = gr.Markdown(
                            value=f"**已固定的预设:** {', '.join(pinned_list) if pinned_list else '无'}"
                        )
                
                def refresh_preset_dropdown():
                    """刷新预设下拉列表"""
                    presets = get_custom_preset_names()
                    return gr.update(choices=presets, value=presets[0] if presets else None)
                
                refresh_preset_list_btn.click(
                    fn=refresh_preset_dropdown,
                    inputs=None,
                    outputs=[preset_to_pin]
                )
                
                def do_pin_preset(name):
                    if not name:
                        pinned = get_pinned_presets()
                        return f"请先选择预设 (当前已固定 {len(pinned)} 个)", \
                               f"**已固定的预设:** {', '.join(pinned) if pinned else '无'}"
                    
                    # 去除可能的空格
                    name = name.strip()
                    success, msg = pin_preset(name)
                    pinned = get_pinned_presets()
                    status = f"{msg} (已固定 {len(pinned)} 个)"
                    return status, f"**已固定的预设:** {', '.join(pinned) if pinned else '无'}"
                
                def do_unpin_preset(name):
                    if not name:
                        pinned = get_pinned_presets()
                        return f"请先选择预设 (当前已固定 {len(pinned)} 个)", \
                               f"**已固定的预设:** {', '.join(pinned) if pinned else '无'}"
                    
                    # 去除可能的空格
                    name = name.strip()
                    success, msg = unpin_preset(name)
                    pinned = get_pinned_presets()
                    status = f"{msg} (已固定 {len(pinned)} 个)"
                    return status, f"**已固定的预设:** {', '.join(pinned) if pinned else '无'}"
                
                pin_btn.click(
                    fn=do_pin_preset,
                    inputs=[preset_to_pin],
                    outputs=[pinned_status, pinned_display]
                )
                
                unpin_btn.click(
                    fn=do_unpin_preset,
                    inputs=[preset_to_pin],
                    outputs=[pinned_status, pinned_display]
                )
                
                gr.Markdown("---")
                
                # === 网络分享设置 ===
                gr.Markdown(f"## {t('network_share')}")
                gr.Markdown(t("share_desc"))
                
                # 读取当前配置
                current_share_setting = load_share_config()
                
                with gr.Row():
                    with gr.Column(scale=1):
                        share_enabled = gr.Checkbox(
                            label=t("enable_share"),
                            value=current_share_setting,
                            info=t("enable_share_info")
                        )
                        
                        save_config_btn = gr.Button(t("save_settings"), variant="primary")
                        
                        config_status = gr.Textbox(
                            label=t("config_status"),
                            value=f"{t('current_config')}: " + (t("share_enabled") if current_share_setting else t("local_only")),
                            interactive=False,
                            lines=1
                        )
                        
                        gr.Markdown(f"### {t('access_address')}")
                        
                        local_url_display = gr.Textbox(
                            label=t("local_address"),
                            value="http://127.0.0.1:7860",
                            interactive=False,
                            lines=1
                        )
                        
                        share_url_display = gr.Textbox(
                            label=t("public_link"),
                            value=t("generating") if current_share_setting else t("not_enabled"),
                            interactive=False,
                            lines=1
                        )
                        
                        refresh_share_btn = gr.Button(t("refresh_link"), variant="secondary")
                        
                        gr.Markdown(t("settings_note"))
                    
                    with gr.Column(scale=1):
                        gr.Markdown(f"### {t('engine_status')}")
                        engine_info = gr.Markdown(value=f"```\n{status_text}\n```")
                        
                        gr.Markdown(f"""
                        ### {t('dir_info')}
                        - {t('output_dir')}: `{OUTPUT_DIR}`
                        - {t('program_dir')}: `{BASE_DIR}`
                        - {t('data_dir')}: `{DATA_DIR}`
                        - {t('config_file')}: `{CONFIG_FILE}`
                        - {t('preset_file')}: `{CUSTOM_PRESETS_FILE}`
                        """)
                        
                        gr.Markdown(f"### {t('custom_presets')}")
                        custom_preset_count = gr.Markdown(
                            value=t("saved_presets_count").format(count=len(get_custom_preset_names()))
                        )
                
                def save_share_setting(enabled: bool) -> str:
                    """保存分享设置"""
                    save_share_config(enabled)
                    status = t("share_enabled") if enabled else t("local_only")
                    return t("msg_settings_saved").format(status=status)
                
                save_config_btn.click(
                    fn=save_share_setting,
                    inputs=[share_enabled],
                    outputs=[config_status]
                )
                
                refresh_share_btn.click(
                    fn=refresh_share_url,
                    inputs=None,
                    outputs=[share_url_display]
                )
            
            # 帮助标签页
            with gr.Tab(t("tab_help")):
                with gr.Tabs():
                    with gr.Tab(t("help_engines")):
                        gr.Markdown("""
## Real-CUGAN (Real Cascade U-Net GAN)

### 简介
Real-CUGAN 是由 BiliBili 开发的动漫图像超分辨率模型, 专为二次元图片设计。基于 Cascade U-Net 结构, 
结合了 GAN (生成对抗网络) 技术, 能够在放大图像的同时保持甚至增强画面细节。

### 模型版本
| 模型 | 特点 | 适用场景 |
|------|------|----------|
| **SE (Standard Edition)** | 标准版, 效果均衡, 处理速度快 | 一般动漫图片、插画 |
| **Pro (Professional)** | 专业版, 细节还原更好, 边缘更锐利 | 高质量原图、需要保留细节的场景 |

### 参数说明
- **放大倍率**: 2x / 3x / 4x, 倍率越高处理时间越长
- **降噪等级**: 
  - `-1` (无降噪): 完全保留原图噪点, 适合高质量原图
  - `0` (保守降噪): 轻微降噪, 保留大部分细节
  - `3` (强力降噪): 强力去噪, 适合有明显噪点的图片

### 优点
- 针对动漫图片优化, 线条清晰锐利
- 色块边缘处理干净, 不会出现模糊
- 支持多种降噪等级, 可根据需求调节
- 处理速度较快

### 缺点
- 对真实照片效果一般
- 过度放大可能产生伪影
- 某些复杂纹理可能丢失细节

---

## Real-ESRGAN (Enhanced Super-Resolution GAN)

### 简介
Real-ESRGAN 是目前最强大的通用图像超分辨率模型之一, 由腾讯 ARC 实验室开发。
它采用了改进的 ESRGAN 架构, 能够处理各种类型的低质量图片, 包括模糊、噪点、压缩伪影等问题。

### 模型版本
| 模型 | 放大倍率 | 特点 |
|------|----------|------|
| **realesr-animevideov3** | 2x / 3x | 针对动漫视频优化, 时序稳定性好 |
| **realesrgan-x4plus-anime** | 4x | 动漫图片专用, 效果最佳 |
| **realesrgan-x4plus** | 4x | 通用模型, 适合真实照片 |

### 优点
- 修复能力极强, 能处理严重退化的图片
- 对压缩伪影 (如JPEG马赛克) 有很好的修复效果
- 既支持动漫也支持真实照片
- 输出质量稳定

### 缺点
- 处理速度相对较慢
- 可能过度平滑某些细节
- 对于高质量原图可能"过度处理"
- 显存占用较大

### 最佳实践
- 模糊/压缩严重的图片: 使用 4x 模型
- 动漫视频截图: 使用 animevideov3
- 真实照片: 使用 x4plus (非anime版本)

---

## Waifu2x

### 简介
Waifu2x 是最早的 AI 图像超分辨率工具之一, 最初由 nagadomi 开发。
虽然技术相对较老, 但其降噪效果依然非常出色, 特别适合需要柔和画面效果的场景。

### 模型版本
| 模型 | 特点 | 适用场景 |
|------|------|----------|
| **cunet** | 最新模型, 效果最好 | 默认推荐 |
| **upconv_7_anime_style_art_rgb** | 动漫风格优化 | 纯二次元图片 |
| **upconv_7_photo** | 照片优化 | 真实照片 |

### 参数说明
- **放大倍率**: 1x (仅降噪) / 2x / 4x
- **降噪等级**: 0-3, 数值越大降噪越强
  - `0`: 无降噪
  - `1`: 轻度降噪
  - `2`: 中度降噪  
  - `3`: 强力降噪 (画面会变得非常柔和)

### 优点
- 降噪效果极佳, 画面柔和细腻
- 处理速度快
- 显存占用小
- 可单独进行降噪 (1x模式)

### 缺点
- 放大效果不如新一代模型
- 可能导致画面过于模糊
- 细节保留能力较弱
- 锐度不足

### 最佳实践
- 需要强力降噪时选择 Waifu2x
- 配合其他工具使用: 先用 Waifu2x 降噪, 再用 CUGAN 放大

---

## Anime4KCPP (Anime4K CNN++)

### 简介
Anime4KCPP 是基于 Anime4K 算法的 C++ 高性能实现, 由 TianZerL 开发。
V3 版本使用纯 CNN (卷积神经网络) 算法, 专为动漫图像和视频设计, 
支持 CPU、OpenCL 和 CUDA 三种加速方式。

### 模型版本
| 模型 | 特点 | 适用场景 |
|------|------|----------|
| **ACNet-GAN** | GAN增强版本, 质量更好 | 追求最佳画质时使用 |
| **ACNet** | 标准CNN模型, 速度更快 | 批量处理、实时预览 |

### 参数说明
- **放大倍率**: 2x / 3x / 4x
- **处理器类型**:
  - `OpenCL`: 兼容性最好, 支持大多数显卡
  - `CPU`: 无需显卡, 但速度较慢
  - `CUDA`: NVIDIA显卡专用, 速度最快
- **设备索引**: 多显卡时选择具体使用哪张显卡

### 优点
- **处理速度极快**: 针对实时处理优化, 适合视频和动图
- **显存占用低**: 可处理大尺寸图片
- **多平台加速**: 支持OpenCL/CUDA/CPU
- **动图友好**: 处理GIF等动态图像效果好

### 缺点
- 相比其他模型细节还原略弱
- 对复杂纹理处理一般
- 主要针对动漫风格优化

### 最佳实践
- GIF动图处理首选 Anime4K
- 需要快速预览效果时使用
- 视频帧处理推荐使用

### 与其他引擎对比
| 特性 | Real-CUGAN | Real-ESRGAN | Waifu2x | Anime4K |
|------|------------|-------------|---------|---------|
| 处理速度 | 中 | 慢 | 快 | 极快 |
| 显存占用 | 中 | 高 | 低 | 极低 |
| 细节还原 | 优 | 优 | 良 | 良 |
| 动图支持 | 一般 | 一般 | 一般 | 优秀 |
| 降噪能力 | 有 | 有 | 强 | 无 |
                        """)
                    
                    with gr.Tab("预设说明"):
                        gr.Markdown("""
## 内置预设详解

### 通用增强
| 项目 | 设置 |
|------|------|
| 引擎 | Real-CUGAN Pro |
| 放大倍率 | 2x |
| 降噪等级 | 保守降噪 (0) |

**适用场景**:
- 一般动漫图片的放大
- 社交媒体头像制作
- 普通质量图片的增强

**效果特点**:
- 画面清晰度提升明显
- 保留原图大部分细节
- 轻微降噪, 画面更干净

---

### 烂图修复
| 项目 | 设置 |
|------|------|
| 引擎 | Real-ESRGAN x4plus-anime |
| 放大倍率 | 4x |
| 降噪等级 | 自动 |

**适用场景**:
- 严重压缩的图片 (如微信传输后的图)
- 模糊不清的老图
- 有明显马赛克/块状伪影的图片
- 小尺寸缩略图放大

**效果特点**:
- 强力修复各种画质问题
- 大幅提升分辨率
- 可能会"脑补"一些细节

---

### 壁纸制作
| 项目 | 设置 |
|------|------|
| 引擎 | Real-CUGAN SE |
| 放大倍率 | 4x |
| 降噪等级 | 无降噪 (-1) |

**适用场景**:
- 制作桌面壁纸 (1080p -> 4K)
- 高质量原图的放大
- 需要保留所有原始细节的场景

**效果特点**:
- 最大程度保留原图细节和噪点
- 适合高质量原图
- 输出分辨率最高

---

### 极致柔化
| 项目 | 设置 |
|------|------|
| 引擎 | Waifu2x |
| 放大倍率 | 2x |
| 降噪等级 | 强力降噪 (3) |

**适用场景**:
- 需要柔和画面效果
- 去除图片噪点/颗粒感
- 皮肤质感优化
- 动漫截图美化

**效果特点**:
- 画面非常柔和细腻
- 噪点完全去除
- 可能损失部分细节
- 类似"磨皮"效果

---

### 快速超分
| 项目 | 设置 |
|------|------|
| 引擎 | Anime4K |
| 放大倍率 | 2x |
| 模型 | ACNet-GAN |
| 处理器 | CUDA |

**适用场景**:
- GIF 动图超分辨率
- 视频帧批量处理
- 需要快速预览效果
- 动画帧较多的文件

**效果特点**:
- 处理速度极快
- 显存占用极低
- 动画帧处理稳定
- 适合实时预览

---

## 如何选择?

```
图片质量如何?
    |
    +-- 很差 (模糊/压缩) --> 烂图修复
    |
    +-- 一般 --> 是动图/视频帧吗?
                    |
                    +-- 是 --> 快速超分
                    |
                    +-- 否 --> 通用增强
    |
    +-- 很好 --> 想要什么效果?
                    |
                    +-- 保留细节 --> 壁纸制作
                    |
                    +-- 柔和画面 --> 极致柔化
```

## 自定义预设

在"自定义模式"中, 你可以:
1. 自由调节所有参数
2. 将当前配置保存为自定义预设
3. 随时加载、重命名或删除预设
4. 预设会保存在 `custom_presets.json` 文件中
                        """)
                    
                    with gr.Tab("使用技巧"):
                        gr.Markdown("""
## 使用技巧

### 1. 分辨率与倍率选择
- 目标分辨率 = 原图分辨率 x 放大倍率
- 例: 500x500 图片 4x 放大 = 2000x2000
- 建议: 根据目标用途选择合适倍率, 不要过度放大

### 2. 多次处理
对于特别差的图片, 可以尝试多次处理:
1. 第一次: Waifu2x 降噪
2. 第二次: CUGAN 放大
3. 效果往往优于单次大倍率放大

### 3. 显卡选择
- 程序使用 Vulkan API, 支持 NVIDIA/AMD/Intel 显卡
- 建议显存 >= 4GB
- 处理大图时, 程序会自动分块处理

### 4. 批量处理
- 目前需要逐张处理
- 输出图片自动保存在"输出"文件夹
- 可在"图库"中统一管理

### 5. 元数据
- 所有处理后的图片都会嵌入 AIS 元数据
- 记录处理时间、使用的引擎和参数
- 方便追溯和复现效果

---

## 常见问题

**Q: 处理很慢怎么办?**
A: 降低放大倍率, 或使用更强的显卡

**Q: 显存不足怎么办?**
A: 程序会自动分块处理, 但速度会变慢

**Q: 效果不好怎么办?**
A: 尝试不同的引擎和参数组合

**Q: 公开链接无法访问?**
A: Gradio 隧道可能被防火墙阻止, 尝试使用本地地址

---

## 快捷键
- Ctrl+V: 粘贴剪贴板图片
- 拖拽: 直接拖拽图片到上传区域
                        """)
                    
                    with gr.Tab("关于"):
                        gr.Markdown("""
## 关于本项目

**AIS (AIS - AI Image Super-resolution)**

By SENyiAi | [GitHub](https://github.com/SENyiAi/AIS)

### 使用的开源项目
- [Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) - BiliBili
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Tencent ARC Lab
- [Waifu2x](https://github.com/nagadomi/waifu2x) - nagadomi
- [Gradio](https://gradio.app/) - Hugging Face

### 注意事项
- 本程序完全免费且开源
- 仅供学习交流使用
- 请勿用于商业用途
- 请尊重原图版权

### 系统要求
- Windows 10/11 64位
- 支持 Vulkan 的显卡 (NVIDIA/AMD/Intel)
- 建议显存 >= 4GB
- 建议内存 >= 8GB
                        """)
        
        gr.Markdown("""
        ---
        **提示**: 处理完成后可使用对比滑块查看效果差异 | 输出文件保存在 `输出` 文件夹 | 在图库中可浏览所有输出
        """)
        
        # 语言切换 - 使用JS强制刷新页面 (在设置页定义的 lang_selector)
        def on_language_change(lang_choice):
            """处理语言切换"""
            new_lang = "zh-CN" if lang_choice == "简体中文" else "en"
            set_lang(new_lang)
            return "[已保存] 刷新页面后生效"
        
        lang_selector.change(
            fn=on_language_change,
            inputs=[lang_selector],
            outputs=[lang_save_status],
            js="() => { setTimeout(() => { window.location.reload(); }, 500); }"
        )
        
        # 页面加载时自动刷新公开链接
        if current_share_setting:
            app.load(
                fn=refresh_share_url,
                inputs=None,
                outputs=[share_url_display]
            )
    
    return app


def print_startup_info(engine_status: Dict[str, bool], share_enabled: bool) -> None:
    """打印启动信息"""
    print("\n" + "=" * 60)
    print("         AIS - Web UI")
    print("                By SENyiAi")
    print("=" * 60)
    print("\n[检测] 引擎状态:")
    for name, available in engine_status.items():
        status = "[OK]" if available else "[X]"
        print(f"  {status} {name.upper()}")
    print("\n" + "=" * 60)
    print("本地访问: http://127.0.0.1:7860")
    if share_enabled:
        print("公开链接: 启动后将显示 (可能需要等待几秒)")
    else:
        print("公开链接: 未启用 (可在设置中开启)")
    print("按 Ctrl+C 停止服务")
    print("=" * 60 + "\n")


def save_share_url_to_file(url: str) -> None:
    """将公开链接保存到文件"""
    global SHARE_URL
    if not url or not url.startswith("https://"):
        return
    
    SHARE_URL = url
    share_url_file = BASE_DIR / "share_url.txt"
    try:
        share_url_file.write_text(url, encoding='utf-8')
    except (IOError, OSError) as e:
        log_info(f"保存公开链接失败: {e}")


def load_share_url_from_file() -> Optional[str]:
    """从文件加载公开链接"""
    share_url_file = BASE_DIR / "share_url.txt"
    if share_url_file.exists():
        try:
            url = share_url_file.read_text(encoding='utf-8').strip()
            # 验证 URL 格式
            if url.startswith("https://") and ".gradio.live" in url:
                return url
        except (IOError, OSError, UnicodeDecodeError):
            pass
    return None


if __name__ == "__main__":
    # 检查引擎状态
    engine_status = check_engines()
    
    # 加载分享配置
    share_enabled = load_share_config()
    
    # 打印启动信息
    print_startup_info(engine_status, share_enabled)
    log_info("程序启动")
    log_info(f"分享模式: {'启用' if share_enabled else '禁用'}")
    
    # 创建应用
    app = create_ui()
    
    # 启动应用
    try:
        log_info("正在启动Gradio服务...")
        # 使用 prevent_thread_lock=True 以便获取返回值
        # 使用 head 参数注入剪贴板粘贴 JavaScript (Gradio 6.0 正确方式)
        result = app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=share_enabled,
            inbrowser=False,
            prevent_thread_lock=True,
            quiet=True,
            head=CLIPBOARD_PASTE_JS
        )
        
        # 如果启用了share, 尝试获取并保存公开链接
        if share_enabled and result is not None:
            # result 可能是元组 (app, local_url, share_url) 或直接是 app
            share_url = None
            if isinstance(result, tuple) and len(result) >= 3:
                share_url = result[2]
            elif hasattr(app, 'share_url'):
                share_url = app.share_url
            
            if share_url:
                save_share_url_to_file(share_url)
                log_info(f"公开链接: {share_url}")
                print(f"\n[公开链接] {share_url}\n")
        
        # 保持运行直到用户中断
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[停止] 服务已关闭")
    except Exception as e:
        print(f"\n[错误] 启动失败: {e}")