import sys
import subprocess
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime

# 基础路径配置
BASE_DIR = Path(__file__).parent.absolute()
PREREQ_DIR = BASE_DIR / "前置"
MODEL_DIR = BASE_DIR / "模型"
OUTPUT_DIR = BASE_DIR / "输出"
OUTPUT_DIR.mkdir(exist_ok=True)

# 设置本地Python库路径
LOCAL_PYTHON_DIR = PREREQ_DIR / "python-3.14.0-embed-amd64"
if LOCAL_PYTHON_DIR.exists():
    LOCAL_LIB_PATH = LOCAL_PYTHON_DIR / "Lib" / "site-packages"
    if LOCAL_LIB_PATH.exists() and str(LOCAL_LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LOCAL_LIB_PATH))

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
        except:
            pass
    
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
    }
}

# 预设配置
# 预设配置 - 使用 key 而非直接文本，便于 i18n
PRESET_KEYS = ["preset_universal", "preset_repair", "preset_wallpaper", "preset_soft"]

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
    try:
        img = Image.open(image_path)
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
        pnginfo.add_text("AIS_Engine", metadata.get("engine", "unknown"))
        pnginfo.add_text("AIS_Scale", str(metadata.get("scale", 0)))
        if "denoise" in metadata:
            pnginfo.add_text("AIS_Denoise", str(metadata.get("denoise", 0)))
        if "model" in metadata:
            pnginfo.add_text("AIS_Model", metadata.get("model", ""))
        
        # 保存带元数据的图片
        img.save(image_path, pnginfo=pnginfo)
        return True
    except Exception as e:
        print(f"[警告] 写入元数据失败: {e}")
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
    return {
        name: (config["dir"] / config["exe"]).exists()
        for name, config in ENGINES.items()
    }


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


def process_image(input_path: str, engine: str, **params) -> Tuple[Optional[str], str]:
    """处理单张图片
    
    通用参数:
        tile_size: Tile大小 (0=自动)
        gpu_id: GPU设备 (-2=自动, -1=CPU, 0/1/2...=指定GPU)
        threads: 线程数 (load:proc:save)
        tta_mode: TTA模式
        output_format: 输出格式 (png/jpg/webp)
    """
    if not input_path:
        return None, "[错误] 请先上传图片"
    
    input_file = Path(input_path)
    if not input_file.exists():
        return None, "[错误] 输入文件不存在"
    
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

def process_with_preset(input_image: Optional[str], 
                        preset_name: str) -> Tuple[Optional[str], Optional[Tuple], Optional[str], Optional[str], str]:
    """使用预设处理图片
    返回: (处理结果, 对比元组, 原图路径, 结果路径, 状态消息)
    """
    if input_image is None:
        return None, None, None, None, "[错误] 请先上传图片"
    
    preset = PRESETS.get(preset_name)
    if not preset:
        return None, None, None, None, "[错误] 未知预设"
    
    output_path, result_msg = process_image(
        input_image,
        preset["engine"],
        **preset["params"]
    )
    
    if output_path:
        return output_path, (input_image, output_path), input_image, output_path, result_msg
    return None, None, None, None, result_msg


def process_all_presets(input_image: Optional[str]) -> Tuple[List[Optional[str]], str]:
    """执行所有预设并返回结果"""
    if input_image is None:
        return [None] * 4, "[错误] 请先上传图片"
    
    results: List[Optional[str]] = []
    messages: List[str] = []
    
    for preset_name, preset in PRESETS.items():
        messages.append(f"正在执行: {preset_name}")
        output_path, msg = process_image(
            input_image,
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


def process_custom(input_image: Optional[str], engine: str, 
                   cugan_model: str, cugan_scale: int, cugan_denoise: str,
                   esrgan_scale: int,
                   waifu_scale: int, waifu_denoise: int) -> Tuple[Optional[str], Optional[Tuple], Optional[str], Optional[str], str]:
    """自定义模式处理
    返回: (处理结果, 对比元组, 原图路径, 结果路径, 状态消息)
    """
    if input_image is None:
        return None, None, None, None, "[错误] 请先上传图片"
    
    output: Optional[str] = None
    msg: str = ""
    
    if engine == "Real-CUGAN":
        model = "Pro" if "Pro" in cugan_model else "SE"
        denoise_map = {"无降噪": -1, "保守降噪": 0, "强力降噪": 3}
        denoise = denoise_map.get(cugan_denoise, 0)
        output, msg = process_image(
            input_image, "cugan", 
            scale=int(cugan_scale), denoise=denoise, model=model
        )
    
    elif engine == "Real-ESRGAN":
        output, msg = process_image(
            input_image, "esrgan", 
            scale=int(esrgan_scale)
        )
    
    elif engine == "Waifu2x":
        output, msg = process_image(
            input_image, "waifu2x",
            scale=int(waifu_scale), denoise=int(waifu_denoise)
        )
    else:
        return None, None, None, None, "[错误] 请选择引擎"
    
    if output:
        return output, (input_image, output), input_image, output, msg
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
# 图库功能
# ============================================================

def get_gallery_images() -> List[str]:
    """获取输出目录中的所有图片路径"""
    if not OUTPUT_DIR.exists():
        return []
    
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    images = []
    
    for f in OUTPUT_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in extensions:
            images.append(str(f))
    
    # 按修改时间倒序排列(最新的在前)
    images.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return images


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
    """删除图库中的图片"""
    if not image_path:
        return get_gallery_images(), "请先选择要删除的图片"
    
    path = Path(image_path)
    if path.exists():
        try:
            path.unlink()
            return get_gallery_images(), f"[完成] 已删除: {path.name}"
        except Exception as e:
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
            content = LOG_FILE.read_text(encoding='utf-8')
            import re
            match = re.search(r'https://[a-zA-Z0-9-]+\.gradio\.live', content)
            if match:
                url = match.group(0)
                SHARE_URL = url
                return url
        except:
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
    status_list = []
    for name, available in engine_status.items():
        icon = "[OK]" if available else "[X]"
        status_list.append(f"{icon} {name.upper()}")
    return " | ".join(status_list)


def create_ui() -> gr.Blocks:
    """创建 Gradio 界面"""
    
    # 加载语言配置
    load_lang_config()
    
    engine_status = check_engines()
    status_text = get_engine_status_text()
    current_presets = get_presets()
    
    with gr.Blocks(title="AIS") as app:
        # 使用HTML隐藏底栏 + 自定义样式
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
        </style>
        """)
        
        # 标题栏 + 语言切换
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown(f"""
                # {t("app_title")}
                **[GitHub](https://github.com/SENyiAi/AIS)** | {t("app_subtitle")}: {status_text}
                """)
            with gr.Column(scale=1, min_width=150):
                lang_selector = gr.Radio(
                    choices=["简体中文", "English"],
                    value="简体中文" if get_current_lang() == "zh-CN" else "English",
                    label=t("language"),
                    interactive=True
                )
        
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
                        
                        quick_preset = gr.Radio(
                            choices=list(current_presets.keys()),
                            value=list(current_presets.keys())[0],
                            label=t("select_preset")
                        )
                        
                        quick_preset_desc = gr.Markdown(
                            value=f"[{t('msg_info').strip('[]')}] {list(current_presets.values())[0]['desc']}"
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
                
                def update_preset_desc(preset_name: str) -> str:
                    presets = get_presets()
                    preset = presets.get(preset_name, PRESETS.get(preset_name, {}))
                    return f"[{t('msg_info').strip('[]')}] {preset.get('desc', '')}"
                
                quick_preset.change(
                    fn=update_preset_desc,
                    inputs=[quick_preset],
                    outputs=[quick_preset_desc]
                )
                
                quick_btn.click(
                    fn=process_with_preset,
                    inputs=[quick_input, quick_preset],
                    outputs=[quick_output, quick_compare, zoom_original, zoom_result, quick_status]
                )
                
                def run_all_and_update(input_image):
                    """执行所有预设并返回结果"""
                    results, msg = process_all_presets(input_image)
                    # 构建结果字典用于对比选择
                    results_dict = {
                        "原图": input_image,
                        "通用增强": results[0],
                        "烂图修复": results[1],
                        "壁纸制作": results[2],
                        "极致柔化": results[3]
                    }
                    # 默认对比: 原图 vs 通用增强
                    default_compare = (input_image, results[0]) if results[0] else None
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
                
                # 各引擎处理函数 - 支持完整参数
                def process_cugan(img, model, scale, denoise, syncgap, tile, tta, gpu, threads, fmt):
                    if img is None:
                        return None, None, "[错误] 请先上传图片"
                    model_key = "Pro" if "Pro" in model else "SE"
                    denoise_map = {"无降噪": -1, "保守降噪": 0, "强力降噪": 3}
                    output, msg = process_image(
                        img, "cugan",
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
                
                def process_esrgan(img, model_name, scale, tile, tta, gpu, threads, fmt):
                    if img is None:
                        return None, None, "[错误] 请先上传图片"
                    output, msg = process_image(
                        img, "esrgan",
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
                
                def process_waifu(img, model_type, scale, denoise, tile, tta, gpu, threads, fmt):
                    if img is None:
                        return None, None, "[错误] 请先上传图片"
                    output, msg = process_image(
                        img, "waifu2x",
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
                
                cugan_btn.click(
                    fn=process_cugan,
                    inputs=[custom_input, cugan_model, cugan_scale, cugan_denoise,
                            cugan_syncgap, cugan_tile, cugan_tta, cugan_gpu, cugan_threads, cugan_format],
                    outputs=[custom_output, custom_compare, custom_status]
                )
                
                esrgan_btn.click(
                    fn=process_esrgan,
                    inputs=[custom_input, esrgan_model, esrgan_scale,
                            esrgan_tile, esrgan_tta, esrgan_gpu, esrgan_threads, esrgan_format],
                    outputs=[custom_output, custom_compare, custom_status]
                )
                
                waifu_btn.click(
                    fn=process_waifu,
                    inputs=[custom_input, waifu_model, waifu_scale, waifu_denoise,
                            waifu_tile, waifu_tta, waifu_gpu, waifu_threads, waifu_format],
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
                    presets[name.strip()] = {
                        "all_params": all_params,
                        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    if save_custom_presets(presets):
                        return f"[完成] 预设 '{name}' 已保存", gr.update(choices=get_custom_preset_names(), value=name)
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
                    
                    presets = load_custom_presets()
                    if preset_name not in presets:
                        return [gr.update()] * 18 + ["预设不存在"]
                    
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
                gr.Markdown(t("gallery_desc"))
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # 图库组件 - 使用官方推荐配置
                        gallery = gr.Gallery(
                            label=t("image_list"),
                            value=get_gallery_images(),
                            columns=5,
                            rows=3,
                            height="auto",
                            object_fit="cover",
                            allow_preview=True,
                            show_label=False
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
                        preview_image = gr.Image(
                            label=t("preview"),
                            type="filepath",
                            height=300,
                            interactive=False
                        )
                        
                        gr.Markdown(f"### {t('image_info')}")
                        image_info = gr.Textbox(
                            label=t("detail_info"),
                            value=t("select_show"),
                            lines=12,
                            interactive=False
                        )
                
                # 用于存储当前选中的图片路径
                selected_image_path = gr.State(value=None)
                
                def on_gallery_select(evt: gr.SelectData):
                    """图库选择事件"""
                    images = get_gallery_images()
                    if evt.index < len(images):
                        img_path = images[evt.index]
                        preview, info = get_image_info(img_path)
                        return img_path, preview, info
                    return None, None, t("select_show")
                
                gallery.select(
                    fn=on_gallery_select,
                    inputs=None,
                    outputs=[selected_image_path, preview_image, image_info]
                )
                
                def refresh_gallery():
                    """刷新图库"""
                    images = get_gallery_images()
                    return images, t("msg_refresh_count").format(count=len(images))
                
                refresh_btn.click(
                    fn=refresh_gallery,
                    inputs=None,
                    outputs=[gallery, gallery_status]
                )
                
                def delete_selected(img_path):
                    """删除选中的图片"""
                    if not img_path:
                        images = get_gallery_images()
                        return images, t("msg_select_delete"), None, None, t("select_show")
                    
                    images, msg = delete_gallery_image(img_path)
                    return images, msg, None, None, t("msg_image_deleted")
                
                delete_btn.click(
                    fn=delete_selected,
                    inputs=[selected_image_path],
                    outputs=[gallery, gallery_status, selected_image_path, preview_image, image_info]
                )
            
            # 设置标签页
            with gr.Tab(t("tab_settings")):
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

## 如何选择?

```
图片质量如何?
    |
    +-- 很差 (模糊/压缩) --> 烂图修复
    |
    +-- 一般 --> 通用增强
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
        
        # 语言切换 - 使用JS强制刷新页面
        def on_language_change(lang_choice):
            """处理语言切换"""
            new_lang = "zh-CN" if lang_choice == "简体中文" else "en"
            set_lang(new_lang)
            return None
        
        lang_selector.change(
            fn=on_language_change,
            inputs=[lang_selector],
            outputs=None,
            js="() => { setTimeout(() => { window.location.reload(); }, 300); }"
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
    SHARE_URL = url
    share_url_file = BASE_DIR / "share_url.txt"
    try:
        share_url_file.write_text(url, encoding='utf-8')
    except Exception:
        pass


def load_share_url_from_file() -> Optional[str]:
    """从文件加载公开链接"""
    share_url_file = BASE_DIR / "share_url.txt"
    if share_url_file.exists():
        try:
            url = share_url_file.read_text(encoding='utf-8').strip()
            if url.startswith("http"):
                return url
        except Exception:
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
        # launch返回 (app, local_url, share_url)
        log_info("正在启动Gradio服务...")
        result = app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=share_enabled,
            inbrowser=False,
            prevent_thread_lock=False,
            quiet=True  # 禁用Gradio广告信息
        )
        
        # 如果启用了share, 尝试获取并保存公开链接
        # result是元组: (app, local_url, share_url)
        if share_enabled and isinstance(result, tuple) and len(result) >= 3:
            share_url = result[2]
            if share_url:
                save_share_url_to_file(share_url)
                log_info(f"公开链接: {share_url}")
                print(f"\n[公开链接] {share_url}\n")
            
    except KeyboardInterrupt:
        print("\n[停止] 服务已关闭")
    except Exception as e:
        print(f"\n[错误] 启动失败: {e}")