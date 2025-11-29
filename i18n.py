from typing import Dict, Any
from pathlib import Path
import json

# 配置文件路径
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "数据"
DATA_DIR.mkdir(exist_ok=True)
CONFIG_FILE = DATA_DIR / "config.json"

# 当前语言 (默认简体中文)
_current_lang = "zh-CN"


def get_current_lang() -> str:
    """获取当前语言"""
    return _current_lang


def set_lang(lang: str) -> None:
    """设置当前语言"""
    global _current_lang
    if lang in LANGUAGES:
        _current_lang = lang
        save_lang_config(lang)


def _load_config() -> Dict[str, Any]:
    """加载配置文件"""
    if CONFIG_FILE.exists():
        try:
            content = CONFIG_FILE.read_text(encoding='utf-8')
            return json.loads(content)
        except (IOError, OSError, json.JSONDecodeError):
            pass
    return {}


def _save_config(config: Dict[str, Any]) -> bool:
    """保存配置文件"""
    try:
        content = json.dumps(config, ensure_ascii=False, indent=2)
        CONFIG_FILE.write_text(content, encoding='utf-8')
        return True
    except (IOError, OSError, TypeError):
        return False


def load_lang_config() -> str:
    """从配置文件加载语言设置"""
    global _current_lang
    config = _load_config()
    lang = config.get("language", "zh-CN")
    if lang in LANGUAGES:
        _current_lang = lang
        return lang
    return "zh-CN"


def save_lang_config(lang: str) -> None:
    """保存语言设置到配置文件"""
    config = _load_config()
    config["language"] = lang
    _save_config(config)


def t(key: str) -> str:
    """翻译函数 - 根据当前语言返回对应文本"""
    lang_dict = LANGUAGES.get(_current_lang, LANGUAGES["zh-CN"])
    return lang_dict.get(key, key)


def get_choices(key: str) -> list:
    """获取选项列表的翻译"""
    lang_dict = LANGUAGES.get(_current_lang, LANGUAGES["zh-CN"])
    choices_dict = lang_dict.get("_choices", {})
    return choices_dict.get(key, [])


# ============================================================
# 语言字典
# ============================================================

LANGUAGES: Dict[str, Dict[str, Any]] = {
    # ========== 简体中文 ==========
    "zh-CN": {
        # 通用
        "app_title": "AIS",
        "app_subtitle": "引擎状态",
        "language": "语言",
        "lang_zh": "简体中文",
        "lang_en": "English",
        
        # 标签页
        "tab_quick": "快速处理",
        "tab_custom": "自定义模式",
        "tab_gallery": "图库",
        "tab_settings": "设置",
        "tab_help": "帮助",
        
        # 快速处理
        "quick_desc": "上传图片, 选择预设, 一键处理。处理完成后可使用滑块对比原图与结果。",
        "upload_image": "上传图片",
        "select_preset": "选择预设",
        "start_process": "开始处理",
        "process_all": "我全都要",
        "status": "状态",
        "compare_title": "效果对比 (拖动中间分割线)",
        "compare_label": "<- 原图 | 结果 ->",
        "click_zoom": "点击放大对比",
        "click_zoom_tip": "点击图片可放大查看细节",
        "original": "原图",
        "result": "处理结果",
        "view_result": "单独查看处理结果",
        "result_download": "处理结果(点击可下载)",
        
        # 预设
        "preset_universal": "通用增强",
        "preset_repair": "烂图修复",
        "preset_wallpaper": "壁纸制作",
        "preset_soft": "极致柔化",
        "preset_anime4k": "快速超分",
        "preset_universal_desc": "Real-CUGAN Pro 2x 保守降噪, 适合大多数场景",
        "preset_repair_desc": "Real-ESRGAN 4x, 强力修复低质量图片",
        "preset_wallpaper_desc": "Real-CUGAN SE 4x 无降噪, 保留细节制作高清壁纸",
        "preset_soft_desc": "Waifu2x 2x 强力降噪, 画面柔和细腻",
        "preset_anime4k_desc": "Anime4K 2x 快速处理, 适合动图与视频这类动画帧较多的文件",
        
        # 对比
        "all_preset_compare": "全部预设结果对比",
        "all_preset_desc": "一次运行全部4种预设, 方便对比选择最佳效果。可自由选择左右对比图源。",
        "free_compare": "自由对比",
        "left_source": "左侧图源",
        "right_source": "右侧图源",
        
        # 自定义模式
        "custom_desc": "选择引擎并调节参数，满足专业需求。点击「高级选项」展开更多参数。",
        "model_version": "模型版本",
        "model_version_info": "Pro效果更好，SE速度更快",
        "scale_ratio": "放大倍率",
        "denoise_level": "降噪强度",
        "denoise_level_info": "-1=无降噪, 0=保守, 3=强力",
        "advanced_options": "高级选项",
        "advanced_params": "输出设置",
        "syncgap_mode": "同步模式 (SyncGap)",
        "syncgap_info": "0=无同步, 1=精确, 2=粗略, 3=非常粗略(默认)",
        "tile_size": "Tile 大小",
        "tile_info": "0=自动, 值越小显存占用越低",
        "tta_mode": "TTA 模式",
        "tta_info": "8倍时间换取更好效果",
        "gpu_select": "GPU 选择",
        "threads": "线程数 (load:proc:save)",
        "threads_info": "多小图可用4:4:4，大图用2:2:2",
        "output_format": "输出格式",
        "gif_output_format": "动图输出格式",
        "gif_output_format_info": "WebP支持24-bit颜色无色带，GIF仅256色但兼容性更好",
        
        # 模型选项
        "model_se": "SE (标准版)",
        "model_pro": "Pro (专业版)",
        "no_denoise": "无降噪",
        "conservative_denoise": "保守降噪",
        "strong_denoise": "强力降噪",
        "auto_select": "自动选择",
        "auto": "自动",
        "cpu": "CPU",
        
        # ESRGAN 模型
        "esrgan_model_select": "模型选择",
        "esrgan_model_info": "自动模式: 2x/3x用AnimevideV3, 4x用plus-anime",
        "esrgan_animevideo": "AnimevideV3 (动漫视频)",
        "esrgan_x4plus": "x4plus (通用照片)",
        "esrgan_x4plus_anime": "x4plus-anime (动漫图片)",
        
        # Waifu2x 模型
        "waifu_model_select": "模型选择",
        "waifu_model_info": "CUNet效果最佳, Photo适合真实照片",
        "waifu_cunet": "CUNet (默认)",
        "waifu_anime": "Anime Style Art RGB",
        "waifu_photo": "Photo",
        "waifu_denoise_info": "-1=无效果, 0-3=降噪强度递增",
        "waifu_scale_info": "支持1/2/4/8/16/32",
        
        # Anime4K 模型
        "anime4k_model_select": "模型选择",
        "anime4k_model_info": "acnet-gan效果更好，acnet速度更快",
        "anime4k_acnet": "ACNet (快速)",
        "anime4k_acnet_gan": "ACNet-GAN (高质量)",
        "anime4k_processor": "处理器类型",
        "anime4k_processor_info": "OpenCL兼容性最好，CUDA需要NVIDIA显卡",
        "anime4k_device": "设备索引",
        "anime4k_device_info": "一般设为0即可，多显卡时可选择",
        "anime4k_scale_info": "支持小数倍率如1.5, 2.5",
        "anime4k_not_installed": "Anime4K 未安装，请下载 Anime4KCPP 并放入模型文件夹",
        
        # 结果
        "process_result": "处理结果",
        "result_preview": "结果预览",
        "download": "⬇️ 下载",
        "zoom": "🔍 放大",
        "effect_compare": "效果对比",
        
        # 预设管理
        "preset_manage": "💾 预设管理",
        "preset_name": "预设名称",
        "preset_name_placeholder": "输入名称...",
        "save": "保存",
        "load": "加载",
        "saved_presets": "已保存预设",
        "rename": "重命名",
        "delete": "删除",
        "new_name": "新名称",
        "new_name_placeholder": "输入新名称...",
        "operation_status": "操作状态",
        
        # 图库
        "gallery_title": "输出图片库",
        "gallery_desc": "浏览所有处理过的图片，点击可查看详情和超分参数。",
        "image_list": "图片列表",
        "refresh_gallery": "刷新图库",
        "delete_selected": "删除选中",
        "click_view_detail": "点击图片查看详情",
        "image_preview": "图片预览",
        "preview": "预览",
        "image_info": "图片信息",
        "detail_info": "详细信息",
        "select_show": "选择图片后显示",
        
        # 设置
        "network_share": "网络分享设置",
        "share_desc": "启用公开链接后, 可以通过互联网访问此工具(使用Gradio隧道服务)。",
        "enable_share": "启用公开链接",
        "enable_share_info": "启用后将生成可公开访问的链接",
        "save_settings": "保存设置",
        "config_status": "配置状态",
        "current_config": "当前配置",
        "share_enabled": "已启用公开链接",
        "local_only": "仅本地访问",
        "access_address": "访问地址",
        "local_address": "本地访问地址",
        "public_link": "公开链接",
        "not_enabled": "未启用公开链接",
        "generating": "未启用或正在生成中...",
        "refresh_link": "刷新公开链接",
        "settings_note": """### 说明
- 保存设置后需要重新启动程序才能生效
- 公开链接生成可能需要几秒到几十秒
- 公开链接有效期为72小时
- 请勿在公开链接上处理敏感图片
- 如果公开链接长时间未显示, 请点击刷新按钮
- 本地访问地址始终可用""",
        "engine_status": "引擎状态",
        "dir_info": "目录信息",
        "output_dir": "输出目录",
        "program_dir": "程序目录",
        "data_dir": "数据目录",
        "temp_dir": "临时目录",
        "config_file": "配置文件",
        "preset_file": "预设文件",
        "custom_presets": "自定义预设",
        "saved_presets_count": "已保存 {count} 个自定义预设",
        
        # 帮助
        "help_engines": "引擎介绍",
        "help_presets": "预设说明",
        "help_faq": "常见问题",
        "help_about": "关于",
        "help_video": "视频处理指南",
        "help_video_content": """
## 视频处理 (Beta) 使用指南

### 功能概述
视频处理功能可以对视频进行逐帧超分辨率放大，自动提取帧、处理、重组为高清视频。支持多音轨和字幕轨保留。

### 超分引擎选择
参考 [SVFI 模型选择说明](https://doc.svfi.group/pages/model-spec/)：

| 引擎 | 适用场景 | 速度 | 特点 |
|------|----------|------|------|
| **Anime4K** | 动漫2D | ⚡极速 | 实时超分，适合预览，较保守 |
| **RealCUGAN Pro** | 动漫2D | 🔵中等 | 效果优秀，推荐动漫使用 |
| **RealCUGAN SE** | 动漫2D | 🔵中等 | 速度与质量平衡 |
| **RealESRGAN Anime** | 动漫2D | 🔵中等 | 脑补细节，画面更锐利艳丽 |
| **waifu2x** | 动漫2D | 🟠较慢 | 经典算法，保守稳定 |
| **RealESRGAN** | 写实/3D | 🔵中等 | 通用模型，3D/实拍均可 |

### 编码设置建议
参考 [SVFI 高级设置](https://doc.svfi.group/pages/advanced-settings/)：

**编码器选择：**
- **H.264 (AVC)**：兼容性最佳，适合分享上传
- **H.265 (HEVC)**：压缩效率更高，2K/4K推荐使用
- **VP9**：开源高效，适合WebM格式

**CRF质量参数：**
- **16**：收藏级质量，文件较大
- **18-23**：日常使用推荐范围
- **28+**：快速预览，质量较低

**编码速度：**
- 速度越慢，压缩效率越高，文件越小
- 推荐日常使用 **medium**
- 追求质量使用 **slow** 或 **veryslow**

### 降噪设置
仅 RealCUGAN 和 waifu2x 支持降噪：
- **-1**：不降噪（保持原始）
- **0-1**：轻微降噪
- **2-3**：强降噪（慎用，可能丢失细节）

### 注意事项
1. 视频处理耗时较长，建议先用短片段测试效果
2. 4K输出建议使用H.265编码以减小文件体积
3. MKV格式支持多音轨和字幕轨保留
""",
        
        # 语言设置（新增到设置中）
        "language_settings": "语言设置",
        "language_desc": "切换界面语言，保存后刷新页面生效。",
        
        # 固定预设
        "pinned_presets": "固定预设",
        "pinned_presets_desc": "选择要固定在首页「选择预设」中的自定义预设。",
        "pinned_count": "已固定 {count} 个预设",
        "pin_preset": "固定到首页",
        "unpin_preset": "取消固定",
        
        # 预设分类
        "author_presets": "系统预设",
        "user_presets": "用户预设",
        "no_user_presets": "暂无用户预设，请在「自定义处理」中保存预设后在「设置」中固定",
        
        # GIF处理
        "gif_processing": "正在处理GIF...",
        "gif_frame": "帧 {current}/{total}",
        "gif_done": "GIF处理完成，共{count}帧",
        "gif_error": "GIF处理失败: {error}",
        "gif_compare_note": "注意：GIF动画在滑动对比中可能不同步，请点击下方展开查看原图和结果",
        
        # 视频处理 (Beta)
        "tab_video": "视频处理 (Beta)",
        "video_desc": "上传视频，选择超分引擎，自动逐帧处理后重组为高清视频。支持多音轨/字幕保留。",
        "upload_video": "上传视频",
        
        # 视频引擎选项
        "video_engine": "超分引擎",
        "video_engine_info": "动漫素材推荐 Anime4K/RealCUGAN，写实素材推荐 RealESRGAN",
        "video_engine_anime4k": "Anime4K - 动漫极速",
        "video_engine_realcugan_pro": "RealCUGAN Pro - 动漫高质",
        "video_engine_realcugan_se": "RealCUGAN SE - 动漫标准",
        "video_engine_realesrgan_anime": "RealESRGAN Anime - 动漫通用",
        "video_engine_waifu2x": "waifu2x CuNet - 动漫经典",
        "video_engine_realesrgan": "RealESRGAN - 写实通用",
        
        # 超分设置
        "video_sr_settings": "超分模型设置",
        "video_scale": "放大倍数",
        "video_scale_info": "2x适合1080p→4K，4x适合480p→1080p",
        "video_denoise": "降噪强度",
        "video_denoise_info": "RealCUGAN/waifu2x专用，数值越大降噪越强",
        "video_denoise_none": "无降噪",
        "video_denoise_light": "轻微降噪 (0)",
        "video_denoise_medium": "中度降噪 (1)",
        "video_denoise_strong": "强降噪 (2)",
        "video_denoise_max": "极强降噪 (3)",
        
        # 引擎特定设置
        "video_engine_settings": "引擎特定设置",
        "video_a4k_mode": "Anime4K 处理模式",
        "video_a4k_mode_native": "原生模式",
        "video_a4k_mode_builtin": "AIS 内置模式",
        "video_a4k_mode_native_info": "由 Anime4KCPP 直接处理视频，速度最快\n编码设置不生效 (CRF/预设等)，取消按钮无法终止处理",
        "video_a4k_mode_builtin_info": "批量处理帧后用 FFmpeg 编码，避免重复初始化 GPU\n编码设置完全可控，支持取消处理",
        "video_a4k_batch_size": "批量处理帧数",
        "video_a4k_batch_size_info": "每批次处理的帧数，值越大越快但内存占用越高",
        "video_segment_duration": "切片时长 (秒)",
        "video_segment_duration_info": "将视频分割成小段处理，避免内存溢出。0=不分割(仅小视频)",
        "video_segment_processing": "切片处理",
        "video_segment_status": "处理切片 {current}/{total}",
        "video_a4k_model": "Anime4K 模型",
        "video_a4k_model_acnet": "ACNet (默认)",
        "video_a4k_model_acnet_gan": "ACNet GAN (高质量)",
        "video_a4k_processor": "Anime4K 处理器",
        "video_a4k_processor_cuda": "CUDA (NVIDIA GPU)",
        "video_a4k_processor_opencl": "OpenCL (通用GPU)",
        "video_a4k_processor_cpu": "CPU (兼容模式)",
        "video_a4k_device": "GPU 设备索引",
        "video_native_mode_warning": "原生模式下，编码器/CRF/速度预设不生效",
        "video_cugan_model": "RealCUGAN 模型类型",
        "video_cugan_model_pro": "Pro (高质量)",
        "video_cugan_model_se": "SE (标准效率)",
        "video_cugan_model_nose": "Nose (无锐化)",
        "video_cugan_syncgap": "同步间隔 (SyncGap)",
        "video_cugan_syncgap_info": "值越大处理越快但可能略有瑕疵，0为逐帧处理",
        "video_esrgan_model": "RealESRGAN 模型",
        "video_esrgan_model_anime": "AnimevideV3 (动漫)",
        "video_esrgan_model_x4plus": "x4plus (通用)",
        "video_esrgan_model_anime_x4plus": "x4plus-anime (动漫强化)",
        "video_esrgan_tta": "TTA 模式",
        "video_esrgan_tta_info": "测试时增强，质量更好但速度慢4倍",
        "video_waifu2x_model": "Waifu2x 模型",
        "video_waifu2x_model_cunet": "CuNet (默认，高质量)",
        "video_waifu2x_model_anime": "Anime Style Art (风格化)",
        "video_waifu2x_model_photo": "Photo (真实照片)",
        "video_waifu2x_tta": "TTA 模式",
        "video_waifu2x_tta_info": "测试时增强，质量更好但速度慢8倍",
        "video_gpu_device": "GPU 设备",
        "video_gpu_device_info": "选择处理使用的 GPU，-1 为自动选择",
        
        # 输出分辨率
        "video_output_resolution": "输出分辨率",
        "video_output_resolution_info": "可选择预设分辨率或自定义",
        "video_res_auto": "自动 (按倍数放大)",
        "video_res_1080p": "1080p (1920×1080)",
        "video_res_2k": "2K (2560×1440)",
        "video_res_4k": "4K (3840×2160)",
        "video_res_custom": "自定义",
        "video_custom_width": "自定义宽度",
        "video_custom_height": "自定义高度",
        
        # 视频编码器选项
        "video_codec_h264": "H.264 (AVC) - 兼容性最佳",
        "video_codec_h265": "H.265 (HEVC) - 高效压缩",
        "video_codec_vp9": "VP9 (WebM) - 开源高效",
        
        # 视频输出设置
        "video_output_format": "输出格式",
        "video_output_format_info": "MP4兼容性最好，MKV支持多轨道，WebM开源高效",
        "video_codec": "视频编码器",
        "video_codec_info": "H.264兼容性最佳，H.265压缩效率更高(4K推荐)",
        "video_crf": "压缩质量 (CRF)",
        "video_crf_info": "数值越小画质越高，16为收藏级，18-23推荐日常使用",
        "video_preset": "编码速度",
        "video_preset_info": "速度越慢画质越好，文件越小",
        "video_keep_audio": "保留音轨",
        "video_fps_override": "自定义帧率 (留空=原始)",
        "video_fps_info": "设置输出视频帧率，留空保持原始帧率",
        "video_start_process": "开始处理视频",
        "video_processing": "正在处理视频...",
        "video_extracting": "正在提取帧...",
        "video_upscaling": "正在超分处理帧 {current}/{total}",
        "video_reassembling": "正在重组视频...",
        "video_done": "视频处理完成",
        "video_error": "视频处理失败: {error}",
        "video_info": "视频信息: {width}x{height}, {fps} FPS, 时长: {duration}秒",
        "video_frame_count": "预计帧数: {count}",
        "video_result": "处理结果视频",
        "video_download": "视频下载",
        "video_keep_subtitles": "保留字幕轨",
        "video_audio_tracks": "音轨数",
        "video_subtitle_tracks": "字幕轨数",
        "video_progress": "处理进度",
        "video_format": "封装格式",
        "video_codec_info": "视频编码",
        "video_file_size": "文件大小",
        "video_bitrate": "总比特率",
        "video_cancel": "取消处理",
        "video_cancelled": "任务已取消",
        "video_resume": "继续处理",
        "video_resume_found": "发现未完成的任务: {name} ({progress}%)",
        "reset_eta_data": "重置ETA估算数据",
        "reset_eta_confirm": "确定要重置所有ETA估算数据吗？",
        "reset_eta_done": "ETA估算数据已重置",
        
        # 内存监控
        "memory_warning": "⚠️ 内存使用率过高 ({usage}%)，建议关闭其他程序",
        "memory_critical": "🚨 内存严重不足 ({usage}%)，可能导致程序崩溃",
        "memory_info": "内存: {used:.1f}/{total:.1f} GB ({usage}%)",
        
        # 实时日志
        "video_log": "实时日志",
        "video_log_clear": "清除日志",
        
        # 视频处理预设模板
        "video_preset_template": "快捷预设",
        "video_preset_template_info": "选择预设自动配置参数，或选择自定义手动调整",
        "video_preset_custom": "自定义",
        "video_preset_fast": "快速预览 (极速编码/低画质)",
        "video_preset_balanced": "均衡模式 (速度/画质平衡)",
        "video_preset_hq": "高质量 (慢速编码/高画质)",
        "video_preset_ultra": "极致画质 (超慢编码/最高画质)",
        
        # 编码速度选项（中文化）
        "video_speed_ultrafast": "极速 (质量最低)",
        "video_speed_superfast": "超快",
        "video_speed_veryfast": "很快",
        "video_speed_faster": "较快",
        "video_speed_fast": "快速",
        "video_speed_medium": "中等 (推荐)",
        "video_speed_slow": "慢速",
        "video_speed_slower": "较慢",
        "video_speed_veryslow": "很慢 (质量最高)",
        
        # 图库缩略图
        "loading_thumbnail": "加载缩略图...",
        "click_load_original": "原图",
        "thumbnail_size": "缩略图大小",
        
        # 消息
        "msg_error": "[错误]",
        "msg_success": "[完成]",
        "msg_warning": "[警告]",
        "msg_info": "[提示]",
        "msg_upload_first": "请先上传图片",
        "msg_saved_to": "保存至",
        "msg_failed": "[失败]",
        "msg_process_log": "处理日志",
        "msg_preset_empty": "预设名称不能为空",
        "msg_preset_saved": "预设 '{name}' 已保存",
        "msg_preset_deleted": "预设 '{name}' 已删除",
        "msg_preset_renamed": "已重命名为 '{name}'",
        "msg_preset_not_exist": "预设 '{name}' 不存在",
        "msg_save_failed": "保存失败",
        "msg_select_preset": "请先选择预设",
        "msg_opened_explorer": "已在文件管理器中打开",
        "msg_no_image": "暂无图片",
        "msg_refresh_count": "[刷新] 共 {count} 张图片",
        "msg_select_delete": "请先选择要删除的图片",
        "msg_image_deleted": "图片已删除",
        "msg_settings_saved": "[已保存] {status} - 请重启程序使设置生效",
        "msg_running": "正在执行: {name}",
        "msg_done": "[OK] {name} 完成",
        "msg_fail": "[X] {name} 失败: {error}",
        
        # 选项字典
        "_choices": {
            "gpu": [("自动", -2), ("CPU", -1), ("GPU 0", 0), ("GPU 1", 1)],
            "format": ["png", "jpg", "webp"],
            "gif_format": [
                ("WebP (无色带, 推荐)", "webp"),
                ("GIF (256色, 兼容性好)", "gif")
            ],
            "cugan_model": ["SE (标准版)", "Pro (专业版)"],
            "cugan_denoise": ["无降噪", "保守降噪", "强力降噪"],
            "esrgan_model": [
                ("自动选择", "auto"),
                ("AnimevideV3 (动漫视频)", "realesr-animevideov3"),
                ("x4plus (通用照片)", "realesrgan-x4plus"),
                ("x4plus-anime (动漫图片)", "realesrgan-x4plus-anime")
            ],
            "waifu_model": [
                ("CUNet (默认)", "cunet"),
                ("Anime Style Art RGB", "upconv_7_anime_style_art_rgb"),
                ("Photo", "upconv_7_photo")
            ],
            "anime4k_model": [
                ("ACNet-GAN (质量最佳)", "acnet-gan"),
                ("ACNet (标准)", "acnet")
            ],
            "anime4k_processor": [
                ("CUDA (NVIDIA显卡)", "cuda"),
                ("OpenCL (通用)", "opencl"),
                ("CPU", "cpu")
            ],
            "compare_sources": ["原图", "通用增强", "烂图修复", "壁纸制作", "极致柔化", "快速超分"],
        }
    },
    
    # ========== English ==========
    "en": {
        # General
        "app_title": "AI Image Super-Resolution",
        "app_subtitle": "Engine Status",
        "language": "Language",
        "lang_zh": "简体中文",
        "lang_en": "English",
        
        # Tabs
        "tab_quick": "Quick Process",
        "tab_custom": "Custom Mode",
        "tab_gallery": "Gallery",
        "tab_settings": "Settings",
        "tab_help": "Help",
        
        # Quick Process
        "quick_desc": "Upload an image, select a preset, and process with one click. Use the slider to compare original and result.",
        "upload_image": "Upload Image",
        "select_preset": "Select Preset",
        "start_process": "Start Process",
        "process_all": "Process All",
        "status": "Status",
        "compare_title": "Comparison (Drag the divider)",
        "compare_label": "<- Original | Result ->",
        "click_zoom": "Click to Zoom",
        "click_zoom_tip": "Click image to view details",
        "original": "Original",
        "result": "Result",
        "view_result": "View Result Only",
        "result_download": "Result (Click to download)",
        
        # Presets
        "preset_universal": "Universal Enhance",
        "preset_repair": "Image Repair",
        "preset_wallpaper": "Wallpaper Maker",
        "preset_soft": "Ultra Soft",
        "preset_anime4k": "Fast Upscale",
        "preset_universal_desc": "Real-CUGAN Pro 2x conservative denoise, suitable for most scenarios",
        "preset_repair_desc": "Real-ESRGAN 4x, powerful repair for low-quality images",
        "preset_wallpaper_desc": "Real-CUGAN SE 4x no denoise, preserve details for HD wallpapers",
        "preset_soft_desc": "Waifu2x 2x strong denoise, soft and delicate output",
        "preset_anime4k_desc": "Anime4K 2x fast processing, ideal for animations and videos with many frames",
        
        # Comparison
        "all_preset_compare": "All Presets Comparison",
        "all_preset_desc": "Run all 4 presets at once for easy comparison. Choose left/right sources freely.",
        "free_compare": "Free Compare",
        "left_source": "Left Source",
        "right_source": "Right Source",
        
        # Custom Mode
        "custom_desc": "Select engine and adjust parameters for professional needs. Click 'Advanced Options' for more.",
        "model_version": "Model Version",
        "model_version_info": "Pro has better quality, SE is faster",
        "scale_ratio": "Scale Ratio",
        "denoise_level": "Denoise Level",
        "denoise_level_info": "-1=none, 0=conservative, 3=strong",
        "advanced_options": "⚙️ Advanced Options",
        "advanced_params": "⚙️ Output Settings",
        "syncgap_mode": "SyncGap Mode",
        "syncgap_info": "0=none, 1=accurate, 2=rough, 3=very rough(default)",
        "tile_size": "Tile Size",
        "tile_info": "0=auto, smaller value uses less VRAM",
        "tta_mode": "TTA Mode",
        "tta_info": "8x time for better quality",
        "gpu_select": "GPU Select",
        "threads": "Threads (load:proc:save)",
        "threads_info": "Use 4:4:4 for small images, 2:2:2 for large",
        "output_format": "Output Format",
        "gif_output_format": "Animation Output Format",
        "gif_output_format_info": "WebP supports 24-bit color with no banding, GIF has 256 colors but better compatibility",
        
        # Model Options
        "model_se": "SE (Standard)",
        "model_pro": "Pro (Professional)",
        "no_denoise": "No Denoise",
        "conservative_denoise": "Conservative",
        "strong_denoise": "Strong",
        "auto_select": "Auto Select",
        "auto": "Auto",
        "cpu": "CPU",
        
        # ESRGAN Models
        "esrgan_model_select": "Model Select",
        "esrgan_model_info": "Auto: 2x/3x uses AnimevideV3, 4x uses plus-anime",
        "esrgan_animevideo": "AnimevideV3 (Anime Video)",
        "esrgan_x4plus": "x4plus (General Photo)",
        "esrgan_x4plus_anime": "x4plus-anime (Anime Image)",
        
        # Waifu2x Models
        "waifu_model_select": "Model Select",
        "waifu_model_info": "CUNet is best, Photo for real photos",
        "waifu_cunet": "CUNet (Default)",
        "waifu_anime": "Anime Style Art RGB",
        "waifu_photo": "Photo",
        "waifu_denoise_info": "-1=no effect, 0-3=increasing strength",
        "waifu_scale_info": "Supports 1/2/4/8/16/32",
        
        # Anime4K Models
        "anime4k_model_select": "Model Select",
        "anime4k_model_info": "acnet-gan for quality, acnet for speed",
        "anime4k_acnet": "ACNet (Fast)",
        "anime4k_acnet_gan": "ACNet-GAN (High Quality)",
        "anime4k_processor": "Processor Type",
        "anime4k_processor_info": "OpenCL has best compatibility, CUDA requires NVIDIA GPU",
        "anime4k_device": "Device Index",
        "anime4k_device_info": "Usually 0, select for multi-GPU",
        "anime4k_scale_info": "Supports decimal scales like 1.5, 2.5",
        "anime4k_not_installed": "Anime4K not installed, please download Anime4KCPP and place in models folder",
        
        # Results
        "process_result": "Process Result",
        "result_preview": "Result Preview",
        "download": "⬇️ Download",
        "zoom": "🔍 Zoom",
        "effect_compare": "Effect Comparison",
        
        # Preset Management
        "preset_manage": "💾 Preset Management",
        "preset_name": "Preset Name",
        "preset_name_placeholder": "Enter name...",
        "save": "Save",
        "load": "Load",
        "saved_presets": "Saved Presets",
        "rename": "Rename",
        "delete": "Delete",
        "new_name": "New Name",
        "new_name_placeholder": "Enter new name...",
        "operation_status": "Operation Status",
        
        # Gallery
        "gallery_title": "Output Gallery",
        "gallery_desc": "Browse all processed images. Click to view details and upscale parameters.",
        "image_list": "Image List",
        "refresh_gallery": "Refresh Gallery",
        "delete_selected": "Delete Selected",
        "click_view_detail": "Click image for details",
        "image_preview": "Image Preview",
        "preview": "Preview",
        "image_info": "Image Info",
        "detail_info": "Details",
        "select_show": "Select image to show",
        
        # Settings
        "network_share": "Network Sharing",
        "share_desc": "Enable public link to access this tool via internet (using Gradio tunnel).",
        "enable_share": "Enable Public Link",
        "enable_share_info": "Will generate a publicly accessible link",
        "save_settings": "Save Settings",
        "config_status": "Config Status",
        "current_config": "Current Config",
        "share_enabled": "Public link enabled",
        "local_only": "Local access only",
        "access_address": "Access Address",
        "local_address": "Local Address",
        "public_link": "Public Link",
        "not_enabled": "Public link not enabled",
        "generating": "Not enabled or generating...",
        "refresh_link": "Refresh Public Link",
        "settings_note": """### Notes
- Restart the program after saving for changes to take effect
- Public link generation may take a few seconds to minutes
- Public links are valid for 72 hours
- Do not process sensitive images on public links
- Click refresh if the public link doesn't appear
- Local address is always available""",
        "engine_status": "Engine Status",
        "dir_info": "Directory Info",
        "output_dir": "Output Dir",
        "program_dir": "Program Dir",
        "data_dir": "Data Dir",
        "temp_dir": "Temp Dir",
        "config_file": "Config File",
        "preset_file": "Preset File",
        "custom_presets": "Custom Presets",
        "saved_presets_count": "{count} custom presets saved",
        
        # Help
        "help_engines": "Engine Guide",
        "help_presets": "Preset Guide",
        "help_faq": "FAQ",
        "help_about": "About",
        "help_video": "Video Processing Guide",
        "help_video_content": """
## Video Processing (Beta) Guide

### Overview
Video processing upscales videos frame-by-frame, automatically extracting frames, processing, and reassembling into HD video. Multi-track audio and subtitles are preserved.

### Engine Selection
Reference: [SVFI Model Guide](https://doc.svfi.group/pages/model-spec/)

| Engine | Use Case | Speed | Features |
|--------|----------|-------|----------|
| **Anime4K** | 2D Anime | ⚡Ultra-fast | Real-time, good for preview |
| **RealCUGAN Pro** | 2D Anime | 🔵Medium | Excellent quality, recommended |
| **RealCUGAN SE** | 2D Anime | 🔵Medium | Speed/quality balanced |
| **RealESRGAN Anime** | 2D Anime | 🔵Medium | Adds detail, sharper image |
| **waifu2x** | 2D Anime | 🟠Slower | Classic, stable algorithm |
| **RealESRGAN** | Real-life/3D | 🔵Medium | General purpose |

### Encoding Settings
Reference: [SVFI Advanced Settings](https://doc.svfi.group/pages/advanced-settings/)

**Codec Selection:**
- **H.264 (AVC)**: Best compatibility
- **H.265 (HEVC)**: Better compression, recommended for 2K/4K
- **VP9**: Open-source, efficient

**CRF Quality:**
- **16**: Archival quality, larger file
- **18-23**: Daily use recommended
- **28+**: Quick preview, lower quality

**Encoding Speed:**
- Slower = better compression, smaller file
- Recommended: **medium** for daily use
- Use **slow** or **veryslow** for best quality

### Denoise Settings
Only RealCUGAN and waifu2x support denoising:
- **-1**: No denoise (preserve original)
- **0-1**: Light denoise
- **2-3**: Strong denoise (use carefully)

### Notes
1. Video processing is time-consuming, test with short clips first
2. For 4K output, use H.265 to reduce file size
3. MKV format preserves multi-track audio and subtitles
""",
        
        # Language Settings (added to settings)
        "language_settings": "Language Settings",
        "language_desc": "Switch interface language. Refresh page after saving.",
        
        # Pinned Presets
        "pinned_presets": "Pinned Presets",
        "pinned_presets_desc": "Select custom presets to pin to homepage 'Select Preset'.",
        "pinned_count": "{count} presets pinned",
        "pin_preset": "Pin to Homepage",
        "unpin_preset": "Unpin",
        
        # Preset Categories
        "author_presets": "System Presets",
        "user_presets": "User Presets",
        "no_user_presets": "No user presets yet. Save presets in 'Custom Process' and pin them in 'Settings'",
        
        # GIF Processing
        "gif_processing": "Processing GIF...",
        "gif_frame": "Frame {current}/{total}",
        "gif_done": "GIF done, {count} frames total",
        "gif_error": "GIF processing failed: {error}",
        "gif_compare_note": "Note: GIF animations may not sync in slider comparison. Click below to view original and result separately.",
        
        # Video Processing (Beta)
        "tab_video": "Video Process (Beta)",
        "video_desc": "Upload video, select upscaling engine, process frame-by-frame, and reassemble into HD video. Multi-track audio/subtitles supported.",
        "upload_video": "Upload Video",
        
        # Video engine options
        "video_engine": "Upscaling Engine",
        "video_engine_info": "Anime: Anime4K/RealCUGAN recommended. Real-life: RealESRGAN recommended",
        "video_engine_anime4k": "Anime4K - Anime Ultra Fast",
        "video_engine_realcugan_pro": "RealCUGAN Pro - Anime High Quality",
        "video_engine_realcugan_se": "RealCUGAN SE - Anime Standard",
        "video_engine_realesrgan_anime": "RealESRGAN Anime - Anime General",
        "video_engine_waifu2x": "waifu2x CuNet - Anime Classic",
        "video_engine_realesrgan": "RealESRGAN - Real-life General",
        
        # SR settings
        "video_sr_settings": "Super Resolution Settings",
        "video_scale": "Scale Factor",
        "video_scale_info": "2x for 1080p→4K, 4x for 480p→1080p",
        "video_denoise": "Denoise Level",
        "video_denoise_info": "RealCUGAN/waifu2x only. Higher value = stronger denoising",
        "video_denoise_none": "No Denoise",
        "video_denoise_light": "Light Denoise (0)",
        "video_denoise_medium": "Medium Denoise (1)",
        "video_denoise_strong": "Strong Denoise (2)",
        "video_denoise_max": "Max Denoise (3)",
        
        # Engine-specific settings
        "video_engine_settings": "Engine-Specific Settings",
        "video_a4k_mode": "Anime4K Processing Mode",
        "video_a4k_mode_native": "Native Mode (Ultra Fast)",
        "video_a4k_mode_builtin": "AIS Built-in Mode (Adjustable)",
        "video_a4k_mode_native_info": "⚡ Ultra Fast: Direct video processing by Anime4KCPP\n⚠️ Encoding settings won't apply (CRF/preset etc.), cancel button won't stop processing",
        "video_a4k_mode_builtin_info": "🔧 Flexible: Batch frame processing with FFmpeg encoding, avoids GPU re-init\n✅ Full encoding control, supports cancellation",
        "video_a4k_batch_size": "Batch Size (Frames)",
        "video_a4k_batch_size_info": "Frames per batch, higher = faster but more memory",
        "video_segment_duration": "Segment Duration (sec)",
        "video_segment_duration_info": "Split video into smaller segments to avoid memory overflow. 0=no split (small videos only)",
        "video_segment_processing": "Segment Processing",
        "video_segment_status": "Processing segment {current}/{total}",
        "video_a4k_model": "Anime4K Model",
        "video_a4k_model_acnet": "ACNet (Default)",
        "video_a4k_model_acnet_gan": "ACNet GAN (High Quality)",
        "video_a4k_processor": "Anime4K Processor",
        "video_a4k_processor_cuda": "CUDA (NVIDIA GPU)",
        "video_a4k_processor_opencl": "OpenCL (Universal GPU)",
        "video_a4k_processor_cpu": "CPU (Compatibility Mode)",
        "video_a4k_device": "GPU Device Index",
        "video_native_mode_warning": "⚠️ In native mode, codec/CRF/speed preset won't apply",
        "video_cugan_model": "RealCUGAN Model Type",
        "video_cugan_model_pro": "Pro (High Quality)",
        "video_cugan_model_se": "SE (Standard Efficiency)",
        "video_cugan_model_nose": "Nose (No Sharpening)",
        "video_cugan_syncgap": "Sync Gap",
        "video_cugan_syncgap_info": "Higher = faster but may have minor artifacts, 0 = frame-by-frame",
        "video_esrgan_model": "RealESRGAN Model",
        "video_esrgan_model_anime": "AnimevideV3 (Anime)",
        "video_esrgan_model_x4plus": "x4plus (General)",
        "video_esrgan_model_anime_x4plus": "x4plus-anime (Anime Enhanced)",
        "video_esrgan_tta": "TTA Mode",
        "video_esrgan_tta_info": "Test-Time Augmentation, better quality but 4x slower",
        "video_waifu2x_model": "Waifu2x Model",
        "video_waifu2x_model_cunet": "CuNet (Default, High Quality)",
        "video_waifu2x_model_anime": "Anime Style Art (Stylized)",
        "video_waifu2x_model_photo": "Photo (Real Photos)",
        "video_waifu2x_tta": "TTA Mode",
        "video_waifu2x_tta_info": "Test-Time Augmentation, better quality but 8x slower",
        "video_gpu_device": "GPU Device",
        "video_gpu_device_info": "Select GPU for processing, -1 for auto-select",
        
        # Output resolution
        "video_output_resolution": "Output Resolution",
        "video_output_resolution_info": "Choose preset resolution or custom",
        "video_res_auto": "Auto (by scale factor)",
        "video_res_1080p": "1080p (1920×1080)",
        "video_res_2k": "2K (2560×1440)",
        "video_res_4k": "4K (3840×2160)",
        "video_res_custom": "Custom",
        "video_custom_width": "Custom Width",
        "video_custom_height": "Custom Height",
        
        # Video codec options
        "video_codec_h264": "H.264 (AVC) - Best Compatibility",
        "video_codec_h265": "H.265 (HEVC) - Efficient Compression",
        "video_codec_vp9": "VP9 (WebM) - Open Efficient",
        
        # Video output settings
        "video_output_format": "Output Format",
        "video_output_format_info": "MP4 best compatibility, MKV multi-track, WebM open efficient",
        "video_codec": "Video Codec",
        "video_codec_info": "H.264 best compatibility, H.265 better compression (4K recommended)",
        "video_crf": "Quality (CRF)",
        "video_crf_info": "Lower = higher quality. 16 archival, 18-23 daily use recommended",
        "video_preset": "Encoding Speed",
        "video_preset_info": "Slower = better quality, smaller file",
        "video_keep_audio": "Keep Audio",
        "video_fps_override": "Custom FPS (empty=original)",
        "video_fps_info": "Set output video frame rate, leave empty to keep original",
        "video_start_process": "Start Video Processing",
        "video_processing": "Processing video...",
        "video_extracting": "Extracting frames...",
        "video_upscaling": "Upscaling frame {current}/{total}",
        "video_reassembling": "Reassembling video...",
        "video_done": "Video processing complete",
        "video_error": "Video processing failed: {error}",
        "video_info": "Video info: {width}x{height}, {fps} FPS, Duration: {duration}s",
        "video_frame_count": "Estimated frames: {count}",
        "video_result": "Processed Video",
        "video_download": "Download Video",
        "video_keep_subtitles": "Keep Subtitles",
        "video_audio_tracks": "Audio Tracks",
        "video_subtitle_tracks": "Subtitle Tracks",
        "video_progress": "Processing Progress",
        "video_format": "Format",
        "video_codec_info": "Video Codec",
        "video_file_size": "File Size",
        "video_bitrate": "Bitrate",
        "video_cancel": "Cancel",
        "video_cancelled": "Task Cancelled",
        "video_resume": "Resume",
        "video_resume_found": "Found unfinished task: {name} ({progress}%)",
        "reset_eta_data": "Reset ETA Data",
        "reset_eta_confirm": "Are you sure to reset all ETA estimation data?",
        "reset_eta_done": "ETA estimation data has been reset",
        
        # Memory monitoring
        "memory_warning": "⚠️ High memory usage ({usage}%), consider closing other programs",
        "memory_critical": "🚨 Critical memory shortage ({usage}%), may cause crash",
        "memory_info": "Memory: {used:.1f}/{total:.1f} GB ({usage}%)",
        
        # Real-time log
        "video_log": "Real-time Log",
        "video_log_clear": "Clear Log",
        
        # Video preset templates
        "video_preset_template": "Quick Preset",
        "video_preset_template_info": "Select preset to auto-configure, or choose Custom to adjust manually",
        "video_preset_custom": "Custom",
        "video_preset_fast": "Fast Preview (fastest/low quality)",
        "video_preset_balanced": "Balanced (speed/quality trade-off)",
        "video_preset_hq": "High Quality (slow encoding/high quality)",
        "video_preset_ultra": "Ultra Quality (very slow/best quality)",
        
        # Encoding speed options
        "video_speed_ultrafast": "Ultrafast (lowest quality)",
        "video_speed_superfast": "Superfast",
        "video_speed_veryfast": "Veryfast",
        "video_speed_faster": "Faster",
        "video_speed_fast": "Fast",
        "video_speed_medium": "Medium (recommended)",
        "video_speed_slow": "Slow",
        "video_speed_slower": "Slower",
        "video_speed_veryslow": "Veryslow (highest quality)",
        "video_no_ffmpeg": "FFmpeg not found, cannot process video",
        "video_too_large": "Video file too large, recommend less than 500MB",
        "video_unsupported": "Unsupported video format",
        
        # Gallery Thumbnail
        "loading_thumbnail": "Loading thumbnail...",
        "click_load_original": "Click to load original",
        "thumbnail_size": "Thumbnail Size",
        
        # Messages
        "msg_error": "[Error]",
        "msg_success": "[Done]",
        "msg_warning": "[Warning]",
        "msg_info": "[Info]",
        "msg_upload_first": "Please upload an image first",
        "msg_saved_to": "Saved to",
        "msg_failed": "[Failed]",
        "msg_process_log": "Process Log",
        "msg_preset_empty": "Preset name cannot be empty",
        "msg_preset_saved": "Preset '{name}' saved",
        "msg_preset_deleted": "Preset '{name}' deleted",
        "msg_preset_renamed": "Renamed to '{name}'",
        "msg_preset_not_exist": "Preset '{name}' does not exist",
        "msg_save_failed": "Save failed",
        "msg_select_preset": "Please select a preset first",
        "msg_opened_explorer": "Opened in file explorer",
        "msg_no_image": "No image",
        "msg_refresh_count": "[Refresh] {count} images total",
        "msg_select_delete": "Please select an image to delete first",
        "msg_image_deleted": "Image deleted",
        "msg_settings_saved": "[Saved] {status} - Please restart for changes to take effect",
        "msg_running": "Running: {name}",
        "msg_done": "[OK] {name} done",
        "msg_fail": "[X] {name} failed: {error}",
        
        # Choices Dictionary
        "_choices": {
            "gpu": [("Auto", -2), ("CPU", -1), ("GPU 0", 0), ("GPU 1", 1)],
            "format": ["png", "jpg", "webp"],
            "gif_format": [
                ("WebP (No Banding, Recommended)", "webp"),
                ("GIF (256 Colors, Compatible)", "gif")
            ],
            "cugan_model": ["SE (Standard)", "Pro (Professional)"],
            "cugan_denoise": ["No Denoise", "Conservative", "Strong"],
            "esrgan_model": [
                ("Auto Select", "auto"),
                ("AnimevideV3 (Anime Video)", "realesr-animevideov3"),
                ("x4plus (General Photo)", "realesrgan-x4plus"),
                ("x4plus-anime (Anime Image)", "realesrgan-x4plus-anime")
            ],
            "waifu_model": [
                ("CUNet (Default)", "cunet"),
                ("Anime Style Art RGB", "upconv_7_anime_style_art_rgb"),
                ("Photo", "upconv_7_photo")
            ],
            "anime4k_model": [
                ("ACNet-GAN (Best Quality)", "acnet-gan"),
                ("ACNet (Standard)", "acnet")
            ],
            "anime4k_processor": [
                ("CUDA (NVIDIA GPU)", "cuda"),
                ("OpenCL (Universal)", "opencl"),
                ("CPU", "cpu")
            ],
            "compare_sources": ["Original", "Universal Enhance", "Image Repair", "Wallpaper Maker", "Ultra Soft", "Fast Upscale"],
        }
    }
}


# 加载语言配置
load_lang_config()
