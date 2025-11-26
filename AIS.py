import os
import subprocess
import sys
import shutil
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()
OUTPUT_DIR = BASE_DIR / "输出"

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir()

CUGAN_DIR = BASE_DIR / "realcugan-ncnn-vulkan-20220728-windows"
CUGAN_EXE = CUGAN_DIR / "realcugan-ncnn-vulkan.exe"
CUGAN_MODEL_SE = CUGAN_DIR / "models-se"
CUGAN_MODEL_PRO = CUGAN_DIR / "models-pro"

ESRGAN_DIR = BASE_DIR / "realesrgan-ncnn-vulkan-20220424-windows"
ESRGAN_EXE = ESRGAN_DIR / "realesrgan-ncnn-vulkan.exe"
ESRGAN_MODEL = ESRGAN_DIR / "models"

WAIFU_DIR = BASE_DIR / "waifu2x-ncnn-vulkan-20250915-windows"
WAIFU_EXE = WAIFU_DIR / "waifu2x-ncnn-vulkan.exe"
WAIFU_MODEL = WAIFU_DIR / "models-cunet"

def clean_path(user_input):
    return Path(user_input.strip().strip('"').strip("'"))

def run_cmd(cmd, cwd):
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError:
        print(f"\n[错误] 执行失败")
        return False

def get_unique_output_path(filename):
    target = OUTPUT_DIR / filename
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    counter = 1
    while target.exists():
        target = OUTPUT_DIR / f"{stem}_{counter}{suffix}"
        counter += 1
    return target

def select_file_gui():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="请选择图片文件",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All files", "*.*")]
    )
    root.destroy()
    return Path(file_path) if file_path else None

def preset_universal_2x(input_path):
    print(">> 正在执行: Real-CUGAN Pro (2x, 保守降噪)")
    out_name = f"{input_path.stem}_Pro_2x.png"
    out_path = get_unique_output_path(out_name)
    cmd = [
        str(CUGAN_EXE), "-i", str(input_path), "-o", str(out_path),
        "-s", "2", "-n", "0", "-m", str(CUGAN_MODEL_PRO)
    ]
    return cmd, CUGAN_DIR

def preset_repair_4x(input_path):
    print(">> 正在执行: Real-ESRGAN (4x, 烂图修复)")
    out_name = f"{input_path.stem}_ESRGAN_4x.png"
    out_path = get_unique_output_path(out_name)
    cmd = [
        str(ESRGAN_EXE), "-i", str(input_path), "-o", str(out_path),
        "-s", "4", "-n", "realesrgan-x4plus-anime", "-m", str(ESRGAN_MODEL)
    ]
    return cmd, ESRGAN_DIR

def preset_wallpaper_4x(input_path):
    print(">> 正在执行: Real-CUGAN SE (4x, 无降噪)")
    out_name = f"{input_path.stem}_CUGAN_4x_NoDenoise.png"
    out_path = get_unique_output_path(out_name)
    cmd = [
        str(CUGAN_EXE), "-i", str(input_path), "-o", str(out_path),
        "-s", "4", "-n", "-1", "-m", str(CUGAN_MODEL_SE)
    ]
    return cmd, CUGAN_DIR

def preset_soft_2x(input_path):
    print(">> 正在执行: Waifu2x Cunet (2x, 强力柔化)")
    out_name = f"{input_path.stem}_Waifu_2x_L3.png"
    out_path = get_unique_output_path(out_name)
    cmd = [
        str(WAIFU_EXE), "-i", str(input_path), "-o", str(out_path),
        "-s", "2", "-n", "3", "-m", str(WAIFU_MODEL)
    ]
    return cmd, WAIFU_DIR

def custom_mode(input_path):
    print("\n[自定义模式]")
    print("1. Real-CUGAN")
    print("2. Real-ESRGAN")
    print("3. Waifu2x")
    
    eng = input("请选择引擎 (1-3): ").strip()
    
    cmd = []
    work_dir = BASE_DIR
    
    if eng == '1':
        work_dir = CUGAN_DIR
        print("模型版本: 1.SE (标准版)  2.Pro (专业版)")
        m_type = input("请选择 (1/2): ").strip()
        model_dir = CUGAN_MODEL_PRO if m_type == '2' else CUGAN_MODEL_SE
        
        print("降噪强度: 1.无(-1)  2.保守(0)  3.强力(3)")
        n_in = input("请选择 (默认 2): ").strip()
        n_map = {'1': '-1', '2': '0', '3': '3'}
        n_val = n_map.get(n_in, '0')
        
        scale = input("放大倍率 (2, 3, 4): ").strip()
        if scale not in ['2', '3', '4']: scale = '2'
        
        out_name = f"{input_path.stem}_Custom_Cugan_{scale}x_n{n_val}.png"
        out_path = get_unique_output_path(out_name)
        
        cmd = [
            str(CUGAN_EXE), "-i", str(input_path), "-o", str(out_path),
            "-s", scale, "-n", n_val, "-m", str(model_dir)
        ]
        
    elif eng == '2':
        work_dir = ESRGAN_DIR
        print("Real-ESRGAN 默认固定为 4倍 放大。")
        out_name = f"{input_path.stem}_Custom_Esrgan_4x.png"
        out_path = get_unique_output_path(out_name)
        
        cmd = [
            str(ESRGAN_EXE), "-i", str(input_path), "-o", str(out_path),
            "-s", "4", "-n", "realesrgan-x4plus-anime", "-m", str(ESRGAN_MODEL)
        ]
        
    elif eng == '3':
        work_dir = WAIFU_DIR
        print("降噪等级 (0-3):")
        n_val = input("请输入 (默认 1): ").strip()
        if n_val not in ['0','1','2','3']: n_val = '1'
        
        scale = input("放大倍率 (1, 2, 4, ...): ").strip()
        if not scale.isdigit(): scale = '2'
        
        out_name = f"{input_path.stem}_Custom_Waifu_{scale}x_n{n_val}.png"
        out_path = get_unique_output_path(out_name)
        
        cmd = [
            str(WAIFU_EXE), "-i", str(input_path), "-o", str(out_path),
            "-s", scale, "-n", n_val, "-m", str(WAIFU_MODEL)
        ]
    
    else:
        return None, None
        
    return cmd, work_dir

def main():
    print("\n==============================================")
    print("图像超分 By SENyiAi")
    print("本程序无需管理员权限 请勿以管理员身份运行")
    print("==============================================\n")

    if not CUGAN_EXE.exists(): print("[警告] 未检测到 Real-CUGAN")
    
    while True:
        print("\n[请选择输入方式]")
        print("1. 手动输入路径 / 拖入图片")
        print("2. 调用系统文件选择器")
        print("q. 退出程序")
        
        method = input("请输入 (1/2/q): ").strip().lower()
        
        input_path = None

        if method == 'q' or method == 'exit':
            break
            
        if method == '2':
            input_path = select_file_gui()
            if not input_path:
                print("[提示] 未选择文件")
                continue
            print(f"[已选择] {input_path}")
            
        elif method == '1':
            print("\n请拖入图片:")
            user_in = input(">>> ")
            if user_in.lower() in ['q', 'exit']: break
            if not user_in: continue
            input_path = clean_path(user_in)
        
        else:
            continue

        if not input_path.exists():
            print("[错误] 文件未找到")
            continue
        if input_path.is_dir():
            print("[错误] 请拖入文件，不支持文件夹")
            continue

        print("\n[预设模式]")
        print("  1. 通用增强 (CUGAN Pro 2x)")
        print("  2. 烂图修复 (ESRGAN 4x)")
        print("  3. 壁纸制作 (CUGAN SE 4x)")
        print("  4. 极致柔化 (Waifu2x 2x)")
        print("  5. 我全都要 (自动执行全部预设)")
        print("  6. 自定义参数")
        
        choice = input("请选择 (默认 1): ").strip()
        
        tasks = []
        
        if choice == '5':
            tasks.append(preset_universal_2x(input_path))
            tasks.append(preset_repair_4x(input_path))
            tasks.append(preset_wallpaper_4x(input_path))
            tasks.append(preset_soft_2x(input_path))
        elif choice == '6':
            res = custom_mode(input_path)
            if res[0]: tasks.append(res)
        elif choice == '2':
            tasks.append(preset_repair_4x(input_path))
        elif choice == '3':
            tasks.append(preset_wallpaper_4x(input_path))
        elif choice == '4':
            tasks.append(preset_soft_2x(input_path))
        else:
            tasks.append(preset_universal_2x(input_path))

        print("---------------------------------------")
        for cmd, work_dir in tasks:
            if run_cmd(cmd, work_dir):
                out_file = Path(cmd[4]).name
                print(f"[成功] 已保存至: 输出/{out_file}")
        print("---------------------------------------")

if __name__ == "__main__":
    main()
