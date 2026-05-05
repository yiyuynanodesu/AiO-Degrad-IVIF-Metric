#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
矩形配置迁移工具（增强版）
功能：
- 从参考图像交互记录矩形（坐标、颜色、启用状态、放大位置）
- 支持自动提取矩形边框颜色（按 y）
- 支持自动识别放大角落位置（按 o）
- 将配置应用到新图像，保持所有矩形参数一致
- 新图像放大框不显示文字标签
"""

import cv2
import json
import os
import numpy as np
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

# ---------- 全局变量 ----------
rects = []
drawing = False
temp_point = None
current_mouse_pos = (0, 0)
current_rect_index = -1
default_color = (0, 255, 0)      # BGR 绿色

# ---------- 字体加载 ----------
def load_chinese_font():
    font_path = None
    if os.name == 'nt':
        font_path = 'C:/Windows/Fonts/simhei.ttf'
    elif os.name == 'posix':
        if os.path.exists('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'):
            font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
        elif os.path.exists('/System/Library/Fonts/PingFang.ttc'):
            font_path = '/System/Library/Fonts/PingFang.ttc'
    if font_path and os.path.exists(font_path):
        return ImageFont.truetype(font_path, 16)
    return ImageFont.load_default()

font_ch = load_chinese_font()

def draw_chinese_text(image, text, position, color=(255,255,255)):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font_ch, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ---------- 颜色自动提取 ----------
def get_rect_border_color(img, x1, y1, x2, y2):
    """从矩形边框上采样颜色，返回 BGR 元组（Python int）"""
    h, w = img.shape[:2]
    border_pixels = []
    # 上边
    for x in range(max(0, x1), min(w, x2+1)):
        if 0 <= y1 < h:
            border_pixels.append(img[y1, x])
    # 下边
    for x in range(max(0, x1), min(w, x2+1)):
        if 0 <= y2 < h:
            border_pixels.append(img[y2, x])
    # 左边（避免重复角点）
    for y in range(max(0, y1+1), min(h, y2)):
        if 0 <= x1 < w:
            border_pixels.append(img[y, x1])
    # 右边
    for y in range(max(0, y1+1), min(h, y2)):
        if 0 <= x2 < w:
            border_pixels.append(img[y, x2])
    if not border_pixels:
        return default_color
    # 量化颜色（16级）后取众数
    quantized = [(b//16, g//16, r//16) for b,g,r in border_pixels]
    most_common = Counter(quantized).most_common(1)[0][0]
    b = most_common[0] * 16 + 8
    g = most_common[1] * 16 + 8
    r = most_common[2] * 16 + 8
    return (int(b), int(g), int(r))

# ---------- 放大位置自动识别 ----------
def detect_zoom_position(img, rect_coords, zoom_factor):
    """根据参考图像中放大框的位置自动判断该矩形对应的放大角落"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = rect_coords
    orig_w = x2 - x1
    orig_h = y2 - y1
    zoom_w = int(orig_w * zoom_factor)
    zoom_h = int(orig_h * zoom_factor)
    corners = [
        (0, 0, zoom_w, zoom_h),                # 左上
        (w - zoom_w, 0, w, zoom_h),            # 右上
        (0, h - zoom_h, zoom_w, h),            # 左下
        (w - zoom_w, h - zoom_h, w, h)         # 右下
    ]
    best_match = 0
    max_score = 0
    for idx, (cx1, cy1, cx2, cy2) in enumerate(corners):
        if cx1 < 0 or cy1 < 0 or cx2 > w or cy2 > h:
            continue
        roi = img[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            continue
        edge_strength = 0
        if cy1+1 < h:
            edge_strength += np.mean(img[cy1, cx1:cx2])
        if cy2-1 >= 0:
            edge_strength += np.mean(img[cy2-1, cx1:cx2])
        if cx1+1 < w:
            edge_strength += np.mean(img[cy1:cy2, cx1])
        if cx2-1 >= 0:
            edge_strength += np.mean(img[cy1:cy2, cx2-1])
        if edge_strength > max_score:
            max_score = edge_strength
            best_match = idx
    return best_match

# ---------- 鼠标回调 ----------
def mouse_callback_record(event, x, y, flags, param):
    global drawing, temp_point, rects, current_rect_index, current_mouse_pos
    current_mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            drawing = True
            temp_point = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing and temp_point is not None:
            drawing = False
            new_rect = {
                "points": [temp_point, [x, y]],
                "enabled": True,
                "color": list(default_color),
                "zoom_position": 0
            }
            rects.append(new_rect)
            current_rect_index = len(rects) - 1
            print(f"矩形 {current_rect_index}: {temp_point} -> ({x},{y})")
            temp_point = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, r in enumerate(rects):
            x1, y1 = r["points"][0]
            x2, y2 = r["points"][1]
            x1, x2 = min(x1,x2), max(x1,x2)
            y1, y2 = min(y1,y2), max(y1,y2)
            if x1 <= x <= x2 and y1 <= y <= y2:
                rects.pop(i)
                print(f"删除矩形 {i}")
                if current_rect_index >= len(rects):
                    current_rect_index = len(rects)-1
                break

def draw_rects_on_image(img, rects, current_idx=-1):
    """在图像上绘制矩形框（安全颜色处理）"""
    for i, r in enumerate(rects):
        p1, p2 = r["points"]
        x1, y1 = p1
        x2, y2 = p2
        x1, x2 = min(x1,x2), max(x1,x2)
        y1, y2 = min(y1,y2), max(y1,y2)
        color_raw = r.get("color", default_color)
        try:
            color = tuple(int(c) for c in color_raw[:3])
        except (TypeError, IndexError, ValueError):
            color = default_color
        thickness = 3 if i == current_idx else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        status = "ON" if r["enabled"] else "OFF"
        status_color = (0,255,0) if r["enabled"] else (0,0,255)
        cv2.putText(img, f"{i}", (x2-20, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(img, status, (x2-40, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
    return img

def convert_to_serializable(obj):
    """递归转换 numpy 类型为 Python 原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def save_config_to_json(json_path, rects, zoom_factor):
    data = {"zoom_factor": zoom_factor, "rects": rects}
    data = convert_to_serializable(data)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"配置已保存至: {json_path}")

def load_config_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("zoom_factor", 2), data.get("rects", [])

def apply_rects_to_image(img, rects, zoom_factor):
    """应用矩形配置到新图像，不显示位置文字"""
    h, w = img.shape[:2]
    result = img.copy()
    enabled_rects = []
    for r in rects:
        p1, p2 = r["points"]
        x1, y1 = p1
        x2, y2 = p2
        x1, x2 = min(x1,x2), max(x1,x2)
        y1, y2 = min(y1,y2), max(y1,y2)
        color_raw = r.get("color", default_color)
        try:
            color = tuple(int(c) for c in color_raw[:3])
        except (TypeError, IndexError, ValueError):
            color = default_color
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        if r["enabled"]:
            enabled_rects.append({
                "coords": (x1, y1, x2, y2),
                "zoom_position": r.get("zoom_position", 0),
                "color": color
            })
    for er in enabled_rects:
        x1, y1, x2, y2 = er["coords"]
        orig_w = x2 - x1
        orig_h = y2 - y1
        if orig_w <= 0 or orig_h <= 0:
            continue
        zoom_w = int(orig_w * zoom_factor)
        zoom_h = int(orig_h * zoom_factor)
        if zoom_w > w or zoom_h > h:
            print(f"警告：放大尺寸 ({zoom_w}x{zoom_h}) 超过图像尺寸，跳过此矩形")
            continue
        roi = img[y1:y2, x1:x2].copy()
        if roi.size == 0:
            continue
        roi_zoom = cv2.resize(roi, (zoom_w, zoom_h))
        pos = er["zoom_position"]
        if pos == 0:      # 左上
            tx, ty = 0, 0
        elif pos == 1:    # 右上
            tx, ty = w - zoom_w, 0
        elif pos == 2:    # 左下
            tx, ty = 0, h - zoom_h
        else:             # 右下
            tx, ty = w - zoom_w, h - zoom_h
        result[ty:ty+zoom_h, tx:tx+zoom_w] = roi_zoom
        cv2.rectangle(result, (tx, ty), (tx+zoom_w, ty+zoom_h), er["color"], 2)
    return result

# ---------- 交互模式 ----------
def interactive_record_mode():
    global current_rect_index, drawing, temp_point, rects
    current_rect_index = -1
    drawing = False
    temp_point = None
    rects.clear()

    img_path = input("请输入参考图像路径（已处理过的图片）: ").strip()
    if not os.path.exists(img_path):
        print("文件不存在")
        return
    img = cv2.imread(img_path)
    if img is None:
        print("无法读取图像")
        return

    cv2.namedWindow("Record Rectangles", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Record Rectangles", mouse_callback_record)

    zoom_factor = 2
    print("\n=== 交互记录矩形模式 ===")
    print("操作说明:")
    print("  左键拖拽绘制矩形（起点→终点）")
    print("  右键点击某个矩形可删除它")
    print("  按键:")
    print("    '+' / '-' : 增加/减少全局放大倍数")
    print("    'e' : 切换当前选中矩形的启用/禁用状态")
    print("    'p' : 切换放大位置 (0:左上 1:右上 2:左下 3:右下)")
    print("    'y' : 从参考图像自动提取当前矩形的边框颜色 ★")
    print("    'o' : 自动识别当前矩形的放大位置（从角落检测）★")
    print("    'c' : 清除所有矩形")
    print("    's' : 保存配置到 JSON 文件")
    print("    'q' : 退出")
    print("==========================================\n")

    while True:
        disp = img.copy()
        if drawing and temp_point:
            cx, cy = current_mouse_pos
            cv2.rectangle(disp, tuple(temp_point), (cx, cy), (255,255,0), 2)
        draw_rects_on_image(disp, rects, current_rect_index)

        info = f"矩形数: {len(rects)} | 放大倍数: {zoom_factor}x | 当前选中: {current_rect_index}"
        disp = draw_chinese_text(disp, info, (10,30), color=(255,255,255))
        if 0 <= current_rect_index < len(rects):
            r = rects[current_rect_index]
            pos_names = ["左上角", "右上角", "左下角", "右下角"]
            stat = "启用" if r["enabled"] else "禁用"
            txt = f"矩形 {current_rect_index}: {stat} | 颜色 BGR{r['color']} | 放大位置: {pos_names[r['zoom_position']]}"
            disp = draw_chinese_text(disp, txt, (10,60), color=(0,255,255))

        cv2.imshow("Record Rectangles", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            if not rects:
                print("没有矩形可保存")
                continue
            json_path = input("保存为 JSON 文件路径（如 config.json）: ").strip()
            if json_path:
                save_config_to_json(json_path, rects, zoom_factor)
        elif key == ord('c'):
            rects.clear()
            current_rect_index = -1
            print("已清除所有矩形")
        elif key == ord('+') or key == ord('='):
            zoom_factor = min(5, zoom_factor+1)
            print(f"放大倍数 -> {zoom_factor}")
        elif key == ord('-') or key == ord('_'):
            zoom_factor = max(1, zoom_factor-1)
            print(f"放大倍数 -> {zoom_factor}")
        elif key == ord('e'):
            if 0 <= current_rect_index < len(rects):
                rects[current_rect_index]["enabled"] = not rects[current_rect_index]["enabled"]
                status = "启用" if rects[current_rect_index]["enabled"] else "禁用"
                print(f"矩形 {current_rect_index} 已{status}")
        elif key == ord('p'):
            if 0 <= current_rect_index < len(rects):
                new_pos = (rects[current_rect_index]["zoom_position"] + 1) % 4
                rects[current_rect_index]["zoom_position"] = new_pos
                pos_names = ["左上","右上","左下","右下"]
                print(f"矩形 {current_rect_index} 放大位置 -> {pos_names[new_pos]}")
        elif key == ord('y'):
            if 0 <= current_rect_index < len(rects):
                p1, p2 = rects[current_rect_index]["points"]
                x1, y1 = p1
                x2, y2 = p2
                x1, x2 = min(x1,x2), max(x1,x2)
                y1, y2 = min(y1,y2), max(y1,y2)
                color = get_rect_border_color(img, x1, y1, x2, y2)
                rects[current_rect_index]["color"] = list(color)
                print(f"矩形 {current_rect_index} 颜色自动提取为 BGR{color}")
        elif key == ord('o'):
            if 0 <= current_rect_index < len(rects):
                p1, p2 = rects[current_rect_index]["points"]
                x1, y1 = p1
                x2, y2 = p2
                x1, x2 = min(x1,x2), max(x1,x2)
                y1, y2 = min(y1,y2), max(y1,y2)
                pos = detect_zoom_position(img, (x1,y1,x2,y2), zoom_factor)
                rects[current_rect_index]["zoom_position"] = pos
                pos_names = ["左上","右上","左下","右下"]
                print(f"矩形 {current_rect_index} 放大位置自动识别为 {pos_names[pos]}")
        elif 48 <= key <= 57:
            idx = key - 48
            if idx < len(rects):
                current_rect_index = idx
                print(f"选中矩形 {idx}")

    cv2.destroyAllWindows()

# ---------- 应用配置模式 ----------
def apply_config_mode():
    json_path = input("请输入 JSON 配置文件路径: ").strip()
    if not os.path.exists(json_path):
        print("配置文件不存在")
        return
    zoom_factor, rects = load_config_from_json(json_path)
    print(f"加载配置: 放大倍数={zoom_factor}, 矩形数量={len(rects)}")
    img_path = input("请输入新图像路径: ").strip()
    if not os.path.exists(img_path):
        print("图像文件不存在")
        return
    img = cv2.imread(img_path)
    if img is None:
        print("无法读取图像")
        return
    result = apply_rects_to_image(img, rects, zoom_factor)
    out_path = input("保存结果图像路径（如 output.png）: ").strip()
    if out_path:
        cv2.imwrite(out_path, result)
        print(f"已保存至 {out_path}")
    else:
        print("未保存")
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------- 主程序 ----------
if __name__ == "__main__":
    print("===== 矩形配置迁移工具（增强版） =====")
    print("模式 1 - 交互记录矩形（支持自动提取颜色/位置）")
    print("模式 2 - 直接应用 JSON 配置到新图像")
    mode = input("输入 1 或 2: ").strip()
    if mode == '1':
        interactive_record_mode()
    elif mode == '2':
        apply_config_mode()
    else:
        print("无效输入")