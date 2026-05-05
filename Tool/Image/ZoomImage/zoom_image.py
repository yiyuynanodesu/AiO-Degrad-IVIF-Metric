import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 全局变量
zoom_fac = 2
rects = []  # 存储所有矩形：[{"points": [[x1,y1], [x2,y2]], "enabled": True/False, "color": (B,G,R), "zoom_position": 0-3}]
selected_position = 0  # 当前选择的放大框位置
current_rect_index = -1  # 当前正在编辑的矩形索引
drawing = False  # 是否正在绘制新矩形
temp_point = None  # 临时存储第一个点
current_mouse_pos = (0, 0)  # 当前鼠标位置
default_color = (0, 255, 0)  # 默认绿色
color_b, color_g, color_r = default_color  # 当前颜色滑块的值

# 中文字体路径，使用系统字体或指定字体文件
try:
    if os.name == 'nt':  # Windows
        font_path = 'C:/Windows/Fonts/simhei.ttf'
    elif os.name == 'posix':  # Linux或macOS
        if os.path.exists('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'):
            font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
        elif os.path.exists('/System/Library/Fonts/PingFang.ttc'):
            font_path = '/System/Library/Fonts/PingFang.ttc'
        else:
            font_path = None
    else:
        font_path = None
        
    if font_path and os.path.exists(font_path):
        font_small = ImageFont.truetype(font_path, 14)
        font_medium = ImageFont.truetype(font_path, 18)
        font_large = ImageFont.truetype(font_path, 22)
    else:
        font_small = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_large = ImageFont.load_default()
        print("警告：未找到中文字体，将使用默认字体显示英文")
except Exception as e:
    print(f"加载字体失败: {e}")
    font_small = ImageFont.load_default()
    font_medium = ImageFont.load_default()
    font_large = ImageFont.load_default()

def draw_chinese_text(image, text, position, font_size='medium', color=(255, 255, 255)):
    """
    在图像上绘制中文文本
    """
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    if font_size == 'small':
        font = font_small
    elif font_size == 'large':
        font = font_large
    else:
        font = font_medium
    
    draw.text(position, text, font=font, fill=color)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_local_zoom_callback(value):
    global zoom_fac
    zoom_fac = value
    print("Zoom factor: {}".format(zoom_fac))

def mouse_callback(event, x1, y1, flags, userdata):
    global rects, drawing, temp_point, current_rect_index, current_mouse_pos, color_b, color_g, color_r
    
    current_mouse_pos = (x1, y1)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            drawing = True
            temp_point = [x1, y1]
            print(f"开始绘制新矩形，起点: ({x1}, {y1})")
        else:
            drawing = False
            new_rect = {
                "points": [temp_point, [x1, y1]],
                "enabled": True,
                "color": (color_b, color_g, color_r),
                "zoom_position": selected_position  # 使用当前全局位置作为默认
            }
            rects.append(new_rect)
            current_rect_index = len(rects) - 1
            print(f"完成绘制矩形 {current_rect_index}: {temp_point} -> ({x1}, {y1})")
            temp_point = None
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, rect in enumerate(rects):
            x1_rect, y1_rect = rect["points"][0]
            x2_rect, y2_rect = rect["points"][1]
            min_x = min(x1_rect, x2_rect)
            max_x = max(x1_rect, x2_rect)
            min_y = min(y1_rect, y2_rect)
            max_y = max(y1_rect, y2_rect)
            
            if min_x <= x1 <= max_x and min_y <= y1 <= max_y:
                current_rect_index = i
                rects[i]["enabled"] = not rects[i]["enabled"]
                status = "启用放大" if rects[i]["enabled"] else "只框选"
                print(f"选择矩形 {i}，状态: {status}")
                
                if "color" in rects[i]:
                    color_b, color_g, color_r = rects[i]["color"]
                    cv2.setTrackbarPos("Color B", "MultiRect", color_b)
                    cv2.setTrackbarPos("Color G", "MultiRect", color_g)
                    cv2.setTrackbarPos("Color R", "MultiRect", color_r)
                
                # 更新位置滑块到当前选中矩形的位置
                if "zoom_position" in rects[i]:
                    cv2.setTrackbarPos("Position\n0:TL 1:TR\n2:BL 3:BR", "MultiRect", rects[i]["zoom_position"])
                break

def position_callback(value):
    global selected_position
    selected_position = value
    # 更新当前选中矩形的放大位置
    if current_rect_index >= 0 and current_rect_index < len(rects):
        rects[current_rect_index]["zoom_position"] = value
    positions = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
    print("当前放大位置: {}".format(positions[value]))

def color_b_callback(value):
    global color_b
    color_b = value
    if current_rect_index >= 0 and current_rect_index < len(rects):
        rects[current_rect_index]["color"] = (color_b, color_g, color_r)

def color_g_callback(value):
    global color_g
    color_g = value
    if current_rect_index >= 0 and current_rect_index < len(rects):
        rects[current_rect_index]["color"] = (color_b, color_g, color_r)

def color_r_callback(value):
    global color_r
    color_r = value
    if current_rect_index >= 0 and current_rect_index < len(rects):
        rects[current_rect_index]["color"] = (color_b, color_g, color_r)

def draw_all_rects(img):
    global rects, zoom_fac, current_rect_index
    
    if not rects:
        return
        
    h, w = img.shape[0], img.shape[1]
    enabled_rects = []
    
    for i, rect in enumerate(rects):
        points = rect["points"]
        enabled = rect["enabled"]
        rect_color = rect.get("color", default_color)
        zoom_position = rect.get("zoom_position", 0)  # 获取矩形的放大位置
        
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        orig_w = x2 - x1
        orig_h = y2 - y1
        
        if orig_w <= 0 or orig_h <= 0:
            continue
        
        if i == current_rect_index:
            border_color = (0, 255, 255)
        else:
            border_color = rect_color
        
        thickness = 3 if i == current_rect_index else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, thickness)
        
        cv2.putText(img, f"{i}", (x2-20, y1+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 2)
        
        status = "ON" if enabled else "OFF"
        status_color = (0, 255, 0) if enabled else (0, 0, 255)
        cv2.putText(img, status, (x2-40, y2-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        if enabled:
            enabled_rects.append({
                "index": i,
                "coords": (x1, y1, x2, y2),
                "size": (orig_w, orig_h),
                "color": rect_color,
                "zoom_position": zoom_position  # 传递放大位置
            })
    
    # 绘制所有启用的放大区域
    for i, rect_info in enumerate(enabled_rects):
        x1, y1, x2, y2 = rect_info["coords"]
        orig_w, orig_h = rect_info["size"]
        rect_color = rect_info["color"]
        zoom_position = rect_info["zoom_position"]  # 使用矩形自己的放大位置
        
        zoom_w = int(orig_w * zoom_fac)
        zoom_h = int(orig_h * zoom_fac)
        
        if zoom_w > w or zoom_h > h:
            print(f"矩形 {rect_info['index']} 放大尺寸超出图像范围，请减小放大系数")
            continue
        
        zoom_in_img = img[y1:y2, x1:x2].copy()
        if zoom_in_img.size == 0:
            continue
            
        zoom_in_img = cv2.resize(zoom_in_img, (zoom_w, zoom_h))
        
        # 使用紧贴边界的放置方式，不移除任何垂直偏移
        pos_index = zoom_position  # 使用矩形自己的放大位置
        
        if pos_index == 0:  # 左上角，紧贴左上边界
            target_y1 = 0  # 紧贴顶部
            target_y2 = target_y1 + zoom_h
            target_x1 = 0  # 紧贴左侧
            target_x2 = target_x1 + zoom_w
                
        elif pos_index == 1:  # 右上角，紧贴右上边界
            target_y1 = 0  # 紧贴顶部
            target_y2 = target_y1 + zoom_h
            target_x1 = w - zoom_w  # 紧贴右侧
            target_x2 = w
                
        elif pos_index == 2:  # 左下角，紧贴左下边界
            target_y1 = h - zoom_h  # 紧贴底部
            target_y2 = h
            target_x1 = 0  # 紧贴左侧
            target_x2 = target_x1 + zoom_w
                
        else:  # 右下角，紧贴右下边界
            target_y1 = h - zoom_h  # 紧贴底部
            target_y2 = h
            target_x1 = w - zoom_w  # 紧贴右侧
            target_x2 = w
        
        # 注意：如果多个放大框放在同一位置，它们将完全重叠
        # 用户需要自己调整它们的位置避免重叠
        
        img[target_y1:target_y2, target_x1:target_x2] = zoom_in_img
        cv2.rectangle(img, (target_x1, target_y1), (target_x2, target_y2), rect_color, 2)
        
        # 显示矩形编号和位置信息
        pos_names = ["TL", "TR", "BL", "BR"]
        info_text = f"[{rect_info['index']}] {pos_names[pos_index]}"
        cv2.putText(img, info_text, (target_x1+5, target_y1+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)

if __name__ == '__main__':
    ws_path = os.path.abspath(".")
    data_path = input('请输入要操作的图片文件夹路径(默认当前目录): ')
    save_path = data_path.replace('VisData', 'VisOutput')
    
    if not os.path.exists(data_path):
        raise Exception("Data path not exist!")
    
    img_file_list = [f for f in os.listdir(data_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not img_file_list:
        raise Exception("Data folder has no image files.")
    
    cv2.namedWindow("MultiRect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MultiRect", 1000, 700)
    
    cv2.setMouseCallback("MultiRect", mouse_callback)
    
    cv2.createTrackbar("ZoomFactor", "MultiRect", 2, 5, draw_local_zoom_callback)
    cv2.createTrackbar("Position\n0:TL 1:TR\n2:BL 3:BR", "MultiRect", 0, 3, position_callback)
    cv2.createTrackbar("Color B", "MultiRect", color_b, 255, color_b_callback)
    cv2.createTrackbar("Color G", "MultiRect", color_g, 255, color_g_callback)
    cv2.createTrackbar("Color R", "MultiRect", color_r, 255, color_r_callback)
    
    output_path = os.path.join(ws_path, save_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    print("=" * 60)
    print("多矩形框选与放大工具")
    print("使用说明:")
    print("1. 左键点击两次定义一个矩形区域（第一次起点，第二次终点）")
    print("2. 右键点击矩形区域可以切换其放大状态（ON/OFF）")
    print("3. ON（绿色）= 框选并放大，OFF（红色）= 只框选不放大")
    print("4. 当前选中的矩形会显示为黄色边框")
    print("5. 颜色滑块只影响当前选中矩形（右键选中后调整）")
    print("6. 位置滑块只影响当前选中矩形的放大位置")
    print("7. 放大框紧贴图像边界，如果多个放大框在同一位置会重叠")
    print("   请调整不同放大框的位置以避免重叠")
    print("键盘快捷键:")
    print("  's' - 保存当前图片")
    print("  'n' - 下一张图片")
    print("  'p' - 上一张图片")
    print("  'c' - 清除当前图片的所有矩形")
    print("  'd' - 删除当前选中的矩形")
    print("  'a' - 启用/禁用所有矩形")
    print("  'e' - 启用所有矩形")
    print("  'x' - 禁用所有矩形")
    print("  'r' - 重置当前选中矩形颜色为默认")
    print("  'l' - 重置当前选中矩形放大位置为默认(左上角)")
    print("  ESC - 退出")
    print("=" * 60)
    
    current_img_idx = 0
    img = cv2.imread(os.path.join(data_path, img_file_list[current_img_idx]))
    
    while True:
        display_img = img.copy()
        
        if drawing and temp_point:
            current_x, current_y = current_mouse_pos
            cv2.rectangle(display_img, 
                         (temp_point[0], temp_point[1]), 
                         (current_x, current_y), 
                         (255, 255, 0), 2)
        
        draw_all_rects(display_img)
        
        enabled_count = sum(1 for r in rects if r["enabled"])
        info_text = f"图片: {current_img_idx+1}/{len(img_file_list)} | 矩形: {len(rects)} | 启用放大: {enabled_count} | 放大倍数: {zoom_fac}x"
        display_img = draw_chinese_text(display_img, info_text, (10, 30), 
                                       font_size='medium', color=(255, 255, 255))
        
        if current_rect_index >= 0 and current_rect_index < len(rects):
            rect_info = rects[current_rect_index]
            status = "框选并放大" if rect_info["enabled"] else "只框选"
            color_info = rect_info.get("color", default_color)
            zoom_pos = rect_info.get("zoom_position", 0)
            pos_names = ["左上角", "右上角", "左下角", "右下角"]
            point_text = f"选中矩形 {current_rect_index}: {status} | 颜色: {color_info} | 放大位置: {pos_names[zoom_pos]}"
            display_img = draw_chinese_text(display_img, point_text, (10, 60), 
                                           font_size='small', color=(0, 255, 255))
        
        hint_text = "左键: 绘制矩形 | 右键: 切换状态/选择 | 按's'保存 | 按'h'显示帮助"
        display_img = draw_chinese_text(display_img, hint_text, (10, display_img.shape[0]-25), 
                                       font_size='small', color=(200, 200, 100))
        
        cv2.imshow("MultiRect", display_img)
        k = cv2.waitKey(1) & 0xFF
        
        if k == 27:
            break
        elif k == ord('s'):
            result_img = img.copy()
            draw_all_rects(result_img)
            output_file = os.path.join(output_path, 
                                      f"{img_file_list[current_img_idx].split('.')[0]}_processed.png")
            cv2.imwrite(output_file, result_img)
            print(f"保存到: {output_file}")
            
        elif k == ord('n'):
            current_img_idx = (current_img_idx + 1) % len(img_file_list)
            img = cv2.imread(os.path.join(data_path, img_file_list[current_img_idx]))
            rects.clear()
            current_rect_index = -1
            drawing = False
            print(f"加载图片: {img_file_list[current_img_idx]}")
            
        elif k == ord('p'):
            current_img_idx = (current_img_idx - 1) % len(img_file_list)
            img = cv2.imread(os.path.join(data_path, img_file_list[current_img_idx]))
            rects.clear()
            current_rect_index = -1
            drawing = False
            print(f"加载图片: {img_file_list[current_img_idx]}")
            
        elif k == ord('c'):
            rects.clear()
            current_rect_index = -1
            drawing = False
            color_b, color_g, color_r = default_color
            cv2.setTrackbarPos("Color B", "MultiRect", color_b)
            cv2.setTrackbarPos("Color G", "MultiRect", color_g)
            cv2.setTrackbarPos("Color R", "MultiRect", color_r)
            print("清除所有矩形")
            
        elif k == ord('d'):
            if 0 <= current_rect_index < len(rects):
                removed = rects.pop(current_rect_index)
                print(f"删除矩形 {current_rect_index}")
                current_rect_index = min(current_rect_index, len(rects) - 1)
                if current_rect_index >= 0:
                    color_b, color_g, color_r = rects[current_rect_index]["color"]
                    cv2.setTrackbarPos("Color B", "MultiRect", color_b)
                    cv2.setTrackbarPos("Color G", "MultiRect", color_g)
                    cv2.setTrackbarPos("Color R", "MultiRect", color_r)
                    zoom_pos = rects[current_rect_index].get("zoom_position", 0)
                    cv2.setTrackbarPos("Position\n0:TL 1:TR\n2:BL 3:BR", "MultiRect", zoom_pos)
                
        elif k == ord('a'):
            if rects:
                all_enabled = all(r["enabled"] for r in rects)
                new_state = not all_enabled
                for r in rects:
                    r["enabled"] = new_state
                status = "启用放大" if new_state else "只框选"
                print(f"{status}所有矩形")
                
        elif k == ord('e'):
            for r in rects:
                r["enabled"] = True
            print("所有矩形启用放大功能")
            
        elif k == ord('x'):
            for r in rects:
                r["enabled"] = False
            print("所有矩形设为只框选不放大")
            
        elif k == ord('r'):
            if 0 <= current_rect_index < len(rects):
                rects[current_rect_index]["color"] = default_color
                color_b, color_g, color_r = default_color
                cv2.setTrackbarPos("Color B", "MultiRect", color_b)
                cv2.setTrackbarPos("Color G", "MultiRect", color_g)
                cv2.setTrackbarPos("Color R", "MultiRect", color_r)
                print(f"重置矩形 {current_rect_index} 颜色为默认")
        
        elif k == ord('l'):
            if 0 <= current_rect_index < len(rects):
                rects[current_rect_index]["zoom_position"] = 0
                cv2.setTrackbarPos("Position\n0:TL 1:TR\n2:BL 3:BR", "MultiRect", 0)
                print(f"重置矩形 {current_rect_index} 放大位置为左上角")
            
        elif k == ord('h'):
            print("\n帮助信息:")
            print("1. 左键点击两次定义一个矩形")
            print("2. 右键点击矩形切换其放大状态")
            print("3. 绿色矩形 = 框选并放大，红色矩形 = 只框选不放大")
            print("4. 颜色滑块只影响当前选中矩形")
            print("5. 位置滑块只影响当前选中矩形的放大位置")
            print("6. 放大框紧贴图像边界，如果多个放大框在同一位置会重叠")
            print("   请调整不同放大框的位置以避免重叠")
            
        elif k == ord('+') or k == ord('='):
            if zoom_fac < 5:
                zoom_fac += 1
                cv2.setTrackbarPos("ZoomFactor", "MultiRect", zoom_fac)
                
        elif k == ord('-') or k == ord('_'):
            if zoom_fac > 1:
                zoom_fac -= 1
                cv2.setTrackbarPos("ZoomFactor", "MultiRect", zoom_fac)
                
        elif 48 <= k <= 57:
            idx = k - 48
            if idx < len(rects):
                current_rect_index = idx
                if "color" in rects[current_rect_index]:
                    color_b, color_g, color_r = rects[current_rect_index]["color"]
                    cv2.setTrackbarPos("Color B", "MultiRect", color_b)
                    cv2.setTrackbarPos("Color G", "MultiRect", color_g)
                    cv2.setTrackbarPos("Color R", "MultiRect", color_r)
                zoom_pos = rects[current_rect_index].get("zoom_position", 0)
                cv2.setTrackbarPos("Position\n0:TL 1:TR\n2:BL 3:BR", "MultiRect", zoom_pos)
                print(f"选择矩形 {idx}")
    
    # 批量处理所有图片
    print("\n开始批量处理所有图片...")
    for i, image_file in enumerate(img_file_list):
        current_img = cv2.imread(os.path.join(data_path, image_file))
        if current_img is None:
            print(f"无法加载图片: {image_file}")
            continue
        
        processed_img = current_img.copy()
        h, w = processed_img.shape[:2]
        
        if rects:
            enabled_rects = []
            
            for idx, rect in enumerate(rects):
                points = rect["points"]
                enabled = rect["enabled"]
                rect_color = rect.get("color", default_color)
                zoom_position = rect.get("zoom_position", 0)
                
                x1, y1 = points[0]
                x2, y2 = points[1]
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                orig_w = x2 - x1
                orig_h = y2 - y1
                
                if orig_w > 0 and orig_h > 0:
                    cv2.rectangle(processed_img, (x1, y1), (x2, y2), rect_color, 2)
                    
                    if enabled:
                        enabled_rects.append({
                            "index": idx,
                            "coords": (x1, y1, x2, y2),
                            "size": (orig_w, orig_h),
                            "color": rect_color,
                            "zoom_position": zoom_position
                        })
        
            for i_zoom, rect_info in enumerate(enabled_rects):
                x1, y1, x2, y2 = rect_info["coords"]
                orig_w, orig_h = rect_info["size"]
                rect_color = rect_info["color"]
                zoom_position = rect_info["zoom_position"]
                
                zoom_w = int(orig_w * zoom_fac)
                zoom_h = int(orig_h * zoom_fac)
                
                if zoom_w <= w and zoom_h <= h:
                    zoom_in_img = current_img[y1:y2, x1:x2].copy()
                    if zoom_in_img.size > 0:
                        zoom_in_img = cv2.resize(zoom_in_img, (zoom_w, zoom_h))
                        
                        # 紧贴边界的放置方式，不移除任何垂直偏移
                        pos_index = zoom_position
                        
                        if pos_index == 0:  # 左上角
                            target_y1 = 0
                            target_y2 = target_y1 + zoom_h
                            target_x1 = 0
                            target_x2 = target_x1 + zoom_w
                            
                        elif pos_index == 1:  # 右上角
                            target_y1 = 0
                            target_y2 = target_y1 + zoom_h
                            target_x1 = w - zoom_w
                            target_x2 = w
                            
                        elif pos_index == 2:  # 左下角
                            target_y1 = h - zoom_h
                            target_y2 = h
                            target_x1 = 0
                            target_x2 = target_x1 + zoom_w
                            
                        else:  # 右下角
                            target_y1 = h - zoom_h
                            target_y2 = h
                            target_x1 = w - zoom_w
                            target_x2 = w
                        
                        # 注意：如果多个放大框放在同一位置，它们将完全重叠
                        
                        processed_img[target_y1:target_y2, target_x1:target_x2] = zoom_in_img
                        cv2.rectangle(processed_img, (target_x1, target_y1), (target_x2, target_y2), rect_color, 2)
    
        output_file = os.path.join(output_path, 
                                  f"{image_file.split('.')[0]}_batch.png")
        cv2.imwrite(output_file, processed_img)
        print(f"处理进度 {i+1}/{len(img_file_list)}: {image_file}")
    
    print("批量处理完成！")
    cv2.destroyAllWindows()