import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from datetime import datetime

class LocalMagnificationTool:
    """多图实时局部放大对比工具 - 可独立调整裁剪区域宽高，放大图自动适配比例"""
    def __init__(self, root):
        self.root = root
        self.root.title("局部放大图对比工具 - 实时放大/点击保存")
        self.root.geometry("1400x800")

        # 数据存储
        self.images = []           # 原始PIL图像列表
        self.image_names = []      # 图像名称列表
        self.display_photos = []   # 用于显示的PhotoImage列表
        self.canvas_list = []      # 每个图像对应的Canvas
        self.zoom_labels = []      # 每个图像对应的放大图Label
        self.rect_ids = []         # 每个Canvas上的矩形框ID
        self.scale_factors = []    # 缩放因子列表 (scale_x, scale_y) 原始->显示

        # 当前鼠标位置对应的原图坐标
        self.current_orig_x = None
        self.current_orig_y = None
        self.last_valid_x = None
        self.last_valid_y = None

        # 可调节参数
        self.crop_width = tk.IntVar(value=96)    # 裁剪区域宽度(原图像素)
        self.crop_height = tk.IntVar(value=96)   # 裁剪区域高度(原图像素)
        self.zoom_max_size = tk.IntVar(value=200) # 放大图显示的最大边长(像素)，保持比例

        # 保存结果文件夹
        self.save_dir = "results"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.setup_ui()

    def setup_ui(self):
        """构建界面"""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(control_frame, text="打开图片", command=self.load_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="清空所有", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="重置默认", command=self.reset_defaults).pack(side=tk.LEFT, padx=5)

        # 裁剪区域大小设置(独立宽高)
        ttk.Label(control_frame, text="裁剪宽度(px):").pack(side=tk.LEFT, padx=(10,2))
        crop_w_spin = ttk.Spinbox(control_frame, from_=32, to=512, increment=8, textvariable=self.crop_width, width=6)
        crop_w_spin.pack(side=tk.LEFT, padx=2)
        ttk.Label(control_frame, text="裁剪高度(px):").pack(side=tk.LEFT, padx=(10,2))
        crop_h_spin = ttk.Spinbox(control_frame, from_=32, to=512, increment=8, textvariable=self.crop_height, width=6)
        crop_h_spin.pack(side=tk.LEFT, padx=2)

        # 放大图显示最大边长
        ttk.Label(control_frame, text="放大图最大边长(px):").pack(side=tk.LEFT, padx=(10,2))
        zoom_max_spin = ttk.Spinbox(control_frame, from_=64, to=400, increment=8, textvariable=self.zoom_max_size, width=6)
        zoom_max_spin.pack(side=tk.LEFT, padx=2)

        ttk.Button(control_frame, text="应用尺寸", command=self.apply_size_change).pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="提示: 移动鼠标查看局部放大 | 点击左键保存当前区域").pack(side=tk.RIGHT, padx=10)

        # 图片显示区域(带滚动条)
        canvas_container = ttk.Frame(self.root)
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.scroll_canvas = tk.Canvas(canvas_container, borderwidth=0, highlightthickness=0)
        scrollbar_y = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        scrollbar_x = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.scroll_canvas.xview)
        self.scroll_canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.images_frame = ttk.Frame(self.scroll_canvas)
        self.scroll_canvas.create_window((0, 0), window=self.images_frame, anchor=tk.NW)
        self.images_frame.bind("<Configure>", self._on_frame_configure)

    def _on_frame_configure(self, event=None):
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def load_images(self):
        file_paths = filedialog.askopenfilenames(
            title="选择对比图片(建议相同分辨率)",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif")]
        )
        if not file_paths:
            return

        self.clear_all()

        for path in file_paths:
            try:
                img = Image.open(path).convert("RGB")
                self.images.append(img)
                name = os.path.basename(path).split('.')[0]
                self.image_names.append(name)
            except Exception as e:
                print(f"加载失败 {path}: {e}")

        if not self.images:
            messagebox.showerror("错误", "没有有效图片可加载")
            return

        self.display_height = 300
        self.display_widths = []
        self.display_photos.clear()
        self.canvas_list.clear()
        self.zoom_labels.clear()
        self.rect_ids.clear()
        self.scale_factors.clear()

        for idx, img in enumerate(self.images):
            orig_w, orig_h = img.size
            ratio = self.display_height / orig_h
            disp_w = int(orig_w * ratio)
            self.display_widths.append(disp_w)
            disp_img = img.resize((disp_w, self.display_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(disp_img)
            self.display_photos.append(photo)

            scale_x = orig_w / disp_w
            scale_y = orig_h / self.display_height
            self.scale_factors.append((scale_x, scale_y))

            frame = ttk.Frame(self.images_frame, relief=tk.RIDGE, borderwidth=2)
            frame.pack(side=tk.LEFT, padx=8, pady=8, fill=tk.Y)

            title_label = ttk.Label(frame, text=self.image_names[idx], font=("Arial", 10, "bold"))
            title_label.pack(pady=(2,0))

            canvas = tk.Canvas(frame, width=disp_w, height=self.display_height, bg='gray')
            canvas.pack(padx=2, pady=2)
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo

            rect_id = canvas.create_rectangle(0, 0, 0, 0, outline='red', width=2, tags="rect")
            self.rect_ids.append(rect_id)

            zoom_label = ttk.Label(frame, text="局部放大", relief=tk.SUNKEN, background="#f0f0f0")
            zoom_label.pack(pady=(4,2), padx=2, fill=tk.X)
            self.zoom_labels.append(zoom_label)

            canvas.bind("<Motion>", lambda e, idx=idx: self.on_mouse_move(e, idx))
            canvas.bind("<Leave>", lambda e, idx=idx: self.on_mouse_leave(idx))
            canvas.bind("<Button-1>", lambda e, idx=idx: self.on_mouse_click(idx))

            self.canvas_list.append(canvas)

        self.images_frame.update_idletasks()
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def clear_all(self):
        self.images.clear()
        self.image_names.clear()
        self.display_photos.clear()
        self.scale_factors.clear()
        self.current_orig_x = None
        self.current_orig_y = None
        self.last_valid_x = None
        self.last_valid_y = None

        for widget in self.images_frame.winfo_children():
            widget.destroy()
        self.canvas_list.clear()
        self.zoom_labels.clear()
        self.rect_ids.clear()

    def reset_defaults(self):
        """恢复默认：裁剪宽96，裁剪高96，放大图最大边长200"""
        self.crop_width.set(96)
        self.crop_height.set(96)
        self.zoom_max_size.set(200)
        self.apply_size_change()

    def apply_size_change(self):
        if self.last_valid_x is not None and self.last_valid_y is not None:
            self.update_all_rectangles(self.last_valid_x, self.last_valid_y)
            self.update_all_magnifications(self.last_valid_x, self.last_valid_y)
        else:
            self.clear_all_rectangles()

    def on_mouse_move(self, event, img_index):
        canvas = event.widget
        disp_x = event.x
        disp_y = event.y
        if disp_x < 0 or disp_y < 0 or disp_x > canvas.winfo_width() or disp_y > canvas.winfo_height():
            return

        scale_x, scale_y = self.scale_factors[img_index]
        orig_x = disp_x * scale_x
        orig_y = disp_y * scale_y

        orig_w, orig_h = self.images[img_index].size
        orig_x = max(0, min(orig_w - 1, orig_x))
        orig_y = max(0, min(orig_h - 1, orig_y))

        self.current_orig_x = orig_x
        self.current_orig_y = orig_y
        self.last_valid_x = orig_x
        self.last_valid_y = orig_y

        self.update_all_rectangles(orig_x, orig_y)
        self.update_all_magnifications(orig_x, orig_y)

    def on_mouse_leave(self, img_index):
        pass

    def on_mouse_click(self, img_index):
        if self.last_valid_x is None or self.last_valid_y is None:
            messagebox.showwarning("警告", "请先移动鼠标至感兴趣区域")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.save_dir, f"session_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)

        crop_w = self.crop_width.get()
        crop_h = self.crop_height.get()

        saved_count = 0
        for idx, img in enumerate(self.images):
            cx = int(self.last_valid_x)
            cy = int(self.last_valid_y)
            left = max(0, cx - crop_w // 2)
            top = max(0, cy - crop_h // 2)
            right = min(img.width, left + crop_w)
            bottom = min(img.height, top + crop_h)
            if right - left < crop_w:
                if left == 0:
                    right = min(img.width, crop_w)
                else:
                    left = max(0, right - crop_w)
            if bottom - top < crop_h:
                if top == 0:
                    bottom = min(img.height, crop_h)
                else:
                    top = max(0, bottom - crop_h)

            cropped = img.crop((left, top, right, bottom))
            crop_filename = f"{self.image_names[idx]}_crop_{cx}_{cy}_{crop_w}x{crop_h}.png"
            cropped.save(os.path.join(session_dir, crop_filename))

            img_marked = img.copy()
            draw = ImageDraw.Draw(img_marked)
            draw.rectangle([left, top, right, bottom], outline='red', width=3)
            marked_filename = f"{self.image_names[idx]}_marked_{cx}_{cy}.png"
            img_marked.save(os.path.join(session_dir, marked_filename))

            saved_count += 2

        messagebox.showinfo("保存成功", f"已保存 {saved_count} 张图片(裁剪图+标记图)\n保存路径: {session_dir}")

    def update_all_rectangles(self, orig_x, orig_y):
        crop_w = self.crop_width.get()
        crop_h = self.crop_height.get()

        for idx, canvas in enumerate(self.canvas_list):
            scale_x, scale_y = self.scale_factors[idx]
            left_disp = (orig_x - crop_w / 2) / scale_x
            top_disp = (orig_y - crop_h / 2) / scale_y
            right_disp = (orig_x + crop_w / 2) / scale_x
            bottom_disp = (orig_y + crop_h / 2) / scale_y

            disp_w = self.display_widths[idx]
            disp_h = self.display_height
            left_disp = max(0, min(disp_w, left_disp))
            top_disp = max(0, min(disp_h, top_disp))
            right_disp = max(0, min(disp_w, right_disp))
            bottom_disp = max(0, min(disp_h, bottom_disp))

            rect_id = self.rect_ids[idx]
            canvas.coords(rect_id, left_disp, top_disp, right_disp, bottom_disp)
            if (right_disp - left_disp) < 2 or (bottom_disp - top_disp) < 2:
                canvas.itemconfig(rect_id, outline='yellow')
            else:
                canvas.itemconfig(rect_id, outline='red')
            canvas.tag_raise(rect_id)

    def clear_all_rectangles(self):
        for idx, canvas in enumerate(self.canvas_list):
            rect_id = self.rect_ids[idx]
            canvas.coords(rect_id, 0, 0, 0, 0)

    def update_all_magnifications(self, orig_x, orig_y):
        crop_w = self.crop_width.get()
        crop_h = self.crop_height.get()
        max_display = self.zoom_max_size.get()

        for idx, img in enumerate(self.images):
            cx = int(orig_x)
            cy = int(orig_y)
            left = max(0, cx - crop_w // 2)
            top = max(0, cy - crop_h // 2)
            right = min(img.width, left + crop_w)
            bottom = min(img.height, top + crop_h)
            if right - left < crop_w:
                if left == 0:
                    right = min(img.width, crop_w)
                else:
                    left = max(0, right - crop_w)
            if bottom - top < crop_h:
                if top == 0:
                    bottom = min(img.height, crop_h)
                else:
                    top = max(0, bottom - crop_h)

            cropped = img.crop((left, top, right, bottom))
            # 保持宽高比缩放到最大边长不超过 max_display
            orig_crop_w, orig_crop_h = cropped.size
            if orig_crop_w > orig_crop_h:
                new_w = max_display
                new_h = int(orig_crop_h * (max_display / orig_crop_w))
            else:
                new_h = max_display
                new_w = int(orig_crop_w * (max_display / orig_crop_h))
            # 确保至少1px
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            zoomed = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
            zoom_photo = ImageTk.PhotoImage(zoomed)

            label = self.zoom_labels[idx]
            label.config(image=zoom_photo, text="")
            label.image = zoom_photo
            # 更新Label的尺寸以适应图片
            label.update_idletasks()

        self.root.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = LocalMagnificationTool(root)
    root.mainloop()