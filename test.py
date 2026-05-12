import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox, scrolledtext
from ultralytics import YOLO
import cv2
import os
from collections import deque
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime

# UI 主题
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# 新建保存结果文件夹
SAVE_DIR = "detect_result"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

class TankDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("坦克目标检测系统")
        self.root.geometry("1150x750")
        self.center_window(1150, 750)

        # 核心变量
        self.model = None
        self.selected_file = None
        self.file_type = None
        self.detecting = False
        self.current_imgtk = None
        self.video_capture = None
        self.conf_threshold = 0.6
        self.last_annotated = None

        # 播放帧率控制
        self.playback_fps = 25
        self.playback_interval = 1.0 / self.playback_fps

        # 缓冲区与线程同步
        self.frame_buffer = deque(maxlen=30)
        self.buffer_lock = threading.Lock()
        self.buffer_cond = threading.Condition(self.buffer_lock)
        self.stop_event = threading.Event()

        # 流畅度参数
        self.skip_frame = 2
        self.resize_scale = 0.5
        self.ui_update_interval = 0.03

        # 统计参数
        self.fps = 0.0
        self.total_target = 0
        self.frame_start_time = 0

        # 构建UI + 加载模型
        self.build_ui()
        self.log("系统初始化完成，开始加载模型...")
        self.load_model()

    def center_window(self, width, height):
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - width) // 2
        y = (screen_h - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def load_model(self):
        try:
            model_path = r"C:\Users\20753\Documents\WeChat Files\wxid_h6cinb9lwogj22\FileStorage\File\2026-05\best(1).pt"
            if not os.path.exists(model_path):
                model_path = r"runs/detect/tank_train_final/weights/best.pt"
            if not os.path.exists(model_path):
                messagebox.showerror("错误", "模型不存在！请检查路径")
                self.status_label.configure(text="❌ 模型不存在", text_color="red")
                self.log("模型加载失败：路径不存在")
                return
            self.model = YOLO(model_path)
            self.log(self.model.device)
            self.model.verbose = False
            self.status_label.configure(text="✅ 模型加载成功", text_color="green")
            self.log(f"成功加载模型：{model_path}")
        except Exception as e:
            self.status_label.configure(text="❌ 模型加载失败", text_color="red")
            self.log(f"模型加载异常：{str(e)}")

    def build_ui(self):
        # 顶部标题
        title_frame = ctk.CTkFrame(self.root, height=60)
        title_frame.pack(fill="x", padx=20, pady=10)
        title_label = ctk.CTkLabel(title_frame, text="坦克目标检测系统", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(expand=True)

        # 左侧总面板
        left_frame = ctk.CTkFrame(self.root, width=320)
        left_frame.pack(side="left", fill="y", padx=15, pady=10)

        # 状态标签
        self.status_label = ctk.CTkLabel(left_frame, text="🔄 加载模型中...", font=ctk.CTkFont(size=12))
        self.status_label.pack(padx=20, pady=10)

        # 1. 置信度滑动条
        conf_frame = ctk.CTkFrame(left_frame)
        conf_frame.pack(fill="x", padx=15, pady=10)
        ctk.CTkLabel(conf_frame, text="置信度阈值调节").pack(pady=5)
        self.conf_slider = ctk.CTkSlider(conf_frame, from_=0.3, to=0.8, number_of_steps=50, command=self.update_conf)
        self.conf_slider.set(0.6)
        self.conf_slider.pack(fill="x", padx=10, pady=5)
        self.conf_text = ctk.CTkLabel(conf_frame, text=f"当前：{self.conf_threshold:.2f}")
        self.conf_text.pack(pady=2)

        self.fps_slider = ctk.CTkSlider(conf_frame, from_=1, to=60, number_of_steps=59, command=self.update_playback_fps)
        self.fps_slider.set(self.playback_fps)
        self.fps_slider.pack(fill="x", padx=10, pady=5)
        self.fps_text = ctk.CTkLabel(conf_frame, text=f"播放帧率：{self.playback_fps:.0f}")
        self.fps_text.pack(pady=2)

        # 2. 文件选择
        file_frame = ctk.CTkFrame(left_frame)
        file_frame.pack(fill="x", padx=15, pady=10)
        ctk.CTkLabel(file_frame, text="选择检测文件").pack(pady=5)
        ctk.CTkButton(file_frame, text="📷 选择图片", command=lambda: self.select_file("image")).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(file_frame, text="📹 选择视频", command=lambda: self.select_file("video")).pack(fill="x", padx=10, pady=5)

        # 3. 控制按钮
        self.detect_btn = ctk.CTkButton(left_frame, text="🚀 开始检测", command=self.start_detect, fg_color="green")
        self.detect_btn.pack(fill="x", padx=15, pady=15)
        self.stop_btn = ctk.CTkButton(left_frame, text="🛑 停止检测", command=self.stop_detect, fg_color="red", state="disabled")
        self.stop_btn.pack(fill="x", padx=15, pady=5)

        # 4. 统计信息面板
        stat_frame = ctk.CTkFrame(left_frame)
        stat_frame.pack(fill="x", padx=15, pady=10)
        ctk.CTkLabel(stat_frame, text="实时检测统计", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.fps_label = ctk.CTkLabel(stat_frame, text="实时FPS：0.00")
        self.fps_label.pack(pady=2)
        self.count_label = ctk.CTkLabel(stat_frame, text="当前目标数：0")
        self.count_label.pack(pady=2)

        # 右侧：预览 + 日志
        right_frame = ctk.CTkFrame(self.root)
        right_frame.pack(side="right", expand=True, fill="both", padx=15, pady=10)

        # 预览区
        ctk.CTkLabel(right_frame, text="检测结果预览", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        self.canvas = tk.Canvas(right_frame, bg="black", highlightthickness=0)
        self.canvas.pack(expand=True, fill="both", padx=10, pady=10)
        self.result_label = ctk.CTkLabel(right_frame, text="等待检测...", font=ctk.CTkFont(size=12))
        self.result_label.pack(pady=5)

        # 日志面板
        log_frame = ctk.CTkFrame(right_frame, height=150)
        log_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(log_frame, text="运行日志", font=ctk.CTkFont(weight="bold")).pack(pady=3)
        self.log_text = scrolledtext.ScrolledText(log_frame, width=80, height=6, bg="#2b2b2b", fg="#cccccc")
        self.log_text.pack(padx=10, pady=5, fill="x")

    def log(self, msg):
        """写入日志"""
        t = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{t}] {msg}\n")
        self.log_text.see(tk.END)
        print("Log: ", msg)

    def update_conf(self, val):
        """滑动条更新置信度"""
        self.conf_threshold = round(float(val), 2)
        self.conf_text.configure(text=f"当前：{self.conf_threshold:.2f}")

    def update_playback_fps(self, val):
        """播放帧率调节"""
        self.playback_fps = max(1.0, float(val))
        self.playback_interval = 1.0 / self.playback_fps
        self.fps_text.configure(text=f"播放帧率：{self.playback_fps:.0f}")

    def select_file(self, file_type):
        self.stop_detect()
        filetypes = [("图片文件", "*.jpg *.png *.jpeg")] if file_type == "image" else [("视频文件", "*.mp4 *.avi *.mov")]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if not path:
            return
        self.selected_file = path
        self.file_type = file_type
        self.result_label.configure(text=f"已选择：{os.path.basename(path)}", text_color="blue")
        self.detect_btn.configure(state="normal")
        self.log(f"选择文件：{os.path.basename(path)}，类型：{'图片' if file_type=='image' else '视频'}")

    def show_img(self, img_cv):
        try:
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw < 150:
                cw = 650
                ch = 480
            ratio = min(cw / img_pil.width, ch / img_pil.height)
            w = int(img_pil.width * ratio)
            h = int(img_pil.height * ratio)
            img_pil = img_pil.resize((w, h), Image.Resampling.LANCZOS)
            self.current_imgtk = ImageTk.PhotoImage(img_pil)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=self.current_imgtk, anchor="center")
        except:
            pass

    def start_detect(self):
        if not self.model or not self.selected_file:
            messagebox.showwarning("提示", "请先加载模型并选择文件")
            return
        self.detecting = True
        self.stop_event.clear()
        self.detect_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.log("开始检测任务...")
        threading.Thread(target=self.run_detect, daemon=True, name="DisplayThread").start()

    def stop_detect(self):
        self.detecting = False
        self.stop_event.set()
        with self.buffer_cond:
            self.buffer_cond.notify_all()
        try:
            if self.video_capture:
                self.video_capture.release()
        except:
            pass
        self.detect_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.log("已停止检测")

    def save_result_image(self, frame):
        """保存检测后图片"""
        name = os.path.splitext(os.path.basename(self.selected_file))[0]
        save_path = os.path.join(SAVE_DIR, f"{name}_detect.jpg")
        cv2.imwrite(save_path, frame)
        self.log(f"检测结果已保存：{save_path}")

    def save_result_video(self, width, height, fps):
        """创建视频保存句柄"""
        name = os.path.splitext(os.path.basename(self.selected_file))[0]
        save_path = os.path.join(SAVE_DIR, f"{name}_detect.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        return out, save_path

    def video_process_loop(self, conf):
        """视频读取与模型推理线程：全速读取视频并生成标注帧"""
        try:
            self.video_capture = cv2.VideoCapture(self.selected_file)
            frame_count = 0
            while self.detecting and self.video_capture.isOpened() and not self.stop_event.is_set():
                ret, frame = self.video_capture.read()
                if not ret:
                    break

                if frame_count % self.skip_frame != 0:
                    frame_count += 1
                    continue

                small_frame = cv2.resize(frame, (0, 0), fx=self.resize_scale, fy=self.resize_scale)
                res = self.model(small_frame, conf=conf, imgsz=640, verbose=False)
                annotated = res[0].plot()
                curr_num = len(res[0].boxes)
                annotated_full = cv2.resize(annotated, (frame.shape[1], frame.shape[0]))

                with self.buffer_cond:
                    if len(self.frame_buffer) >= self.frame_buffer.maxlen:
                        self.frame_buffer.popleft()
                    self.frame_buffer.append((annotated_full, curr_num))
                    self.buffer_cond.notify()

                self.root.after(0, lambda n=curr_num: self.count_label.configure(text=f"当前目标数：{n}"))
                frame_count += 1
        finally:
            self.detecting = False
            if self.video_capture:
                self.video_capture.release()
            with self.buffer_cond:
                self.buffer_cond.notify_all()

    def run_detect(self):
        conf = self.conf_threshold
        self.result_label.configure(text="检测中...", text_color="orange")
        self.total_target = 0
        with self.buffer_cond:
            self.frame_buffer.clear()

        if self.file_type == "image":
            self.frame_start_time = time.time()
            res = self.model(self.selected_file, conf=conf, imgsz=640)
            frame = res[0].plot()
            # 统计目标数
            num = len(res[0].boxes)
            self.total_target = num
            # 更新UI统计
            self.root.after(0, lambda: self.count_label.configure(text=f"当前目标数：{num}"))
            self.root.after(0, lambda: self.fps_label.configure(text="实时FPS：静态图片"))
            self.root.after(0, lambda: self.show_img(frame))
            self.save_result_image(frame)
            self.root.after(0, lambda: self.result_label.configure(text="✅ 图片检测完成", text_color="green"))
            self.log(f"图片检测完成，识别到目标数量：{num}")

        elif self.file_type == "video":
            temp_cap = cv2.VideoCapture(self.selected_file)
            width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = temp_cap.get(cv2.CAP_PROP_FPS) or self.playback_fps
            temp_cap.release()

            out_writer, save_path = self.save_result_video(width, height, fps)

            video_thread = threading.Thread(
                target=self.video_process_loop,
                args=(conf,),
                daemon=True,
                name="VideoProcessThread"
            )
            video_thread.start()

            end_time = time.time()
            while (self.detecting or self.frame_buffer) and not self.stop_event.is_set():
                with self.buffer_cond:
                    while not self.frame_buffer and self.detecting and not self.stop_event.is_set():
                        self.buffer_cond.wait(timeout=0.1)
                    if not self.frame_buffer:
                        if not self.detecting or self.stop_event.is_set():
                            break
                        continue
                    frame, curr_num = self.frame_buffer.popleft()

                self.root.after(0, lambda f=frame: self.show_img(f))
                out_writer.write(frame)
                self.root.after(0, lambda: self.fps_label.configure(text=f"实时FPS：{self.playback_fps:.2f}"))

                display_cost = time.time() - end_time
                delay = max(0.0, self.playback_interval - display_cost)
                if delay > 0:
                    time.sleep(delay)
                end_time = time.time()

            video_thread.join(timeout=1.0)
            out_writer.release()
            self.root.after(0, lambda: self.result_label.configure(text="✅ 视频检测完成", text_color="green"))
            self.log(f"视频检测完成，结果已自动保存至：{save_path}")

        self.detect_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

if __name__ == "__main__":
    root = ctk.CTk()
    app = TankDetectionUI(root)
    root.mainloop()