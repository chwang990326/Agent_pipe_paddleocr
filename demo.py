import cv2
import tkinter as tk
from PIL import Image, ImageTk
import os
import numpy as np

# ==================== GUI 与 核心逻辑 ====================

# 创建主窗口并设置窗口大小
root = tk.Tk()
root.title("船舶中大型构建型号识别系统")
root.geometry("1600x900")  
root.configure(bg='darkblue')

# 设置视频捕获
cap = cv2.VideoCapture(2)  # 通常使用0作为摄像头索引

# 指定保存图片的文件夹路径
save_folder = '/home/nvidia/PaddleOCR-Flask-main/images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)  # 如果文件夹不存在，则创建文件夹
captured_file_path = os.path.join(save_folder, 'captured_image.jpg')

def update_camera():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return
    
    # 获取video_frame的当前大小
    width = video_frame.winfo_width()
    height = video_frame.winfo_height()
    
    # 如果大小为0，使用默认大小
    if width <= 0 or height <= 0:
        width, height = 700, 500  # 默认大小
    
    # 调整视频帧的大小以适应video_frame
    frame = cv2.resize(frame, (width, height))

    # 转换颜色空间并显示
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lbl_camera.config(image=imgtk)
    lbl_camera.image = imgtk  # 保持对PhotoImage的引用
    
    # 递归调用更新函数
    root.after(10, update_camera)

def save_frame():
    ret, frame = cap.read()
    if ret:
        # 转换为RGB格式
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)

        # 获取photo_frame的当前大小
        width = photo_frame.winfo_width()
        height = photo_frame.winfo_height()

        # 确保宽度和高度是合理的
        if width <= 0 or height <= 0:
            width, height = 700, 500

        # 调整图片大小以适应photo_frame
        try:
            resample_method = Image.Resampling.LANCZOS
        except AttributeError:
            resample_method = Image.ANTIALIAS
            
        img_resized = img.resize((width, height), resample_method)

        # 转换为PhotoImage对象
        imgtk = ImageTk.PhotoImage(image=img_resized)

        # 显示图片
        lbl_image.config(image=imgtk)
        lbl_image.image = imgtk  # 保持对PhotoImage的引用

        # 保存图片
        try:
            cv2.imwrite(captured_file_path, frame)
            print(f"Image saved successfully at {captured_file_path}")
        except Exception as e:
            print(f"Failed to save image: {e}")

def identify_objects():
    if not os.path.exists(captured_file_path):
        print("请先拍摄图片！")
        return

    # 1. 读取拍摄的图片
    img = cv2.imread(captured_file_path)
    if img is None:
        print("无法读取图片！")
        return

    # 2. 获取图片的宽和高
    img_h, img_w, _ = img.shape
    
    # 3. 计算上下两半的分割线
    half_h = img_h // 2
    
    # ==================== 绘制上半部分的框（宽度较小） ====================
    # 宽度比例 0.45，高度比例 0.60
    box_w_top = int(img_w * 0.45)
    box_h_top = int(half_h * 0.60)
    
    # 计算左上角坐标，使其稍微偏左（中心点在总宽度的 40% 处）
    center_x_top = int(img_w * 0.4)
    x1_top = center_x_top - (box_w_top // 2)
    y1_top = (half_h - box_h_top) // 2
    x2_top = x1_top + box_w_top
    y2_top = y1_top + box_h_top
    
    # 画框 (线宽调整为 2)
    cv2.rectangle(img, (x1_top, y1_top), (x2_top, y2_top), (0, 255, 0), 2)
    # 写字 (字号 0.7，线条粗细调整为 2，使其加粗)
    text_y_top = max(y1_top - 10, 20)
    cv2.putText(img, "TU10E1-SCA226-01", (x1_top, text_y_top), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ==================== 绘制下半部分的框（保持原样） ====================
    # 宽度比例 0.75，高度比例 0.75
    box_w_bot = int(img_w * 0.75)
    box_h_bot = int(half_h * 0.75)
    
    # 计算左上角坐标，使其稍微偏右（中心点在总宽度的 60% 处）
    center_x_bot = int(img_w * 0.6)
    x1_bot = center_x_bot - (box_w_bot // 2)
    y1_bot = half_h + (half_h - box_h_bot) // 2
    x2_bot = x1_bot + box_w_bot
    y2_bot = y1_bot + box_h_bot
    
    # 画框 (线宽调整为 2)
    cv2.rectangle(img, (x1_bot, y1_bot), (x2_bot, y2_bot), (0, 255, 0), 2)
    # 写字 (字号 0.7，线条粗细调整为 2，使其加粗)
    text_y_bot = max(y1_bot - 10, half_h + 20)
    cv2.putText(img, "TU10E1-WMK37-01", (x1_bot, text_y_bot), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # 4. 更新右侧界面的显示为带有画框的图片
    cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2image)
    
    width = photo_frame.winfo_width()
    height = photo_frame.winfo_height()
    if width <= 0 or height <= 0:
        width, height = 700, 500

    try:
        resample_method = Image.Resampling.LANCZOS
    except AttributeError:
        resample_method = Image.ANTIALIAS

    pil_img_resized = pil_img.resize((width, height), resample_method)
    imgtk = ImageTk.PhotoImage(image=pil_img_resized)
    
    lbl_image.config(image=imgtk)
    lbl_image.image = imgtk

def close_window():
    cap.release()
    root.destroy()

# ==================== GUI 组件布局 ====================

# 标题区域
title = tk.Frame(root, bg='darkblue')
title.place(x=0, y=0, width=1600, height=100)
title_label = tk.Label(title, text="船舶中大型构件型号识别系统", font=("Microsoft YaHei", 32, "bold"), bg='darkblue', fg='white')
title_label.place(relx=0.5, rely=0.5, anchor="center")

# 左侧视频流区域
video_frame = tk.Frame(root)
video_frame.place(x=100, y=100, width=700, height=500)
video_frame.config(bd=5, relief="ridge")
lbl_camera = tk.Label(video_frame)
lbl_camera.pack(expand=True, fill=tk.BOTH)

# 右侧拍摄/识别结果区域
photo_frame = tk.LabelFrame(root)
photo_frame.place(x=800, y=100, width=700, height=500)
photo_frame.config(bd=5, relief="ridge")
lbl_image = tk.Label(photo_frame)
lbl_image.pack(expand=True, fill=tk.BOTH)

# 按钮区域
btn_capture = tk.Button(root, text="拍摄", font=("Microsoft YaHei", 16, "bold"), command=save_frame, width=10, height=2)
btn_capture.place(x=350, y=650)  

btn_detect = tk.Button(root, text="识别", font=("Microsoft YaHei", 16, "bold"), command=identify_objects, width=10, height=2)
btn_detect.place(x=1050, y=650)  

# 启动视频流更新并进入主循环
update_camera()

# 绑定窗口关闭事件
root.protocol("WM_DELETE_WINDOW", close_window)
root.mainloop()