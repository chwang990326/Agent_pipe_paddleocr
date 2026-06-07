import torch
from ultralytics import YOLO
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import os
import numpy as np
import paddle
from paddle.vision.transforms import Compose, Resize, ToTensor, Normalize

# ==================== 模型加载与初始化 ====================

# 1. 加载你自己微调训练出的 YOLOv8 模型
yolo_model = YOLO('best.pt') 

# 2. 加载 Paddle 形状分类推断模型
def load_inference_model(model_path):
    model_file = os.path.join(model_path, "inference.pdmodel")
    params_file = os.path.join(model_path, "inference.pdiparams")
    config = paddle.inference.Config(model_file, params_file)
    predictor = paddle.inference.create_predictor(config)
    return predictor

def load_labels(label_file):
    if not os.path.exists(label_file):
        return ["Unknown"] * 100 # 如果找不到标签文件，提供默认降级方案
    with open(label_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# 初始化分类模型和标签
model_path = "./inference/PPLCNet_x1_0_infer"
label_file = "./label.txt"
predictor = load_inference_model(model_path)
labels = load_labels(label_file)

# 修改预处理函数：直接接收 OpenCV 裁剪后的图像数组，而不是文件路径
def preprocess_image_from_cv2(cv_img, input_size=(224, 224)):
    transform = Compose([
        Resize(input_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 将 CV2 的 BGR 转换为 RGB 并转为 PIL Image
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    image = transform(pil_img)
    image = image[np.newaxis, ...]  # 添加 batch 维度
    return image

def classify_image(predictor, image):
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    input_handle = predictor.get_input_handle(input_names[0])
    output_handle = predictor.get_output_handle(output_names[0])

    image = image.numpy()  # 将 Paddle Tensor 转换为 NumPy 数组
    input_handle.reshape(image.shape)
    input_handle.copy_from_cpu(image)
    predictor.run()
    output = output_handle.copy_to_cpu()
    return output


# ==================== GUI 与 核心逻辑 ====================

# 创建主窗口并设置窗口大小
root = tk.Tk()
root.title("船舶中大型构建型号识别系统")
root.geometry("1600x900")  # 因为去掉了底部，高度可以适当缩小，原为1600x1200
root.configure(bg='darkblue')

# 设置视频捕获
cap = cv2.VideoCapture(2)  # 根据实际情况修改摄像头索引

# 指定保存图片的文件夹路径
save_folder = '/home/nvidia/PaddleOCR-Flask-main/images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
captured_file_path = os.path.join(save_folder, 'captured_image.jpg')

def update_camera():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return
    
    width = video_frame.winfo_width()
    height = video_frame.winfo_height()
    if width <= 0 or height <= 0:
        width, height = 700, 500
    
    frame = cv2.resize(frame, (width, height))
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lbl_camera.config(image=imgtk)
    lbl_camera.image = imgtk
    
    root.after(10, update_camera)

def save_frame():
    ret, frame = cap.read()
    if ret:
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)

        width = photo_frame.winfo_width()
        height = photo_frame.winfo_height()
        if width <= 0 or height <= 0:
            width, height = 700, 500

        # 兼容较新版本的 Pillow
        try:
            resample_method = Image.Resampling.LANCZOS
        except AttributeError:
            resample_method = Image.ANTIALIAS

        img_resized = img.resize((width, height), resample_method)
        imgtk = ImageTk.PhotoImage(image=img_resized)

        lbl_image.config(image=imgtk)
        lbl_image.image = imgtk

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

    # 2. 使用你的专属 YOLO 模型进行目标检测
    results = yolo_model(img)

    # 3. 遍历检测到的每一个目标
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 获取边界框坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 过滤掉不合理的框
            if x2 <= x1 or y2 <= y1:
                continue

            # 4. Crop: 裁剪出管件区域
            crop_img = img[y1:y2, x1:x2]

            # 5. 形状分类：对裁剪出的图片进行预处理和推断
            paddle_input = preprocess_image_from_cv2(crop_img)
            output = classify_image(predictor, paddle_input)
            
            pred_class = np.argmax(output)
            
            # 根据你原代码的逻辑进行赋值
            result_text = labels[pred_class] if pred_class < len(labels) else "Unknown"
            if pred_class == 0:
                result_text = "TU10E1-SCA226-01"
            elif pred_class == 1:
                result_text = "TU10E1-WMK37-01"

            # 6. 在原图上画框并写上识别出的编号
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 保证文字不会超出图片顶部
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(img, result_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 7. 更新右侧界面的显示为带有画框的图片
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
title_label = tk.Label(title, text="船舶中大型构件型号识别系统", font=("Arial", 32), bg='darkblue', fg='white')
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

# 按钮区域 (放置在画面正下方)
btn_capture = tk.Button(root, text="拍摄", font=("Arial", 16), command=save_frame, width=10, height=2)
btn_capture.place(x=350, y=650)  # 左侧视频框的正下方附近

btn_detect = tk.Button(root, text="识别", font=("Arial", 16), command=identify_objects, width=10, height=2)
btn_detect.place(x=1050, y=650)  # 右侧图片框的正下方附近

# 启动视频流更新并进入主循环
update_camera()

# 绑定窗口关闭事件，确保释放摄像头资源
root.protocol("WM_DELETE_WINDOW", close_window)
root.mainloop()