import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from fuzzywuzzy import process
import os
from paddleocr import PaddleOCR, draw_ocr
import re
import paddle
from paddle.vision.transforms import Compose, Resize, ToTensor, Normalize
import paddle.vision.transforms.functional as F
from PIL import Image
import numpy as np


# 创建主窗口并设置窗口大小
root = tk.Tk()
root.title("船舶中大型构建型号识别系统")
root.geometry("1600x1200")  # 设置窗口初始大小为1600x800像素
root.configure(bg='darkblue')  # 设置界面背景为深蓝色

# 设置视频捕获
cap = cv2.VideoCapture(2)  # 通常使用0作为摄像头索引

# 指定保存图片的文件夹路径
save_folder = '/home/nvidia/PaddleOCR-Flask-main/images'  # 替换为你的文件夹路径
if not os.path.exists(save_folder):
    os.makedirs(save_folder)  # 如果文件夹不存在，则创建文件夹

def replace_with_most_similar(input_string, string_list, threshold=60):
    matches = process.extract(input_string, string_list)
    if matches:
        best_match = max(matches, key=lambda x: x[1])
        return best_match[0]
    else:
        return input_string

def process_part3(part3):
    part3 = re.sub(r'[A-Za-z]', '', part3)
    part3 = part3.replace('0', '')
    if not part3:
        return "01"
    min_digit = min(part3)
    result = "0" + min_digit
    return result

def save_frame():
    ret, frame = cap.read()
    if ret:
        file_path = os.path.join(save_folder, 'captured_image.jpg')
        
        # 转换为RGB格式
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)

        # 获取photo_frame的当前大小
        width = photo_frame.winfo_width()
        height = photo_frame.winfo_height()

        # 确保宽度和高度是合理的
        if width <= 0 or height <= 0:
            width = 350  # 默认宽度
            height = 250  # 默认高度

        # 调整图片大小以适应photo_frame
        img = img.resize((width, height), Image.ANTIALIAS)

        # 转换为PhotoImage对象
        imgtk = ImageTk.PhotoImage(image=img)

        # 显示图片
        lbl_image.config(image=imgtk)
        lbl_image.image = imgtk  # 保持对PhotoImage的引用

        # 保存图片
        try:
            cv2.imwrite(file_path, frame)
            print(f"Image saved successfully at {file_path}")
        except Exception as e:
            print(f"Failed to save image: {e}")

        # 检查图片是否成功保存
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > 0:
                print(f"Image saved successfully. File size: {file_size} bytes")
            else:
                print("Image file exists but is empty.")
        else:
            print("Image file does not exist.")

def replace_text(input_string):
    part1_list = ["TU10E1", "TU10E2", "TU10E3", "TU10E4", "TU10E5", "TUE203", "TU10B12", "TU10B11", "TU10B7", "TUE201"]
    part2_list = ["BWK65", "DOK04", "FRA07", "FTD31", "MMB107", "MW50", "PMG01", "SCA226", "SCA227", "SMK67", "SMK341", "WMK03", "WMK37", "WMK42", "WMK43", "ASK23", "BWK30", "BWK15", "BWK16", "BWK25", "BWK52", "FRA60", "MMB52", "MMB76", "WMK34", "WMK60", "WMK61", "ASZ89", "MW49", "MW51", "WMK05", "WMK27", "WMK29", "BMD152", "BMK04", "BMK139", "BMK18", "BMK20", "BMK21", "MW67", "LEC05", "LEC06", "XLK122", "XLK221", "XLK101", "XLK201", "XLK202", "XLK203", "XLK313", "XLK60", "WMU10", "WMU11", "WMU12", "WMU13", "WMU14", "WMU15", "XLK104", "XLK105", "XLK204", "XLK205", "XLK41", "XLK58"]
    input_string = re.sub(r'-{2,}', '-', input_string)
    if input_string:
        input_string = "TU" + input_string[2:]
    pattern = r'^([^-\s]+)-(.+)-([^-\s]+)$'
    match = re.match(pattern, input_string)
    if match:
        part1, part2, part3 = match.groups()
        new_part1 = replace_with_most_similar(part1, part1_list, 60)
        new_part2 = replace_with_most_similar(part2, part2_list, 60)
        new_part3 = process_part3(part3)
        new_string = f"{new_part1}-{new_part2}-{new_part3}"
        return new_string
    else:
        return input_string



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

def detect():
    img = cv2.imread("/home/nvidia/PaddleOCR-Flask-main/images/captured_image.jpg")
    ocr = PaddleOCR(use_angle_cls=True, lang='en', rec_model_dir='./rec_inf_model', det_model_dir='./det_inf_model', use_gpu=True)
    img_result = ocr.ocr(img)
    results = []
    for sublist in img_result:
        for subsublist in sublist:
            text = subsublist[1][0]
            results.append(text)
    if results:
        final_result = max(results, key=len)
    else:
        final_result = ''
    final_result = replace_text(final_result)
    print(final_result)
    
    # 获取当前lbl_result的文本内容
    current_text = lbl_result.cget("text")
    
    # 将新的识别结果追加到现有文本内容中，并另起一行
    new_text = f"{current_text}\n文字识别结果：{final_result}"
    
    # 更新lbl_result的文本内容
    lbl_result.config(text=new_text)

def close_window():
    cap.release()
    root.destroy()
'''
=================================================================================================
'''

#分类模型实现


# 加载推理模型
def load_inference_model(model_path):
    model_file = os.path.join(model_path, "inference.pdmodel")
    params_file = os.path.join(model_path, "inference.pdiparams")
    config = paddle.inference.Config(model_file, params_file)
    predictor = paddle.inference.create_predictor(config)
    return predictor


# 预处理图片
def preprocess_image(image_path, input_size=(224, 224)):
    transform = Compose([
        Resize(input_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image[np.newaxis, ...]  # 添加 batch 维度
    return image


def classify_image(predictor, image):
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    input_handle = predictor.get_input_handle(input_names[0])
    output_handle = predictor.get_output_handle(output_names[0])

    # 确保输入是 NumPy 数组
    image = image.numpy()  # 将 Paddle Tensor 转换为 NumPy 数组

    input_handle.reshape(image.shape)
    input_handle.copy_from_cpu(image)

    predictor.run()

    output = output_handle.copy_to_cpu()
    return output

# 加载类别标签
def load_labels(label_file):
    with open(label_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels




# 主函数
def classfication():
    # 加载模型
    
    model_path = "./inference/PPLCNet_x1_0_infer"
    predictor = load_inference_model(model_path) 
    # 加载图片并预处理

# 加载图片并预处理 
    image = preprocess_image("/home/nvidia/PaddleOCR-Flask-main/images/captured_image.jpg")
    #image = preprocess_image("./train_data/val/image_0345.jpg")
    # 进行分类
    output = classify_image(predictor, image)

    label_file = "./label.txt"
    # 加载类别标签
    labels = load_labels(label_file)

    # 获取最高概率的类别
    pred_class = np.argmax(output)
    pred_score = np.max(output)
    pred_label = labels[pred_class]

    # 输出结果
    print(f"分类结果为：{pred_label}，置信度：{pred_score:.4f}")
    result_text = ""
    if pred_class == 0:
        result_text = "TU10E1-SCA226-01"  # 如果匹配，则赋值
    if pred_class == 1:
        result_text = "TU10E1-WMK37-01"
     # 获取当前lbl_result的文本内容
    current_text = lbl_result.cget("text")
    print(pred_class)
    # 将新的识别结果追加到现有文本内容中，并另起一行
    new_text = f"{current_text}\n形状识别结果：{result_text}" 
    
    # 更新lbl_result的文本内容
    lbl_result.config(text=new_text)




'''
      # 读取图像
    image = cv2.imread("/home/nvidia/PaddleOCR-Flask-main/images/captured_image.jpg")

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用 Canny 边缘检测
    edges = cv2.Canny(gray, 10, 200)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个空白图像用于绘制轮廓
    contour_image = np.zeros_like(image)

    # 绘制轮廓
    for contour in contours:
        cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)

    # 构建输出文件路径
    output_path = os.path.join("/home/nvidia/PaddleOCR-Flask-main/images", "captured_image.jpg")

    # 保存轮廓图
    cv2.imwrite(output_path, contour_image)
'''
    






'''
===============================================================================================
'''





# 创建 GUI 组件

title = tk.Frame(root,bg='darkblue')
title.place(x=0, y=0,  width=1600, height=100)


# 创建视频流区域，
video_frame = tk.Frame(root)
video_frame.place(x=100, y=100,  width=700, height=500)
video_frame.config(bd=5, relief="ridge")


#创建拍摄图片的区域
photo_frame = tk.LabelFrame(root)
photo_frame.place(x=800, y=100, width=700, height=500)
photo_frame.config(bd=5, relief="ridge")

# 创建识别记录区域

record_frame = tk.LabelFrame(root)
record_frame.place(x=100, y=700, width=1400, height=250)
record_frame.config(bd=5, relief="ridge")


canvas = tk.Canvas(record_frame, highlightthickness=0)
scrollbar = tk.Scrollbar(record_frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg='darkblue')

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# 创建标题文本
title_label = tk.Label(title, text="船舶中大型构件型号识别系统", font=("Arial", 32), bg='darkblue', fg='white')
title_label.place(relx=0.5, rely=0.5, anchor="center")

lbl_camera = tk.Label(video_frame)
lbl_camera.pack(expand=True, fill=tk.BOTH)

lbl_image = tk.Label(photo_frame)
lbl_image.pack(expand=True, fill=tk.BOTH)

lbl_result = tk.Label(scrollable_frame, font=("Arial", 12), fg='black', justify=tk.LEFT)
lbl_result.pack(expand=True, fill=tk.BOTH)


# 创建清除按钮
def clear_records():
    lbl_result.config(text="")

btn_clear = tk.Button(record_frame, text="清除记录", command=clear_records)
btn_clear.pack(side="bottom", fill="x")


btn_capture = tk.Button(root, text="拍摄", font=("Arial", 12), command=save_frame, width=8, height=1)
btn_capture.place(x=300, y=640)

btn_detect = tk.Button(root, text="文字识别", font=("Arial", 12), command=detect, width=8, height=1)
btn_detect.place(x=700, y=640)

btn_shape = tk.Button(root, text="形状识别", font=("Arial", 12), command=classfication, width=8, height=1)
btn_shape.place(x=1100, y=640)



update_camera()

root.mainloop()