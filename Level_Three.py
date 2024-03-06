# CV
import cv2.ximgproc
import numpy as np
from scipy.spatial import distance

# xArm7 integrated
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from xarm.wrapper import XArmAPI

# Mask edit > Generative AI
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import openai
import requests
from io import BytesIO
import random

openai.api_key = "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"

def generate_image(prompt_text, image_number):
    # Generate a random number between 1 and image_number
    random_num = random.randint(1, 8)
    # Format the filenames with the random number
    original_filename = f"../../LevelThreeDataBase/level3_original{random_num}.png"
    mask_filename = f"../../LevelThreeDataBase/level3_mask{random_num}.png"

    # Call the OpenAI API with the formatted filenames
    response = openai.Image.create_edit(
        image=open(original_filename, "rb"),
        mask=open(mask_filename, "rb"),
        prompt="monochrome, only one single black line, no intersection of line, representing " + prompt_text,
        n=1,
        size="512x512"
    )

    # Get the image URL from the API response
    image_url = response["data"][0]["url"]

    # Download the image data
    image_data = requests.get(image_url).content

    # Open the image from the downloaded data
    image = Image.open(BytesIO(image_data))

    # Save the image to a local file
    image.save(f"../../output/level3_generated_{image_number}.png")

    # Display the image in a window
    tk_image = ImageTk.PhotoImage(image)
    label = tk.Label(image_frame, image=tk_image)
    label.image = tk_image
    label.grid(row=0, column=image_number)

    # Create a button to save the image
    button = tk.Button(image_frame, text=f"Confirm {image_number}", command=lambda: save_image(f"../../output/level3_generated_{image_number}.png"))
    button.grid(row=1, column=image_number)

def save_image(image_path):
    # Save the image to a local file
    Image.open(image_path).save("../../output/level3_generated.png")

def confirm():
    prompt_text = prompt_entry.get()
    generate_image(prompt_text, 1)
    generate_image(prompt_text, 2)

# Create the tkinter window
window = tk.Tk()

# Add an input box for the prompt text
prompt_label = tk.Label(window, text="Enter prompt:")
prompt_label.pack()
prompt_entry = tk.Entry(window)
prompt_entry.pack()

# Add a "Confirm" button to trigger the image generation
confirm_button = tk.Button(window, text="Confirm", command=confirm)
confirm_button.pack()

# Add a frame to hold the images and buttons
image_frame = tk.Frame(window)
image_frame.pack()

# Start the tkinter main loop
window.mainloop()


# Read the image and convert to grayscale
img = cv2.imread("../../output/level3_generated.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

"""
# 对灰度图像进行高斯模糊
img = cv2.GaussianBlur(img, (15, 15), 0)
borderType = cv2.BORDER_CONSTANT
dst = cv2.copyMakeBorder(img, 20, 20, 20, 20, borderType, None, (255, 255, 255))
dst = cv2.GaussianBlur(dst, (5, 5), 100)
dst = dst[20:-20, 20:-20]

gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

# define the contrast and brightness value
contrast = 1.6  # Contrast control ( 0 to 127)
brightness = 1  # Brightness control (0-100)

# call addWeighted function. use beta = 0 to effectively only
# operate on one image
out = cv2.addWeighted(gray, contrast, gray, 0, brightness)

# Create a CLAHE object with desired parameters
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))

# Apply CLAHE to the image
img_clahe = clahe.apply(out)
# # 对图像进行边界填充
# borderType = cv2.BORDER_CONSTANT
# dst = cv2.copyMakeBorder(img, 20, 20, 20, 20, borderType, None, (255,255,255))
# cv2.imshow('dst', dst)
# cv2.waitKey(3000) 
# cv2.destroyAllWindows()
#
# # 进行其他的图像处理算法，比如滤波和降噪等
# dst = cv2.GaussianBlur(dst, (5,5), 200)
# # 去除填充的边界
# dst = dst[20:-20, 20:-20]
"""
# 沿y轴反转
#img_flipped = cv2.flip(img_clahe, 1)
img_flipped = cv2.flip(gray, 1)

_, binaryImg = cv2.threshold(img_flipped, 127, 255, cv2.THRESH_BINARY_INV)
# _, binaryImg2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY_INV)

binaryImg_Copied = cv2.flip(binaryImg, 1)
cv2.imshow('Binary Image for edit', binaryImg_Copied)
cv2.waitKey(2000)
cv2.destroyAllWindows()

skeletonImg = cv2.ximgproc.thinning(binaryImg)

try:
    coor3ds = cv2.findNonZero(skeletonImg)  # Find the coordinates of all non-zero pixels in the skeleton
    # assume points is a 3-dimensional numpy array
    coords = np.squeeze(coor3ds, axis=1)
    # print(coords[2])
    # s1 = np.shape(coords)
    # print(len(s1), type(coords))
    # for coord in coords:
    #     print(coord)
    # 将三维的数组通过np.squeeze()变成二维的形式
    # shape (n, 1, 2) >> (n, 2)

    num_points = 60
    indices = np.linspace(0, len(coords) - 1, num_points, dtype=np.uint32)
    points = coords[indices]
    for point in points:
        print(point)

    blank_image = np.zeros_like(skeletonImg)
    for point in points:
        x, y = point
        # cv2.circle(skeletonImg, (x, y), 1, (255, 255, 0), -1)
        cv2.circle(blank_image, (x, y), 1, (255, 255, 0), -1)

    for i in range(len(points)):
        x, y = points[i]
        print("Point", i + 1, ":", "(", x, ",", y, ")")

    # 假设points是一个500x2的NumPy数组，表示曲线上的所有点的(x, y)坐标
    # 假设曲线从左向右延伸，找到线的左端点
    left_endpoint = points[np.argmin(points[:, 0], axis=0), :]
    # >>> [ 78 400] 78前的空格（规范

    # 创建一个有序列表来存储沿着曲线遍历的点
    ordered_points = []
    ordered_points.append(left_endpoint)
    for ordered_point in ordered_points:
        print(ordered_point)

    # 从左端点开始，沿着曲线有序地遍历每个点
    current_point = left_endpoint
    print(current_point)
    # s2 = np.shape(current_point)
    # print(len(s2))
    # >>> len(s2)
    # >>> 1

    # Backup the Array points
    points_copied = points

    len_points = len(points)
    print(len_points)

    ######
    # 怎么去除重复点
    while len(ordered_points) < len_points:
        # 如何确保每一个点不重复的遍历，这个很重要
        # 统一不同维度的数组 eg.points == [current_point]
        index_to_remove = np.where(np.all(points == [current_point], axis=1))
        points = np.delete(points, index_to_remove, axis=0)

        # 计算当前点到所有其他点的距离，除了当前点相同坐标的点外。
        distances = distance.cdist(points, [current_point], 'euclidean')
        # distances[np.where(distances == 0)] = np.inf  # 设置为inf，避免选择自身

        # 找到最近的点并添加到有序列表中，引用最近点的x值
        nearest_point = points[np.argmin(distances), :]

        ordered_points.append(nearest_point)
        # 将新的最近点作为下一个起点
        current_point = nearest_point

    for ordered_point in ordered_points:
        print(ordered_point)

    # ordered_points现在包含沿着曲线有序遍历的所有点
    # for i in range(len(ordered_points)):
    #     x, y = ordered_points[i]
    #     print("Ordered Point", i + 1, ":", "(", x, ",", y, ")")

    cv2.imshow('Points on Skeleton', skeletonImg)
    cv2.waitKey(2000)
    cv2.imshow('Points on Skeleton', blank_image)
    cv2.waitKey(2000)

    # ________________________________________________________________________________________________________________
    """
    Start xArm system 
    """
    if len(sys.argv) >= 2:
        ip = sys.argv[1]
    else:
        try:
            from configparser import ConfigParser

            parser = ConfigParser()
            parser.read('../robot.conf')
            ip = parser.get('xArm', 'ip')
        except:
            ip = input('Please input the xArm ip address:')
            if not ip:
                print('input error, exit')
                sys.exit(1)
    ########################################################

    arm = XArmAPI(ip, is_radian=True)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)

    # arm.reset(wait=True)

    k = 200 / 512  # preset

    ######################
    # initial(退刀)
    arm.set_position(x=196, y=-250, z=120.5, roll=-180, pitch=0, yaw=0, speed=100, is_radian=False, wait=True)
    print(arm.get_position(), arm.get_position(is_radian=False))

    #  > 推刀 > working
    first_z = ordered_points[0][1]
    zArm_first = 265 - k * first_z
    arm.set_position(x=196, y=0, z=zArm_first, roll=-180, pitch=0, yaw=0, speed=70, is_radian=False, wait=True)
    print(arm.get_position(), arm.get_position(is_radian=False))

    # questions
    # 1. 先计算好数集？
    # roi_size / xarm_size
    path_feedback = np.zeros_like(skeletonImg)  ####

    for i in range(len(ordered_points)):
        x_img, y_img = ordered_points[i]
        cv2.circle(path_feedback, (x_img, y_img), 2, (255, 255, 0), -1)  ####
        cv2.imshow('Coords to Path', path_feedback)
        cv2.waitKey(1)  # Wait for 1 millisecond to update the window
        print("Coordinate", i + 1, ":", "(", x_img, ",", y_img, ")")

        xArm = k * x_img + 216
        # + offset_xarm
        zArm = 265 - k * y_img
        # + offset_zarm
        # 每次循环都需要计算，是否可以降低效率
        arm.set_position(x=xArm, y=0, z=zArm, roll=-180, pitch=0, yaw=0, speed=8, is_radian=False, wait=True)
        print(arm.get_position(), arm.get_position(is_radian=False))

    # > 退 > initial(退刀)
    last_x = ordered_points[-1][0]
    last_z = ordered_points[-1][1]
    xArm_last = k * x_img + 216
    zArm_last = 265 - k * y_img
    arm.set_position(x=xArm_last + 50, y=0, z=zArm_last + 50, roll=-180, pitch=0, yaw=0, speed=7, is_radian=False,
                     wait=True)
    print(arm.get_position(), arm.get_position(is_radian=False))

    arm.set_position(x=xArm_last + 50, y=-250, z=zArm_last + 50, roll=-180, pitch=0, yaw=0, speed=100, is_radian=False,
                     wait=True)
    print(arm.get_position(), arm.get_position(is_radian=False))

    arm.set_position(x=196, y=-250, z=120.5, roll=-180, pitch=0, yaw=0, speed=100, is_radian=False, wait=True)
    print(arm.get_position(), arm.get_position(is_radian=False))

    # 退刀 offset_xarm
    # xarm = offset_xarm(5mm - 10mm, 是否考虑斜率) + xarm起始坐标 + k * x_img坐标

    # arm.reset(wait=True)

except AttributeError:
    print("No non-zero pixels found in skeleton image")