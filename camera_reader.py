# -*- coding:utf-8 -*-
'''
opencv分类器参考文章 https://blog.csdn.net/u010402786/article/details/52261933
'''

import cv2

from model_train import Model
from data_input import read_name_list

faceData_file_path = './faceData/'
# 脸部图片文件路径

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    # 读取dataset数据集下的子文件夹名称
    name_list = read_name_list(faceData_file_path)
    model = Model()
    model.load()
    while True:
        _, frame = cap.read()  # 获取摄像头图像

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 图像灰度化

        cascade_path = "cv2_data/haarcascade_frontalface_alt.xml"  # opencv人脸检测分类器 人脸检测器（Haar_1）
        cascade = cv2.CascadeClassifier(cascade_path)  # 加载人脸检测分类器

        # 检测到的人脸
        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
        # facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.01, minNeighbors=3, minSize=(3, 3))

        if len(facerect) > 0:  # 检测到人脸
            print(name_list)
            print('face detected')
            color = (0, 0, 255)  # opencv B-G-R 红色
            for rect in facerect:
                # 获取图像上人脸的左上角的x,y坐标，和人脸的宽和高
                x, y = rect[0:2]
                width, height = rect[2:4]
                image = frame[y - 10: y + height, x: x + width]
                cv2.rectangle(frame, (x - 10, y - 10), (x + width + 10, y + height + 10), color, 2)
                result = model.predict(image)
                print(name_list)
                print('result:', result)
                print('人物是:', name_list[result])
                cv2.putText(frame, name_list[result],
                            (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 255, 0), 2)
        cv2.imshow('faceRecogniztion', frame)  # 显示图像
        k = cv2.waitKey(10)  # 10ms内等待输入

        if k == 27:  # ESC(ASCII码为27),按下ESC后 cv2.waitKey(),停止
            break

    # 释放摄像头
    cap.release()
    cv2.destroyAllWindows()
