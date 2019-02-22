# -*- coding:utf-8 -*-
import cv2

from model_train import Model
from data_input import read_name_list

faceData_file_path = './faceData/'
# 脸部图片文件路径

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    # 读取dataset数据集下的子文件夹名称
    name_list = read_name_list(faceData_file_path)
    cascade_path = "F:\Anaconda\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
    model = Model()
    model.load()
    while True:
        _, frame = cap.read()

        # グレースケール変換
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # カスケード分類器の特徴量を取得する
        cascade = cv2.CascadeClassifier(cascade_path)

        # 物体認識（顔認識）の実行
        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
        # facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.01, minNeighbors=3, minSize=(3, 3))
        if len(facerect) > 0:  # 检测到人脸
            print(name_list)
            print('face detected')
            color = (0, 0, 255)  # opencv B-G-R 红色
            for rect in facerect:
                # 検出した顔を囲む矩形の作成
                # cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=2)

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
        c = cv2.waitKey(10)
        # 10msecキー入力待ち
        k = cv2.waitKey(100)
        # Escキーを押されたら終了
        if k == 27:  # ESC(ASCII码为27),按下ESC后 cv2.waitKey(),停止
            break

    # キャプチャを終了
    cap.release()
    cv2.destroyAllWindows()
