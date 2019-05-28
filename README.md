# FaceRecognition_opencv-Keras-Tensorflow
基于opencv ，Keras, Tensorflow的人脸识别
### 基于keras+tensorflow+opencv的简易人脸识别使用说明
>github 下载
```
git clone https://github.com/weiyangtang/FaceRecognition_opencv-Keras-Tensorflow.git
```

#### 一、安装模块
1. 安装好Anaconda，否则很多科学计算的库要手动安装很麻烦
2. 安装好TensorFlow,下面是最简便的方法（但是安装的是cpu版本，但速度相对于GPU版的稍微慢点）
```python
# 安装TensorFlow cpu版
pip install tensorflow 
# 安装keras框架
pip install keras
# 安装opencv 
pip install opencv-python
```
#### 二、项目结构使用
cv2_data： opencv识别的分类器
faceData：存放脸部图片

#### 三、使用说明
1. 运行face_capture.py,拍摄脸部照片 ，记得把main函数中的人名改成的你的
2. 运行model_train.py，训练模型
3. camera_reader.py 识别人脸的入口
**建议开始是你把faceData图片删除掉，否则会影响识别的准确度**
