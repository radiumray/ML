# 自动驾驶

## 放视频

## 常用传感器讲解（传感器）
    激光雷达
    摄像头 *
    深度摄像头
    毫米波雷达

## SLAM/人工智能处理（感知层）
    CNN *
    SLAM
    LaneNet
    TinyYolo

    tensorflow
    Keras

## 路径规划（决策层）
    https://github.com/AtsushiSakai/PythonRobotics

## 控制驱动（控制层）
    控制舵机 *
    控制电机 *
    CAN总线


# 可尝试项目
    donkeycar
    
    Udacity课程 https://github.com/ndrplz/self-driving-car

    colorFilter.py
    cannyEdgy.py
    HoughTransFindLine.py

    laneNet
    yoloV3

# donkeycar
    端到端的神经网络

    训练步骤：
        遥控采集数据并存储到本地  python manage.py drive --js

        找到数据，并用u盘把数据转移到训练机

        人工检查数据是否有问题

        训练机运行一个端到端的模型，根据采集到的数据训练 python manage.py train --tub ./tub/[yourData] --model ./models/[yourModule.h5]

        采集数据的好坏直接影响到了训练效果

        采集数据时遥控器的摇杆要产生一个连续的值

        训练出的模型再转移到小车上

        运行模型，启动网页服务，控制小车速度 python manage.py drive --model ./models/[yourModule.h5]

    优点：模型小，可在树莓派中运行
    缺点：由于用一个模型解决，调试困难




# 实际道路方案
    laneNet+tinyYolo
    opencv轮廓，腐蚀膨胀，立透视


