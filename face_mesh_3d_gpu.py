import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 仅显示错误消息

import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
import cupy as cp
import tensorflow as tf

# 加载图像
image_path = 'S__10092604.jpg'  # 将此路径改为您的图片路径
image = cv2.imread(image_path)

# 将图像转换为TensorFlow张量
image_tf = tf.convert_to_tensor(image)

# 初始化Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# 处理图像
results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# 提取3D点云和颜色信息
points_3d = []
points_2d = []
colors = []
nose_tip = None  # 鼻尖位置
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for i, landmark in enumerate(face_landmarks.landmark):
            # 获取3D坐标
            x = landmark.x * image.shape[1]
            y = landmark.y * image.shape[0]
            z = landmark.z * 1000  # 可能需要调整z的比例
            points_3d.append((x, y, z))
            points_2d.append((x, y))
            # 获取颜色信息
            b, g, r = image[int(y), int(x)]
            colors.append((r / 255.0, g / 255.0, b / 255.0))
            # 获取鼻尖位置
            if i == 1:  # 通常鼻尖索引为1或4，可以根据具体模型调整
                nose_tip = np.array([x, y, z])

# 转换为CuPy数组
points_3d_cp = cp.array(points_3d)
points_2d_cp = cp.array(points_2d)
colors_cp = cp.array(colors)

# 创建Delaunay三角化
tri = Delaunay(cp.asnumpy(points_2d_cp))

# 反转三角形顶点顺序以改变法线方向
triangles = tri.simplices
triangles = cp.array([triangle[::-1] for triangle in triangles])

# 创建Open3D网格对象
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(cp.asnumpy(points_3d_cp))
mesh.triangles = o3d.utility.Vector3iVector(cp.asnumpy(triangles))

# 为网格添加颜色
mesh.vertex_colors = o3d.utility.Vector3dVector(cp.asnumpy(colors_cp))

# 创建纹理映射
texture_image = o3d.geometry.Image(image)
mesh.textures = [texture_image]

# 创建可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window()

# 添加网格到可视化窗口
vis.add_geometry(mesh)

# 设置相机参数，使其正对鼻尖
if nose_tip is not None:
    ctr = vis.get_view_control()
    
    # 设置相机位置和观察方向
    cam_position = nose_tip + np.array([0, 0, -1000])
    ctr.set_lookat(nose_tip)
    ctr.set_front((cam_position - nose_tip) / np.linalg.norm(cam_position - nose_tip))
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.5)  # 调整缩放以确保整个面部在视图中

# 可视化
vis.run()
vis.destroy_window()
