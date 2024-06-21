import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 仅显示错误消息

import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

# 加载图像
image_path = 'S__10092604.jpg'  # 将此路径改为您的图片路径
image = cv2.imread(image_path)

# 初始化Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# 处理图像
results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# 提取3D点云和颜色信息
points_3d = []
points_2d = []
colors = []
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for landmark in face_landmarks.landmark:
            # 获取3D坐标
            x = landmark.x * image.shape[1]
            y = landmark.y * image.shape[0]
            z = landmark.z * 1000  # 可能需要调整z的比例
            points_3d.append((x, y, z))
            points_2d.append((x, y))
            # 获取颜色信息
            b, g, r = image[int(y), int(x)]
            colors.append((r / 255.0, g / 255.0, b / 255.0))

# 转换为NumPy数组
points_3d = np.array(points_3d)
points_2d = np.array(points_2d)
colors = np.array(colors)

# 创建Delaunay三角化
tri = Delaunay(points_2d)

# 创建Open3D网格对象
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(points_3d)
mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)

# 为网格添加颜色
mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

# 创建纹理映射
texture_image = o3d.geometry.Image(image)
mesh.textures = [texture_image]

# 可视化
o3d.visualization.draw_geometries([mesh])
