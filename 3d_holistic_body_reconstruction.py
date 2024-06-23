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

# 初始化Mediapipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=True, model_complexity=2)

# 处理图像
results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# 提取3D点云和颜色信息
def extract_points(landmarks, scale_z=1000):
    points_3d = []
    points_2d = []
    colors = []
    if landmarks:
        for i, landmark in enumerate(landmarks.landmark):
            # 获取3D坐标
            x = landmark.x * image.shape[1]
            y = landmark.y * image.shape[0]
            z = landmark.z * scale_z  # 可能需要调整z的比例

            # 边界检查
            if x >= image.shape[1] or y >= image.shape[0] or x < 0 or y < 0:
                continue

            points_3d.append((x, y, z))
            points_2d.append((landmark.x, landmark.y))  # 归一化的2D坐标
            # 获取颜色信息
            b, g, r = image[int(y), int(x)]
            colors.append((r / 255.0, g / 255.0, b / 255.0))
    return points_3d, points_2d, colors

# 获取不同部分的点云数据
face_points_3d, face_points_2d, face_colors = extract_points(results.face_landmarks)
body_points_3d, body_points_2d, body_colors = extract_points(results.pose_landmarks)
left_hand_points_3d, left_hand_points_2d, left_hand_colors = extract_points(results.left_hand_landmarks)
right_hand_points_3d, right_hand_points_2d, right_hand_colors = extract_points(results.right_hand_landmarks)

# 创建Open3D点云对象
def create_point_cloud(points_3d, colors):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

face_point_cloud = create_point_cloud(face_points_3d, face_colors)
body_point_cloud = create_point_cloud(body_points_3d, body_colors)
left_hand_point_cloud = create_point_cloud(left_hand_points_3d, left_hand_colors)
right_hand_point_cloud = create_point_cloud(right_hand_points_3d, right_hand_colors)

# 分别处理脸部和身体
def create_mesh(point_cloud):
    points = np.asarray(point_cloud.points)
    if len(points) == 0:
        return None

    tri = Delaunay(points[:, :2])
    # 反转三角形顶点顺序以改变法线方向
    triangles = tri.simplices
    triangles = np.array([triangle[::-1] for triangle in triangles])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = point_cloud.colors
    return mesh

face_mesh = create_mesh(face_point_cloud)
body_mesh = create_mesh(body_point_cloud)
left_hand_mesh = create_mesh(left_hand_point_cloud)
right_hand_mesh = create_mesh(right_hand_point_cloud)

# 创建可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window()

# 添加网格到可视化窗口
if face_mesh:
    vis.add_geometry(face_mesh)
if body_mesh:
    vis.add_geometry(body_mesh)
if left_hand_mesh:
    vis.add_geometry(left_hand_mesh)
if right_hand_mesh:
    vis.add_geometry(right_hand_mesh)

# 设置相机参数，使其正对鼻尖
nose_tip = None
if results.face_landmarks:
    nose_tip = np.array([results.face_landmarks.landmark[1].x * image.shape[1],
                         results.face_landmarks.landmark[1].y * image.shape[0],
                         results.face_landmarks.landmark[1].z * 1000])

if nose_tip is not None:
    ctr = vis.get_view_control()
    cam_position = nose_tip + np.array([0, 0, -1000])
    ctr.set_lookat(nose_tip)
    ctr.set_front((cam_position - nose_tip) / np.linalg.norm(cam_position - nose_tip))
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.5)  # 调整缩放以确保整个面部在视图中

# 可视化
vis.run()
vis.destroy_window()