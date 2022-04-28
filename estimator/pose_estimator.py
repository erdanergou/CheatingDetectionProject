import cv2
import numpy as np
import torch

from utils.alphapose.transforms import heatmap_to_coord_simple_regress
from utils.simple_transform import SimpleTransform
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


class AlphaPoseEstimator:  # AlphaPose评价器
    empty_tensor = torch.tensor([])  # 空tensor

    def __init__(self, weights, device, input_size=(256, 192), output_size=(64, 48), face_aligner_weights=None):
        self._input_size = input_size  # 输入大小
        self._output_size = output_size  # 输出大小
        self._sigma = 2  # 标准差
        self.device = device  # 设备
        self.model = torch.jit.load(weights).to(device)  # 加载模型
        self.transformation = SimpleTransform(
            scale_factor=0,
            input_size=self._input_size,
            output_size=self._output_size,
            rot=0, sigma=self._sigma, add_dpg=False, gpu_device=self.device)
        # 热图是heatmap的直译，用暖色表示数值大，冷色表示数值小。行为基因，列为样本。
        self.heatmap_to_coord = heatmap_to_coord_simple_regress

    def preprocess(self, frame, detections):
        # 预处理，返回tensor类型图片和帧图片
        inps = []
        cropped_boxes = []  # 裁剪框
        for i in range(detections.shape[0]):  # 循环
            box = detections[i, :4]  # 获取框体范围
            inp, cropped_box = self.transformation.test_transform(frame, box)  # 获取输入和裁剪后的框体
            inps.append(torch.FloatTensor(inp))  # 转为32位浮点数并且加入到输入列表
            cropped_boxes.append(torch.FloatTensor(cropped_box))  # 转为32位浮点数并且加入到裁剪框列表
        return torch.stack(inps, dim=0).to(self.device), torch.stack(cropped_boxes, dim=0).to(self.device)

    def estimate(self, frame, detections):
        # 判断
        if detections.shape[0] <= 0:  # 如果维度小于等于零，则不存在，返回空
            return self.empty_tensor, self.empty_tensor
        inps, cropped_boxes = self.preprocess(frame, detections)  # 获取预处理后的数据，包括图片集和裁剪框
        # 关键点检测，获取数据
        # hm：torch.Size([1, 136, 64, 48])
        hm = self.model(inps).cpu().detach()  # 处理后的数据
        self.eval_joints = [*range(0, 136)]  # 评估点
        pose_coords = []  # 姿势坐标
        pose_scores = []  # 姿势成绩
        norm_type = 'sigmoid'  # 正常类型
        hm_size = self._output_size  # 维度设置为输出维度
        for i in range(hm.shape[0]):  # 循环所有最外维中所有数据
            bbox = cropped_boxes[i].tolist()  # 将裁剪后的框转为列表类型
            # 获取到坐标与成绩
            pose_coord, pose_score = self.heatmap_to_coord(hm[i][self.eval_joints], bbox, hm_shape=hm_size,
                                                           norm_type=norm_type)
            pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))  # 将获取到的数据添加到列表
            pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))  # 将获取到的数据添加到列表
        preds_kps = torch.cat(pose_coords)  # 把多个tensor进行拼接
        preds_scores = torch.cat(pose_scores)  # 把多个tensor进行拼接
        return preds_kps, preds_scores

    @staticmethod
    def get_face_boxes(keypoints):
        # 获取人脸框
        face_keypoints = keypoints[:, 26:94]  # 26-93, 68个人脸关键点
        face_outline_keypoints = face_keypoints[:, :27]  # 到27为头部轮廓
        x_min = torch.min(face_outline_keypoints[:, :, 0], dim=1).values  # 第一列最小值
        y_min = torch.min(face_outline_keypoints[:, :, 1], dim=1).values  # 第二列最小值
        x_max = torch.max(face_outline_keypoints[:, :, 0], dim=1).values  # 第一列最大值
        y_max = torch.max(face_outline_keypoints[:, :, 1], dim=1).values  # 第二列最大值
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)  # 连接成一维的tensor类型

    @staticmethod
    def get_face_keypoints(face_keypoints):
        """
        获取标准化后的人脸关键点坐标
        :param face_keypoints: 脸部关键点
        :return: 标准化后的人脸关键点坐标，人脸框的位置
        """
        face_outline_keypoints = face_keypoints[:27]
        face_x1 = torch.min(face_outline_keypoints[:, 0])
        face_y1 = torch.min(face_outline_keypoints[:, 1])
        face_x2 = torch.max(face_outline_keypoints[:, 0])
        face_y2 = torch.max(face_outline_keypoints[:, 1])
        # 获取标准化的脸部坐标
        face_x1_y1 = torch.tensor([face_x1, face_y1])
        face_width = torch.tensor([face_x2 - face_x1, face_y2 - face_y1])
        scaled_face_keypoints = (face_keypoints - face_x1_y1) / face_width
        return scaled_face_keypoints, (face_x1, face_y1, face_x2, face_y2)


class PnPPoseEstimator:
    """根据面部特征估计头部姿势"""

    def __init__(self, img_size=(480, 640)):
        self.size = img_size

        # 3D 模型点
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # 鼻尖
            (0.0, -330.0, -65.0),  # 下巴
            (-225.0, 170.0, -135.0),  # 左眼 左眼角
            (225.0, 170.0, -135.0),  # 右眼 右眼角
            (-150.0, -150.0, -125.0),  # 嘴角（左）
            (150.0, -150.0, -125.0)  # 嘴角（右）
        ]) / 4.5

        # 68个模型点
        self.model_points_68 = self._get_full_model_points()

        # 身体关键点
        self.body_model_points = np.array([  # 18,19,5,6
            (0.0, -40.0, 0.0),  # 脖子
            (0.0, 240.0, 5),  # 臀部
            (-80, 0.0, 0.5),  # 左肩
            (+80, 0.0, 0.5),  # 右肩
        ])

        # 脖子关键点
        self.neck_model_points = np.array([  # 17,18,5,6
            (0.0, -90.0, 5),  # 头步
            (0.0, 30.0, 0.0),  # 脖子
            (-60, 70.0, 5),  # 左肩
            (+60, 70.0, 5),  # 右肩
        ])

        # Camera internals 相机设置
        self.focal_length = self.size[1]  # 焦距
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)  # 相机中心
        self.camera_matrix = np.array(  # 摄像机矩阵
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # 假设没有镜头失真
        self.dist_coeefs = np.zeros((4, 1))

        # 旋转向量和平移向量
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])

    @staticmethod
    def _get_full_model_points():
        """从文件中获得所有68个3D模型点"""
        model_points = np.array([
            [-73.393523, -29.801432, -47.667532],
            [-72.775014, -10.949766, -45.909403],
            [-70.533638, 7.929818, -44.84258],
            [-66.850058, 26.07428, -43.141114],
            [-59.790187, 42.56439, -38.635298],
            [-48.368973, 56.48108, -30.750622],
            [-34.121101, 67.246992, -18.456453],
            [-17.875411, 75.056892, -3.609035],
            [0.098749, 77.061286, 0.881698],
            [17.477031, 74.758448, -5.181201],
            [32.648966, 66.929021, -19.176563],
            [46.372358, 56.311389, -30.77057],
            [57.34348, 42.419126, -37.628629],
            [64.388482, 25.45588, -40.886309],
            [68.212038, 6.990805, -42.281449],
            [70.486405, -11.666193, -44.142567],
            [71.375822, -30.365191, -47.140426],
            [-61.119406, -49.361602, -14.254422],
            [-51.287588, -58.769795, -7.268147],
            [-37.8048, -61.996155, -0.442051],
            [-24.022754, -61.033399, 6.606501],
            [-11.635713, -56.686759, 11.967398],
            [12.056636, -57.391033, 12.051204],
            [25.106256, -61.902186, 7.315098],
            [38.338588, -62.777713, 1.022953],
            [51.191007, -59.302347, -5.349435],
            [60.053851, -50.190255, -11.615746],
            [0.65394, -42.19379, 13.380835],
            [0.804809, -30.993721, 21.150853],
            [0.992204, -19.944596, 29.284036],
            [1.226783, -8.414541, 36.94806],
            [-14.772472, 2.598255, 20.132003],
            [-7.180239, 4.751589, 23.536684],
            [0.55592, 6.5629, 25.944448],
            [8.272499, 4.661005, 23.695741],
            [15.214351, 2.643046, 20.858157],
            [-46.04729, -37.471411, -7.037989],
            [-37.674688, -42.73051, -3.021217],
            [-27.883856, -42.711517, -1.353629],
            [-19.648268, -36.754742, 0.111088],
            [-28.272965, -35.134493, 0.147273],
            [-38.082418, -34.919043, -1.476612],
            [19.265868, -37.032306, 0.665746],
            [27.894191, -43.342445, -0.24766],
            [37.437529, -43.110822, -1.696435],
            [45.170805, -38.086515, -4.894163],
            [38.196454, -35.532024, -0.282961],
            [28.764989, -35.484289, 1.172675],
            [-28.916267, 28.612716, 2.24031],
            [-17.533194, 22.172187, 15.934335],
            [-6.68459, 19.029051, 22.611355],
            [0.381001, 20.721118, 23.748437],
            [8.375443, 19.03546, 22.721995],
            [18.876618, 22.394109, 15.610679],
            [28.794412, 28.079924, 3.217393],
            [19.057574, 36.298248, 14.987997],
            [8.956375, 39.634575, 22.554245],
            [0.381549, 40.395647, 23.591626],
            [-7.428895, 39.836405, 22.406106],
            [-18.160634, 36.677899, 15.121907],
            [-24.37749, 28.677771, 4.785684],
            [-6.897633, 25.475976, 20.893742],
            [0.340663, 26.014269, 22.220479],
            [8.444722, 25.326198, 21.02552],
            [24.474473, 28.323008, 5.712776],
            [8.449166, 30.596216, 20.671489],
            [0.205322, 31.408738, 21.90367],
            [-7.198266, 30.844876, 20.328022]])

        return model_points

    # 展示3d模型
    def show_3d_model(self):
        fig = pyplot.figure()  # 创建图形
        ax = Axes3D(fig)  # 画曲面图
        model_points = self.model_points_68  # 获得模型点
        x = model_points[:, 0]  # 获取第一列值
        y = model_points[:, 1]  # 获取第二列值
        z = model_points[:, 2]  # 获取第三列值

        ax.scatter(x, y, z)  # 绘制三维图
        pyplot.xlabel('x')  # x轴
        pyplot.ylabel('y')  # y轴
        pyplot.show()  # 展示图形

    def solve_pose(self, image_points):
        """
        从图像点解决姿势
        Return (rotation_vector, translation_vector) as pose.
        """
        # 断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况
        assert image_points.shape[0] == self.model_points_68.shape[0]
        """3D点和2D点应该是相同的数字"""
        # 这个函数返回旋转和平移向量，这些向量变换对象中表示的3D点
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68, image_points, self.camera_matrix, self.dist_coeefs)
        return rotation_vector, translation_vector

    def solve_body_pose(self, image_points):
        """
        从图像点解决身体姿势
        Return (rotation_vector, translation_vector) as pose.
        """
        assert image_points.shape[0] == self.body_model_points.shape[
            0], "3D points and 2D points should be of same number."
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.body_model_points, image_points, self.camera_matrix, self.dist_coeefs)
        return rotation_vector, translation_vector

    def solve_neck_pose(self, image_points):
        """
        从图像点解决脖子姿势
        Return (rotation_vector, translation_vector) as pose.
        """
        assert image_points.shape[0] == self.neck_model_points.shape[
            0], "3D points and 2D points should be of same number."
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.body_model_points, image_points, self.camera_matrix, self.dist_coeefs)
        return rotation_vector, translation_vector

    def solve_pose_by_68_points(self, image_points):
        """
        从所有68个图像点求解姿势
        Return (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68, image_points, self.camera_matrix, self.dist_coeefs)

        return rotation_vector, translation_vector

    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
        """绘制一个3D框来标注姿势"""
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # 映射到2d图像点
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # 画出所有的线
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)

    def draw_axis(self, img, R, t):
        # 绘制坐标轴
        points = np.float32(
            [[30, 0, 0], [0, 30, 0], [0, 0, 30], [0, 0, 0]]).reshape(-1, 3)

        axisPoints, _ = cv2.projectPoints(
            points, R, t, self.camera_matrix, self.dist_coeefs)  # 通过给定的内参数和外参数计算三维点投影到二维图像平面上的坐标

    def draw_axes(self, img, R, t):
        # 根据姿态估计绘制世界/对象坐标系统的坐标轴。
        cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeefs, R, t, 30)

    def get_pose_marks(self, marks):
        """准备好姿势估计"""
        pose_marks = [marks[30], marks[8], marks[36], marks[45], marks[48], marks[54]]
        return pose_marks

    @staticmethod
    def get_euler(rotation_vector, translation_vector):
        """
        此函数用于从旋转向量计算欧拉角
        :param translation_vector: 输入为偏移向量
        :param rotation_vector: 输入为旋转向量
        :return: 返回欧拉角在三个轴上的值
        """
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        yaw = eulerAngles[1]
        pitch = eulerAngles[0]
        roll = eulerAngles[2]
        rot_params = np.array([yaw, pitch, roll])
        return rot_params
