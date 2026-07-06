"""
人脸检测核心类
实现基于OpenCV DNN的人脸检测功能
"""
import cv2
import numpy as np
from pathlib import Path
import urllib.request
import threading


class FaceDetector:
    """人脸检测器类 - 支持模型缓存和预热"""
    
    # 类级别缓存：所有实例共享同一个模型
    _model_cache = {}
    _cache_lock = threading.Lock()

    def __init__(self, use_dnn=True, cache_model=True):
        """
        初始化人脸检测器
        Args:
            use_dnn: 是否使用DNN模型，False则使用Haar级联
            cache_model: 是否启用模型缓存（默认True）
        """
        self.use_dnn = use_dnn
        self._cache_model = cache_model
        
        if cache_model:
            self._init_with_cache()
        else:
            if use_dnn:
                self._init_dnn_model()
            else:
                self._init_haar_cascade()

    def _init_with_cache(self):
        """使用缓存初始化模型"""
        cache_key = f"dnn_{self.use_dnn}"
        
        with FaceDetector._cache_lock:
            if cache_key in FaceDetector._model_cache:
                # 从缓存加载
                cached = FaceDetector._model_cache[cache_key]
                if self.use_dnn:
                    self.net = cached
                    print("[INFO] DNN模型从缓存加载")
                else:
                    self.face_cascade = cached
                    print("[INFO] Haar模型从缓存加载")
                return
        
        # 缓存未命中，初始化并缓存
        if self.use_dnn:
            self._init_dnn_model()
            with FaceDetector._cache_lock:
                FaceDetector._model_cache[cache_key] = self.net
        else:
            self._init_haar_cascade()
            with FaceDetector._cache_lock:
                FaceDetector._cache_cache[cache_key] = self.face_cascade

    def _init_haar_cascade(self):
        """初始化Haar级联分类器（传统方法）"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("无法加载Haar级联分类器")
        print("[INFO] Haar级联分类器加载成功")

    def _init_dnn_model(self):
        """初始化DNN深度学习模型"""
        # 模型路径
        model_dir = Path(__file__).parent / 'models'
        model_dir.mkdir(exist_ok=True)
        prototxt_path = model_dir / 'deploy.prototxt'
        caffemodel_path = model_dir / 'res10_300x300_ssd_iter_140000_fp16.caffemodel'

        # 检查模型文件是否存在
        if not prototxt_path.exists() or not caffemodel_path.exists():
            print("[INFO] 正在下载DNN模型文件...")
            self._download_models(prototxt_path, caffemodel_path)

        # 加载模型
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))
        print("[INFO] DNN人脸检测模型加载成功")

    def _download_models(self, prototxt_path, caffemodel_path):
        """下载模型文件"""
        prototxt_url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
        caffemodel_url = 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel'

        try:
            print("[INFO] 下载模型配置文件...")
            urllib.request.urlretrieve(prototxt_url, str(prototxt_path))
            print("[INFO] 下载模型权重文件（约5MB）...")
            urllib.request.urlretrieve(caffemodel_url, str(caffemodel_path))
            print("[INFO] 模型文件下载完成")
        except Exception as e:
            print(f"[ERROR] 模型下载失败: {e}")
            print("[INFO] 切换到Haar级联分类器")
            self.use_dnn = False
            self._init_haar_cascade()

    @classmethod
    def warm_up(cls, use_dnn=True):
        """
        模型预热：创建实例并执行一次前向传播
        在应用启动时调用，避免第一次请求时冷启动延迟
        """
        print(f"[INFO] 开始模型预热 (DNN={use_dnn})...")
        detector = cls(use_dnn=use_dnn, cache_model=True)
        
        # 创建一张空白图片进行预热推理
        dummy_image = np.zeros((300, 300, 3), dtype=np.uint8)
        
        if use_dnn and hasattr(detector, 'net'):
            blob = cv2.dnn.blobFromImage(dummy_image, 1.0, (300, 300), (104.0, 177.0, 123.0))
            detector.net.setInput(blob)
            _ = detector.net.forward()
            print("[INFO] DNN模型预热完成")
        elif not use_dnn and hasattr(detector, 'face_cascade'):
            gray = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2GRAY)
            _ = detector.face_cascade.detectMultiScale(gray, 1.1, 4)
            print("[INFO] Haar模型预热完成")
        
        return detector

    @classmethod
    def clear_cache(cls):
        """清理模型缓存"""
        with cls._cache_lock:
            cls._model_cache.clear()
            print("[INFO] 模型缓存已清理")

    def detect_faces(self, image_path, confidence_threshold=0.5):
        """
        检测图片中的人脸
        Args:
            image_path: 图片路径
            confidence_threshold: 置信度阈值（仅DNN模型有效）
        Returns:
            tuple: (人脸数量, 人脸位置列表, 原始图片, 置信度列表)
        """
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 根据模型选择检测方法
        if self.use_dnn:
            return self._detect_faces_dnn(image, confidence_threshold)
        else:
            return self._detect_faces_haar(image)

    def _detect_faces_dnn(self, image, confidence_threshold):
        """使用DNN模型检测人脸"""
        h, w = image.shape[:2]

        # 预处理：创建blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        # 前向传播
        self.net.setInput(blob)
        detections = self.net.forward()

        # 解析检测结果
        faces = []
        confidences = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                # 解析边界框
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                # 确保坐标在图片范围内
                x = max(0, x1)
                y = max(0, y1)
                width = x2 - x
                height = y2 - y

                faces.append([x, y, width, height])
                confidences.append(float(confidence))

        faces = np.array(faces) if faces else np.array([])
        return len(faces), faces, image, confidences

    def _detect_faces_haar(self, image):
        """使用Haar级联检测人脸"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(20, 20)
        )
        return len(faces), faces, image, []

    def mark_faces(self, image, faces, confidences=None, face_names=None):
        """
        在图片上标记人脸
        Args:
            image: 原始图片
            faces: 人脸位置列表
            confidences: 置信度列表
            face_names: 人脸名称列表（可选，用于识别结果显示）
        Returns:
            ndarray: 标记后的图片
        """
        marked_image = image.copy()
        img_h, img_w = image.shape[:2]

        for i, (x, y, w, h) in enumerate(faces, 1):
            # 扩展 35% 边界，确保框住完整脸部
            pad_x = int(w * 0.35)
            pad_y = int(h * 0.35)
            adj_x = max(0, x - pad_x)
            adj_y = max(0, y - pad_y)
            adj_w = min(img_w - adj_x, w + pad_x * 2)
            adj_h = min(img_h - adj_y, h + pad_y * 2)

            # 动态线条粗细：人脸越大线条越粗
            thickness = max(2, min(5, int((adj_w + adj_h) / 150)))

            # 动态颜色：根据索引循环使用不同颜色
            colors = [
                (0, 255, 0),    # 绿色
                (255, 0, 0),    # 蓝色
                (0, 0, 255),    # 红色
                (0, 255, 255),  # 黄色
                (255, 0, 255),  # 紫色
            ]
            color = colors[(i - 1) % len(colors)]

            # 绘制矩形框
            cv2.rectangle(
                marked_image,
                (adj_x, adj_y),
                (adj_x + adj_w, adj_y + adj_h),
                color,
                thickness
            )

            # 生成标签
            if face_names and i <= len(face_names):
                name = face_names[i - 1]
                label = f"{name}"
            elif confidences and i <= len(confidences):
                conf = confidences[i - 1]
                label = f"Face {i} ({conf * 100:.1f}%)"
            else:
                label = f"Face {i}"

            # 计算标签位置
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.4, min(0.9, (adj_w + adj_h) / 400))
            text_thickness = max(1, min(2, int((adj_w + adj_h) / 300)))

            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
            label_y = max(adj_y - 5, text_h + 10)  # 确保标签在图片内

            # 绘制标签背景（半透明黑色）
            bg_x1 = adj_x
            bg_y1 = label_y - text_h - 6
            bg_x2 = adj_x + text_w + 10
            bg_y2 = label_y + 4

            # 确保背景不超出图片
            bg_x2 = min(bg_x2, img_w)
            bg_y1 = max(bg_y1, 0)

            overlay = marked_image.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            # 混合透明度
            alpha = 0.7
            marked_image[bg_y1:bg_y2, bg_x1:bg_x2] = cv2.addWeighted(
                marked_image[bg_y1:bg_y2, bg_x1:bg_x2],
                1 - alpha,
                overlay[bg_y1:bg_y2, bg_x1:bg_x2],
                alpha,
                0
            )

            # 绘制标签文字（白色）
            cv2.putText(
                marked_image,
                label,
                (adj_x + 5, label_y),
                font,
                font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA
            )

        return marked_image

    def process_image(self, input_path, output_path, confidence_threshold=0.5):
        """
        处理图片：检测并标记人脸
        Args:
            input_path: 输入图片路径
            output_path: 输出图片路径
            confidence_threshold: 置信度阈值
        Returns:
            tuple: (人脸数量, 输出文件路径)
        """
        # 检测人脸
        if self.use_dnn:
            face_count, faces, image, confidences = self.detect_faces(
                input_path,
                confidence_threshold
            )
        else:
            face_count, faces, image, confidences = self.detect_faces(input_path)

        # 标记人脸
        if face_count > 0:
            marked_image = self.mark_faces(image, faces, confidences)
        else:
            marked_image = image

        # 保存结果
        cv2.imwrite(str(output_path), marked_image)
        return face_count, output_path