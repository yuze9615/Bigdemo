"""
图片压缩和优化模块（纯 OpenCV 实现）
支持：尺寸压缩、质量压缩、格式优化、内存处理
无需 Pillow 依赖
"""
import cv2
import numpy as np
from pathlib import Path
import io


class ImageOptimizer:
    """图片优化器 - 纯 OpenCV 实现"""
    
    # 默认配置
    DEFAULT_MAX_SIZE = (1920, 1080)      # 最大尺寸
    DEFAULT_MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB
    DEFAULT_QUALITY = 85                  # JPEG质量

    def __init__(self, 
                 max_size=None,
                 max_file_size=None,
                 quality=None):
        """
        初始化优化器
        Args:
            max_size: 最大尺寸 (宽, 高)
            max_file_size: 最大文件大小（字节）
            quality: JPEG质量 (1-100)
        """
        self.max_size = max_size or self.DEFAULT_MAX_SIZE
        self.max_file_size = max_file_size or self.DEFAULT_MAX_FILE_SIZE
        self.quality = quality or self.DEFAULT_QUALITY

    def optimize(self, image_input, target_format='jpeg'):
        """
        优化图片
        Args:
            image_input: 图片路径、bytes或ndarray
            target_format: 目标格式 'jpeg', 'png'
        Returns:
            bytes: 优化后的图片数据
        """
        # 统一加载为 OpenCV ndarray
        img = self._load_image(image_input)
        if img is None:
            return None

        # 1. 尺寸压缩
        img = self._resize_image(img)

        # 2. 格式优化和质量压缩
        optimized = self._compress_image(img, target_format)

        # 3. 如果仍然超过大小限制，进一步压缩
        if len(optimized) > self.max_file_size:
            optimized = self._aggressive_compress(img, target_format)

        return optimized

    def optimize_for_detection(self, image_input, max_dimension=1280):
        """
        专门为人脸检测优化的压缩
        - 保持足够分辨率用于检测
        - 减少内存占用和传输时间
        """
        img = self._load_image(image_input)
        if img is None:
            return None

        h, w = img.shape[:2]
        max_edge = max(w, h)
        
        if max_edge > max_dimension:
            scale = max_dimension / max_edge
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        return img

    def _load_image(self, image_input):
        """统一加载图片为 OpenCV ndarray"""
        if isinstance(image_input, (str, Path)):
            # 文件路径
            return cv2.imread(str(image_input), cv2.IMREAD_COLOR)
        elif isinstance(image_input, bytes):
            # 二进制数据
            arr = np.frombuffer(image_input, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            # 已经是 OpenCV ndarray
            return image_input.copy()
        return None

    def _resize_image(self, img):
        """尺寸压缩"""
        h, w = img.shape[:2]
        max_w, max_h = self.max_size

        if w <= max_w and h <= max_h:
            return img

        # 保持宽高比
        ratio = min(max_w / w, max_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)

        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    def _compress_image(self, img, target_format):
        """质量压缩"""
        # 确定编码格式
        if target_format.lower() in ['jpeg', 'jpg']:
            ext = '.jpg'
            # OpenCV 编码参数: [cv2.IMWRITE_JPEG_QUALITY, quality]
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        elif target_format.lower() == 'png':
            ext = '.png'
            # PNG 压缩级别 0-9，3是平衡速度和压缩率
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        else:
            ext = '.jpg'
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]

        # 编码为字节
        success, buffer = cv2.imencode(ext, img, encode_params)
        if not success:
            raise RuntimeError("图片编码失败")

        return buffer.tobytes()

    def _aggressive_compress(self, img, target_format):
        """激进压缩（当文件仍然过大时）"""
        h, w = img.shape[:2]
        
        # 逐步降低质量
        qualities = [70, 50, 30]
        
        for q in qualities:
            if target_format.lower() in ['jpeg', 'jpg']:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, q]
            else:
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
            
            success, buffer = cv2.imencode('.jpg', img, encode_params)
            if success and len(buffer) <= self.max_file_size:
                return buffer.tobytes()

        # 如果还是太大，进一步缩小尺寸
        img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LANCZOS4)
        return self._compress_image(img, target_format)

    @staticmethod
    def get_image_info(image_input):
        """获取图片信息"""
        img = ImageOptimizer()._load_image(image_input)
        if img is None:
            return None
        
        h, w = img.shape[:2]
        
        # 估算文件大小
        success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        size_bytes = len(buffer) if success else 0
        
        return {
            'width': w,
            'height': h,
            'channels': img.shape[2] if len(img.shape) > 2 else 1,
            'estimated_size': size_bytes,
            'estimated_size_mb': round(size_bytes / 1024 / 1024, 2)
        }


# ========== 便捷函数 ==========

def compress_image(image_input, max_size=(1280, 720), quality=85):
    """
    快速压缩图片
    """
    optimizer = ImageOptimizer(max_size=max_size, quality=quality)
    return optimizer.optimize(image_input, target_format='jpeg')


def optimize_for_api(image_input, max_dimension=1280):
    """
    为API传输优化图片
    """
    optimizer = ImageOptimizer()
    return optimizer.optimize_for_detection(image_input, max_dimension)