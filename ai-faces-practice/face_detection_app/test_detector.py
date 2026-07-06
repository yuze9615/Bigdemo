"""
人脸检测器测试代码
"""
import sys
from pathlib import Path
import cv2

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))
from face_detector import FaceDetector


def test_face_detector():
    """测试人脸检测器"""
    print("=" * 60)
    print("人脸检测器测试")
    print("=" * 60)

    # 创建检测器实例
    detector = FaceDetector()
    print(f"\n使用的模型: {'DNN (SSD)' if detector.use_dnn else 'Haar Cascade'}")

    # 测试检测功能
    test_image = Path("templates/face.jpg")
    if not test_image.exists():
        print(f"\n[ERROR] 测试图片不存在: {test_image}")
        print("请准备一张包含人脸的测试图片")
        return

    # 执行检测
    print(f"\n正在检测图片: {test_image}")
    face_count, faces, image, confidences = detector.detect_faces(test_image)

    print(f"\n检测结果:")
    print(f" - 检测到 {face_count} 张人脸")

    if face_count > 0:
        print(f" - 各人脸置信度:")
        for i, conf in enumerate(confidences, 1):
            print(f"   Face {i}: {conf * 100:.1f}%")

    # 保存标记结果
    output_path = Path("static/images/result_test.jpg")
    output_path.parent.mkdir(exist_ok=True)
    marked_image = detector.mark_faces(image, faces, confidences)
    cv2.imwrite(str(output_path), marked_image)
    print(f"\n标记结果已保存: {output_path}")


if __name__ == "__main__":
    test_face_detector()
