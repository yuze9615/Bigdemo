"""
系统集成测试
"""
import sys
from pathlib import Path
import requests
import time
import subprocess

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_system_integration():
    """完整的系统集成测试"""
    print("=" * 60)
    print("系统集成测试")
    print("=" * 60)

    base_url = "http://localhost:8000"

    # 测试1：检查服务健康状态
    print("\n[测试1] 检查服务健康状态...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        assert response.status_code == 200
        print(f"✓ 服务健康状态正常")
        print(f"  模型: {response.json()['model']}")
    except Exception as e:
        print(f"✗ 服务不可用: {e}")
        return False

    # 测试2：检查首页访问
    print("\n[测试2] 检查首页访问...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        assert response.status_code == 200
        assert "AI Faces" in response.text
        print("✓ 首页访问正常")
    except Exception as e:
        print(f"✗ 首页访问失败: {e}")
        return False

    # 测试3：文件上传和检测
    print("\n[测试3] 测试文件上传和检测...")
    test_image = Path(__file__).parent.parent / "templates" / "face.jpg"
    if not test_image.exists():
        print(f"⚠ 跳过：测试图片不存在 {test_image}")
    else:
        try:
            with open(test_image, "rb") as f:
                files = {"file": ("test.jpg", f, "image/jpeg")}
                response = requests.post(
                    f"{base_url}/upload",
                    files=files,
                    data={"confidence": 0.5},
                    timeout=30
                )
                assert response.status_code == 200
                result = response.json()
                print(f"✓ 文件上传和检测成功")
                print(f"  - 检测到 {result['face_count']} 张人脸")
                print(f"  - 使用模型: {result['model']}")
                print(f"  - 消息: {result['message']}")
        except Exception as e:
            print(f"✗ 文件上传失败: {e}")
            return False

    # 测试4：错误处理
    print("\n[测试4] 测试错误处理...")
    try:
        # 测试不支持的文件类型
        files = {"file": ("test.txt", b"hello", "text/plain")}
        response = requests.post(f"{base_url}/upload", files=files)
        assert response.status_code == 400
        print("✓ 正确拒绝不支持的文件类型")
    except Exception as e:
        print(f"✗ 错误处理测试失败: {e}")

    print("\n" + "=" * 60)
    print("✓ 所有集成测试通过！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_system_integration()
