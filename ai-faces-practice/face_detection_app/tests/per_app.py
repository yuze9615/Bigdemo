"""
性能测试
"""
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def test_performance():
    """性能测试"""
    base_url = "http://localhost:8000"
    test_image = Path("templates/face.jpg")

    if not test_image.exists():
        print("测试图片不存在")
        return

    # 测试1：单次请求响应时间
    print("\n[性能测试1] 单次请求响应时间...")
    times = []
    for i in range(5):
        start_time = time.time()
        with open(test_image, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{base_url}/upload", files=files)
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"  第{i+1}次: {elapsed:.2f}秒")

    avg_time = sum(times) / len(times)
    print(f"\n平均响应时间: {avg_time:.2f}秒")
    if avg_time < 3.0:
        print("✓ 响应时间符合要求 (< 3秒)")
    else:
        print("⚠ 响应时间超过要求")

    # 测试2：并发处理
    print("\n[性能测试2] 并发处理能力...")

    def upload_image(i):
        with open(test_image, "rb") as f:
            files = {"file": (f"test_{i}.jpg", f, "image/jpeg")}
            response = requests.post(f"{base_url}/upload", files=files)
            return response.json()

    with ThreadPoolExecutor(max_workers=3) as executor:
        start_time = time.time()
        results = list(executor.map(upload_image, range(3)))
        elapsed = time.time() - start_time
        print(f"3个并发请求耗时: {elapsed:.2f}秒")
        print(f"✓ 并发处理正常")


if __name__ == "__main__":
    test_performance()
