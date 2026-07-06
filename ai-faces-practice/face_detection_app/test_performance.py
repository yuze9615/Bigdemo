"""
性能测试脚本（纯标准库，无需 requests）
"""
"""“”“
异步测试
”“”"""
import urllib.request
import urllib.parse
import json
import time
import concurrent.futures
from pathlib import Path
import threading

BASE_URL = "http://127.0.0.1:8001"
TEST_IMAGE = Path("test_face.jpg")

def http_post(endpoint, fields=None, files=None):
    """使用 urllib 发送 multipart/form-data POST 请求"""
    fields = fields or {}
    files = files or {}
    
    boundary = '----WebKitFormBoundary' + str(int(time.time() * 1000))
    
    body = b''
    
    # 普通字段
    for key, value in fields.items():
        body += f'--{boundary}\r\n'.encode()
        body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
        body += f'{value}\r\n'.encode()
    
    # 文件字段
    for key, filepath in files.items():
        filepath = Path(filepath)
        filename = filepath.name
        content_type = 'image/jpeg' if filename.endswith(('.jpg', '.jpeg')) else 'image/png'
        
        body += f'--{boundary}\r\n'.encode()
        body += f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode()
        body += f'Content-Type: {content_type}\r\n\r\n'.encode()
        body += filepath.read_bytes()
        body += b'\r\n'
    
    body += f'--{boundary}--\r\n'.encode()
    
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=body,
        headers={
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'Content-Length': str(len(body))
        },
        method='POST'
    )
    
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return resp.status, time.time() - start, data
    except urllib.error.HTTPError as e:
        return e.code, time.time() - start, None

def test_single(endpoint, **kwargs):
    """单次请求测试"""
    files = {"file": TEST_IMAGE}
    fields = {k: str(v) for k, v in kwargs.items()}
    return http_post(endpoint, fields=fields, files=files)

def test_concurrent(endpoint, count=10, workers=5, **kwargs):
    """并发测试"""
    print(f"\n{'='*50}")
    print(f"并发测试: {endpoint} | 请求数: {count} | 并发: {workers}")
    print(f"{'='*50}")
    
    start = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(test_single, endpoint, **kwargs) for _ in range(count)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    total_time = time.time() - start
    
    success = sum(1 for s, _, _ in results if s == 200)
    durations = [d for s, d, _ in results if s == 200]
    
    print(f"总耗时: {total_time:.3f}s")
    print(f"成功: {success}/{count}")
    if durations:
        print(f"平均耗时: {sum(durations)/len(durations):.3f}s")
        print(f"最快: {min(durations):.3f}s")
        print(f"最慢: {max(durations):.3f}s")
    print(f"QPS: {count/total_time:.1f}")

if __name__ == "__main__":
    if not TEST_IMAGE.exists():
        print(f"请准备测试图片: {TEST_IMAGE}")
        print("可用 OpenCV 自带图片: python -c \"import cv2, shutil; shutil.copy(cv2.__file__.replace('__init__.py','data/lena.jpg'), 'test_face.jpg')\"")
        exit(1)
    
    # 1. 单次检测测试
    print("="*50)
    print("单次检测测试")
    print("="*50)
    code, duration, data = test_single("/detect", confidence=0.5, optimize=True)
    print(f"状态: {code}, 耗时: {duration:.3f}s")
    if data:
        print(f"检测到 {data.get('face_count')} 张人脸")
        print(f"处理时间: {data.get('process_time')}s")
        print(f"图片优化: {data.get('optimized')}")
    
    # 2. 并发检测测试
    test_concurrent("/detect", count=20, workers=10, confidence=0.5, optimize=True)
    
    # 3. 并发上传测试
    test_concurrent("/upload", count=10, workers=5, confidence=0.5, compress=True)
    
    # 4. 查看队列统计
    print("\n" + "="*50)
    print("队列统计")
    print("="*50)
    try:
        with urllib.request.urlopen(f"{BASE_URL}/queue/stats", timeout=5) as resp:
            stats = json.loads(resp.read().decode('utf-8'))
            print(json.dumps(stats, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"获取统计失败: {e}")