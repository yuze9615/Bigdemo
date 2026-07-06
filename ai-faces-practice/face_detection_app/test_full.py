"""
AI Faces - 人脸识别服务完整测试脚本
使用 Python 标准库，零额外依赖
"""

import urllib.request
import urllib.parse
import json
import time
import concurrent.futures
from pathlib import Path
import sys


# ==================== 配置 ====================

BASE_URL = "http://127.0.0.1:8001"
TEST_IMAGE = Path("test_face.jpg")


# ==================== HTTP 工具 ====================

def build_multipart_body(fields, files):
    """
    构建 multipart/form-data 请求体
    fields: dict {name: value}
    files: dict {name: filepath}
    """
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
        content_type = 'image/jpeg' if filename.lower().endswith(('.jpg', '.jpeg')) else 'image/png'
        
        body += f'--{boundary}\r\n'.encode()
        body += f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode()
        body += f'Content-Type: {content_type}\r\n\r\n'.encode()
        body += filepath.read_bytes()
        body += b'\r\n'
    
    body += f'--{boundary}--\r\n'.encode()
    
    return body, boundary


def http_post(endpoint, fields=None, files=None, timeout=30):
    """
    发送 POST 请求
    """
    fields = fields or {}
    files = files or {}
    
    body, boundary = build_multipart_body(fields, files)
    
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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return resp.status, time.time() - start, data
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        try:
            error_data = json.loads(error_body)
        except:
            error_data = {"detail": error_body}
        return e.code, time.time() - start, error_data
    except Exception as e:
        return 0, time.time() - start, {"error": str(e)}


def http_get(endpoint, timeout=10):
    """
    发送 GET 请求
    """
    start = time.time()
    try:
        with urllib.request.urlopen(f"{BASE_URL}{endpoint}", timeout=timeout) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return resp.status, time.time() - start, data
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        try:
            error_data = json.loads(error_body)
        except:
            error_data = {"detail": error_body}
        return e.code, time.time() - start, error_data
    except Exception as e:
        return 0, time.time() - start, {"error": str(e)}


# ==================== 打印工具 ====================

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_json(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def print_result(status, duration, data, success_key="success"):
    color_ok = "\033[92m"  # 绿色
    color_err = "\033[91m"  # 红色
    color_reset = "\033[0m"
    
    is_success = status == 200 and data.get(success_key, False) if isinstance(data, dict) else status == 200
    
    if is_success:
        print(f"{color_ok}[✓] 状态: {status}, 耗时: {duration:.3f}s{color_reset}")
    else:
        print(f"{color_err}[✗] 状态: {status}, 耗时: {duration:.3f}s{color_reset}")
    
    print_json(data)


# ==================== 测试用例 ====================

def test_health():
    """测试 1: 健康检查"""
    print_header("测试 1: 健康检查")
    status, duration, data = http_get("/health")
    print_result(status, duration, data, "status")
    return status == 200


def test_detect_single():
    """测试 2: 单张人脸检测"""
    print_header("测试 2: 单张人脸检测 (/detect)")
    
    if not TEST_IMAGE.exists():
        print(f"[✗] 测试图片不存在: {TEST_IMAGE}")
        return False
    
    status, duration, data = http_post(
        "/detect",
        fields={"confidence": "0.5", "optimize": "true"},
        files={"file": TEST_IMAGE}
    )
    print_result(status, duration, data)
    
    if status == 200 and data.get("face_count", 0) > 0:
        faces = data.get("faces", [])
        print(f"\n  检测到 {len(faces)} 张人脸:")
        for i, face in enumerate(faces, 1):
            print(f"    Face {i}: 位置({face['x']}, {face['y']}) 大小({face['w']}x{face['h']}) 置信度{face['confidence']:.2f}")
    
    return status == 200


def test_upload():
    """测试 3: 上传图片并检测"""
    print_header("测试 3: 上传图片并检测 (/upload)")
    
    if not TEST_IMAGE.exists():
        print(f"[✗] 测试图片不存在: {TEST_IMAGE}")
        return False
    
    status, duration, data = http_post(
        "/upload",
        fields={"confidence": "0.5", "compress": "true"},
        files={"file": TEST_IMAGE}
    )
    print_result(status, duration, data)
    
    if status == 200 and data.get("result_filename"):
        result_url = f"{BASE_URL}/result/{data['result_filename']}"
        print(f"\n  结果图片: {result_url}")
    
    return status == 200


def test_register(name="测试人员"):
    """测试 4: 注册人脸"""
    print_header(f"测试 4: 注册人脸 (/register) - 姓名: {name}")
    
    if not TEST_IMAGE.exists():
        print(f"[✗] 测试图片不存在: {TEST_IMAGE}")
        return False
    
    status, duration, data = http_post(
        "/register",
        fields={"name": name},
        files={"file": TEST_IMAGE}
    )
    print_result(status, duration, data)
    return status == 200


def test_recognize():
    """测试 5: 人脸识别"""
    print_header("测试 5: 人脸识别 (/recognize)")
    
    if not TEST_IMAGE.exists():
        print(f"[✗] 测试图片不存在: {TEST_IMAGE}")
        return False
    
    status, duration, data = http_post(
        "/recognize",
        fields={"confidence": "0.5", "recognize_threshold": "60", "optimize": "true"},
        files={"file": TEST_IMAGE}
    )
    print_result(status, duration, data)
    
    if status == 200 and data.get("faces"):
        print(f"\n  识别结果:")
        for i, face in enumerate(data["faces"], 1):
            known = "✓ 已知" if face.get("is_known") else "✗ 未知"
            print(f"    Face {i}: {face['name']} (距离:{face['confidence']}, 阈值:{face['threshold']}) [{known}]")
    
    return status == 200


def test_persons():
    """测试 6: 获取已注册人员列表"""
    print_header("测试 6: 已注册人员列表 (/persons)")
    status, duration, data = http_get("/persons")
    print_result(status, duration, data)
    
    if status == 200:
        persons = data.get("persons", [])
        print(f"\n  已注册 {len(persons)} 人: {', '.join(persons) if persons else '无'}")
    
    return status == 200


def test_queue_stats():
    """测试 7: 队列统计"""
    print_header("测试 7: 队列统计 (/queue/stats)")
    status, duration, data = http_get("/queue/stats")
    print_result(status, duration, data)
    return status == 200


def test_concurrent_detect(count=20, workers=10):
    """测试 8: 并发压力测试"""
    print_header(f"测试 8: 并发压力测试 (/detect) | 请求数: {count} | 并发: {workers}")
    
    if not TEST_IMAGE.exists():
        print(f"[✗] 测试图片不存在: {TEST_IMAGE}")
        return False
    
    def single_request():
        return http_post(
            "/detect",
            fields={"confidence": "0.5", "optimize": "true"},
            files={"file": TEST_IMAGE}
        )
    
    start = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(single_request) for _ in range(count)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    total_time = time.time() - start
    
    success = sum(1 for s, _, _ in results if s == 200)
    durations = [d for s, d, _ in results if s == 200]
    face_counts = [r.get("face_count", 0) for s, _, r in results if s == 200 and isinstance(r, dict)]
    
    print(f"  总耗时: {total_time:.3f}s")
    print(f"  成功: {success}/{count}")
    print(f"  失败: {count - success}")
    
    if durations:
        print(f"  平均耗时: {sum(durations)/len(durations):.3f}s")
        print(f"  最快: {min(durations):.3f}s")
        print(f"  最慢: {max(durations):.3f}s")
    
    if face_counts:
        print(f"  平均检测人脸数: {sum(face_counts)/len(face_counts):.1f}")
    
    print(f"  QPS: {count/total_time:.1f}")
    
    return success == count


# ==================== 测试图片准备 ====================

def prepare_test_image():
    """准备测试图片"""
    global TEST_IMAGE
    
    if TEST_IMAGE.exists():
        return True
    
    print(f"[INFO] 测试图片 {TEST_IMAGE} 不存在，尝试创建...")
    
    # 尝试复制 OpenCV 自带图片
    try:
        import cv2
        import shutil
        cv2_data_path = cv2.__file__.replace('__init__.py', 'data/')
        candidates = ['lena.jpg', 'aloeL.jpg', 'butterfly.jpg']
        
        for candidate in candidates:
            src = Path(cv2_data_path) / candidate
            if src.exists():
                shutil.copy(str(src), str(TEST_IMAGE))
                print(f"[✓] 已复制 OpenCV 测试图片: {TEST_IMAGE}")
                return True
    except ImportError:
        pass
    except Exception as e:
        print(f"[WARN] 复制 OpenCV 图片失败: {e}")
    
    # 创建一张 BMP 测试图
    try:
        import numpy as np
        
        width, height = 400, 400
        img = np.ones((height, width, 3), dtype=np.uint8) * 220  # 浅灰背景
        
        # 画椭圆模拟人脸
        center_y, center_x = 200, 200
        radius_y, radius_x = 100, 80
        
        y, x = np.ogrid[:height, :width]
        mask = ((x - center_x)**2 / radius_x**2 + (y - center_y)**2 / radius_y**2) <= 1
        img[mask] = [160, 130, 110]  # BGR 肤色
        
        # 画两个眼睛
        for eye_y, eye_x in [(170, 170), (170, 230)]:
            eye_mask = (y - eye_y)**2 + (x - eye_x)**2 <= 8**2
            img[eye_mask] = [50, 50, 50]  # 黑色眼睛
        
        # 画嘴巴
        mouth_mask = ((x - 200)**2 / 30**2 + (y - 240)**2 / 15**2) <= 1
        mouth_mask &= y > 235
        img[mouth_mask] = [80, 60, 100]  # 暗红色嘴巴
        
        # 保存为 BMP
        bmp_path = TEST_IMAGE.with_suffix('.bmp')
        
        # BMP 文件头 (14 bytes)
        file_size = 14 + 40 + width * height * 3
        row_size = (width * 3 + 3) & ~3  # 4字节对齐
        pixel_data_size = row_size * height
        file_size = 14 + 40 + pixel_data_size
        
        with open(bmp_path, 'wb') as f:
            # BMP 文件头
            f.write(b'BM')                           # 签名
            f.write(file_size.to_bytes(4, 'little'))   # 文件大小
            f.write((0).to_bytes(4, 'little'))         # 保留
            f.write((54).to_bytes(4, 'little'))        # 像素数据偏移
            
            # DIB 头 (BITMAPINFOHEADER, 40 bytes)
            f.write((40).to_bytes(4, 'little'))      # 头大小
            f.write(width.to_bytes(4, 'little'))      # 宽度
            f.write(height.to_bytes(4, 'little'))     # 高度
            f.write((1).to_bytes(2, 'little'))        # 平面数
            f.write((24).to_bytes(2, 'little'))       # 位深度
            f.write((0).to_bytes(4, 'little'))        # 压缩方式
            f.write(pixel_data_size.to_bytes(4, 'little'))
            f.write((2835).to_bytes(4, 'little'))     # X DPI
            f.write((2835).to_bytes(4, 'little'))     # Y DPI
            f.write((0).to_bytes(4, 'little'))        # 调色板颜色数
            f.write((0).to_bytes(4, 'little'))        # 重要颜色数
            
            # 像素数据（BMP 是从下往上存储）
            padding = b'\x00' * (row_size - width * 3)
            for row in range(height - 1, -1, -1):
                for col in range(width):
                    b, g, r = img[row, col]
                    f.write(bytes([int(b), int(g), int(r)]))
                f.write(padding)
        
        # 如果有 OpenCV，转换为 JPEG
        try:
            import cv2
            img_cv = cv2.imread(str(bmp_path))
            if img_cv is not None:
                cv2.imwrite(str(TEST_IMAGE), img_cv, [cv2.IMWRITE_JPEG_QUALITY, 90])
                bmp_path.unlink()
                print(f"[✓] 已创建测试图片: {TEST_IMAGE}")
                return True
        except:
            pass
        
        # 没有 OpenCV，用 BMP 作为测试图
        TEST_IMAGE = bmp_path
        print(f"[✓] 已创建测试图片: {TEST_IMAGE}")
        return True
            
    except Exception as e:
        print(f"[✗] 无法创建测试图片: {e}")
        return False


# ==================== 主程序 ====================

def run_all_tests():
    """运行全部测试"""
    print(f"\n{'#'*60}")
    print(f"#{'':^58}#")
    print(f"#{'AI Faces 人脸识别服务 - 完整测试':^58}#")
    print(f"#{'':^58}#")
    print(f"{'#'*60}")
    
    # 准备测试图片
    if not prepare_test_image():
        print("\n[✗] 无法准备测试图片，测试终止")
        return
    
    results = []
    
    # 基础测试
    results.append(("健康检查", test_health()))
    results.append(("人脸检测", test_detect_single()))
    results.append(("图片上传", test_upload()))
    
    # 注册和识别测试
    results.append(("人脸注册", test_register("张三")))
    results.append(("人脸识别", test_recognize()))
    results.append(("人员列表", test_persons()))
    
    # 队列和性能测试
    results.append(("队列统计", test_queue_stats()))
    results.append(("并发压力", test_concurrent_detect(count=20, workers=10)))
    
    # 再次识别（测试缓存效果）
    print_header("测试 9: 再次识别（验证缓存）")
    status, duration, data = http_post(
        "/recognize",
        fields={"confidence": "0.5", "recognize_threshold": "60", "optimize": "true"},
        files={"file": TEST_IMAGE}
    )
    print_result(status, duration, data)
    results.append(("缓存识别", status == 200))
    
    # 最终统计
    print(f"\n{'#'*60}")
    print(f"#{'':^58}#")
    print(f"#{'测试完成':^58}#")
    print(f"{'#'*60}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\n  总计: {total} 项 | 通过: {passed} | 失败: {total - passed}")
    print(f"\n  详细结果:")
    for name, result in results:
        icon = "✓" if result else "✗"
        color = "\033[92m" if result else "\033[91m"
        reset = "\033[0m"
        print(f"    {color}[{icon}]{reset} {name}")
    
    if passed == total:
        print(f"\n  \033[92m🎉 所有测试全部通过！\033[0m")
    else:
        print(f"\n  \033[91m⚠️ 有 {total - passed} 项测试未通过，请检查日志\033[0m")


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n[!] 测试已中断")
    except Exception as e:
        print(f"\n\n[✗] 测试异常: {e}")
        import traceback
        traceback.print_exc()