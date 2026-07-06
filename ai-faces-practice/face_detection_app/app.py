"""
工业互联网边缘计算 - 人脸识别Web服务
使用FastAPI框架实现图片上传和人脸检测API
性能优化版：模型缓存 + 图片压缩 + 异步队列
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid
import cv2
import numpy as np
import asyncio
import time

from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from image_optimizer import ImageOptimizer, optimize_for_api
from async_queue import get_task_queue, AsyncTaskQueue

# 创建FastAPI应用
app = FastAPI(
    title="AI Faces - 人脸识别服务",
    description="基于工业互联网边缘计算的人脸识别系统（性能优化版）",
    version="2.0.0"
)

# 配置静态文件目录
STATIC_DIR = Path("static/images")
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 配置
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# 初始化组件
detector = FaceDetector(cache_model=True)
recognizer = FaceRecognizer(known_faces_dir="known_faces", threshold=60)
image_optimizer = ImageOptimizer(max_size=(1920, 1080), quality=85)
task_queue = get_task_queue(max_workers=4, default_timeout=30)

# 启动时预热模型
@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    print("[INFO] 应用启动中...")
    # 预热检测模型
    FaceDetector.warm_up(use_dnn=True)
    print("[INFO] 模型预热完成，服务就绪")


@app.get("/", response_class=HTMLResponse)
async def home():
    """返回前端页面"""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/upload")
async def upload_image(
        file: UploadFile = File(...),
        confidence: float = Form(0.5),
        compress: bool = Form(True)
):
    """
    上传图片并检测人脸（优化版）
    Args:
        file: 上传的图片文件
        confidence: 人脸检测置信度阈值（0.1-0.9）
        compress: 是否启用图片压缩
    Returns:
        dict: 包含检测结果的JSON
    """
    start_time = time.time()

    # 验证文件名
    if not file.filename:
        raise HTTPException(status_code=400, detail="请选择图片文件")

    # 验证文件扩展名
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="只支持 jpg, jpeg, png 格式"
        )

    # 读取文件内容
    content = await file.read()

    # 验证文件大小
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="文件大小不能超过10MB"
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="文件不能为空"
        )

    # 图片压缩优化
    if compress and len(content) > 500 * 1024:  # 大于500KB才压缩
        try:
            optimized = image_optimizer.optimize(content, target_format='jpeg')
            if optimized and len(optimized) < len(content):
                content = optimized
                print(f"[INFO] 图片压缩: {len(content)/1024:.1f}KB -> {len(optimized)/1024:.1f}KB")
        except Exception as e:
            print(f"[WARN] 图片压缩失败: {e}")

    # 保存上传的文件
    unique_id = str(uuid.uuid4())
    input_filename = f"input_{unique_id}{file_ext}"
    input_path = STATIC_DIR / input_filename
    with open(input_path, "wb") as buffer:
        buffer.write(content)

    try:
        # 使用异步队列处理图片
        task_id = f"upload_{unique_id}"
        
        def process_task():
            result_filename = f"result_{unique_id}{file_ext}"
            result_path = STATIC_DIR / result_filename
            face_count, _ = detector.process_image(
                input_path,
                result_path,
                confidence_threshold=confidence
            )
            return {
                "face_count": face_count,
                "result_filename": result_filename
            }

        # 提交到队列执行
        task_result = await task_queue.submit(task_id, process_task, timeout=15)

        if task_result.status.value == "completed":
            result_data = task_result.result
            process_time = round(time.time() - start_time, 3)
            
            return {
                "success": True,
                "face_count": result_data["face_count"],
                "original_filename": file.filename,
                "result_filename": result_data["result_filename"],
                "message": f"检测到 {result_data['face_count']} 张人脸" if result_data["face_count"] > 0 else "未检测到人脸",
                "model": "DNN (SSD)" if detector.use_dnn else "Haar Cascade",
                "process_time": process_time,
                "compressed": compress,
                "task_id": task_id
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"处理失败: {task_result.error}"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"处理图片时出错: {str(e)}"
        )
    finally:
        # 清理临时文件
        if input_path.exists():
            input_path.unlink()


@app.post("/detect")
async def detect_faces(
        file: UploadFile = File(...),
        confidence: float = Form(0.5),
        optimize: bool = Form(True)
):
    """
    仅检测人脸坐标（优化版）
    - 支持图片压缩优化
    - 内存处理，不落盘
    """
    start_time = time.time()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件不能为空")

    # 图片优化（为人脸检测调整尺寸）
    if optimize:
        try:
            image = optimize_for_api(content, max_dimension=1280)
        except Exception:
            # 优化失败则直接解码
            arr = np.frombuffer(content, np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        arr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="无法解码图片")

    try:
        # 直接检测（轻量操作，不需要队列）
        if detector.use_dnn:
            face_count, faces, _, confidences = detector._detect_faces_dnn(image, confidence_threshold=confidence)
        else:
            face_count, faces, _, confidences = detector._detect_faces_haar(image)

        h, w = image.shape[:2]
        result = []
        for i, face in enumerate(faces):
            x, y, fw, fh = face

            # 扩展 20% 边界
            pad_x = int(fw * 0.20)
            pad_y = int(fh * 0.20)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + fw + pad_x)
            y2 = min(h, y + fh + pad_y)

            result.append({
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "confidence": float(confidences[i]) if i < len(confidences) else 1.0
            })

        process_time = round(time.time() - start_time, 3)

        return {
            "success": True,
            "face_count": face_count,
            "image_width": int(w),
            "image_height": int(h),
            "faces": result,
            "model": "DNN (SSD)" if detector.use_dnn else "Haar Cascade",
            "process_time": process_time,
            "optimized": optimize
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@app.post("/recognize")
async def recognize_faces_api(
        file: UploadFile = File(...),
        confidence: float = Form(0.5),
        recognize_threshold: float = Form(60.0),
        optimize: bool = Form(True)
):
    """
    检测人脸并与数据库比对（优化版）
    """
    start_time = time.time()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件不能为空")

    # 图片优化
    if optimize:
        try:
            image = optimize_for_api(content, max_dimension=1280)
        except Exception:
            arr = np.frombuffer(content, np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        arr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="无法解码图片")

    # 使用队列处理识别任务
    task_id = f"recognize_{uuid.uuid4().hex[:8]}"
    
    def recognize_task():
        if detector.use_dnn:
            face_count, faces, _, confidences = detector._detect_faces_dnn(image, confidence_threshold=confidence)
        else:
            face_count, faces, _, confidences = detector._detect_faces_haar(image)

        recognition_results = recognizer.recognize_faces(image, faces)
        
        # 阈值过滤
        filtered_results = []
        known_count = 0
        unknown_count = 0

        for r in recognition_results:
            if recognize_threshold > 0:
                is_known = r.get("confidence", 999) <= recognize_threshold
                if not is_known:
                    r["name"] = "未知"
                    r["is_known"] = False
                    unknown_count += 1
                else:
                    r["is_known"] = True
                    known_count += 1
            else:
                if r.get("is_known", False):
                    known_count += 1
                else:
                    unknown_count += 1
            filtered_results.append(r)

        return {
            "face_count": face_count,
            "known_count": known_count,
            "unknown_count": unknown_count,
            "faces": filtered_results
        }

    try:
        task_result = await task_queue.submit(task_id, recognize_task, timeout=20)
        
        if task_result.status.value == "completed":
            result_data = task_result.result
            h, w = image.shape[:2]
            process_time = round(time.time() - start_time, 3)

            return {
                "success": True,
                "face_count": result_data["face_count"],
                "known_count": result_data["known_count"],
                "unknown_count": result_data["unknown_count"],
                "image_width": int(w),
                "image_height": int(h),
                "trained": recognizer.trained,
                "recognize_threshold": recognize_threshold,
                "faces": result_data["faces"],
                "model": "DNN (SSD) + LBPH" if detector.use_dnn else "Haar Cascade + LBPH",
                "process_time": process_time,
                "task_id": task_id
            }
        else:
            raise HTTPException(status_code=500, detail=f"识别失败: {task_result.error}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")


@app.post("/register")
async def register_face(
        file: UploadFile = File(...),
        name: str = Form(...)
):
    """
    注册一张人脸到已知人脸数据库
    """
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="请输入姓名")
    name = name.strip()
    import re
    if not re.match(r'^[\u4e00-\u9fa5\w\- ]+$', name):
        raise HTTPException(status_code=400, detail="姓名只能包含中英文、数字、下划线和空格")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件不能为空")

    # 优化图片
    try:
        image = optimize_for_api(content, max_dimension=1280)
    except Exception:
        arr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="无法解码图片")

    try:
        # 检测人脸位置
        if detector.use_dnn:
            face_count, faces, _, confidences = detector._detect_faces_dnn(image, confidence_threshold=0.3)
        else:
            face_count, faces, _, confidences = detector._detect_faces_haar(image)

        faces_list = []
        if face_count > 0 and len(faces) > 0:
            for f in faces:
                if hasattr(f, '__len__') and len(f) == 4:
                    faces_list.append([int(v) for v in f])

        # 找最大的人脸注册
        best_idx = 0
        if len(faces_list) > 1:
            best_area = 0
            for i, (x, y, fw, fh) in enumerate(faces_list):
                area = fw * fh
                if area > best_area:
                    best_area = area
                    best_idx = i

        ok = recognizer.register_from_image(
            name, image, 
            faces=faces_list if faces_list else None, 
            face_index=best_idx
        )
        if not ok:
            raise HTTPException(status_code=400, detail="无法处理人脸，请上传清晰的正脸照片")
        
        names = recognizer.get_registered_names()
        person_dir = Path("known_faces") / name
        photo_count = len(
            list(person_dir.glob("*.jpg")) + 
            list(person_dir.glob("*.jpeg")) + 
            list(person_dir.glob("*.png"))
        )

        return {
            "success": True,
            "message": f"已注册 {name}（当前共 {photo_count} 张照片）",
            "registered_persons": names,
            "tip": "建议同一个人注册 3-5 张不同角度/光线的照片，识别效果更好"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"注册失败: {str(e)}")


@app.get("/persons")
async def get_persons():
    """获取已注册的人员列表"""
    return {
        "success": True,
        "trained": recognizer.trained,
        "persons": recognizer.get_registered_names()
    }


@app.delete("/persons/{name}")
async def delete_person(name: str):
    """删除指定人员"""
    ok = recognizer.delete_person(name)
    if not ok:
        raise HTTPException(status_code=404, detail="人员不存在")
    return {"success": True, "message": f"已删除 {name}"}


@app.get("/result/{filename}")
async def get_result(filename: str):
    """
    获取处理结果图片
    """
    file_path = STATIC_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(file_path)


@app.get("/health")
async def health_check():
    """健康检查接口（优化版）"""
    return {
        "status": "healthy",
        "model": "DNN (SSD)" if detector.use_dnn else "Haar Cascade",
        "recognizer_trained": recognizer.trained,
        "registered_persons": recognizer.get_registered_names(),
        "queue_stats": task_queue.stats,
        "features": {
            "model_cache": True,
            "image_compression": True,
            "async_queue": True
        }
    }


@app.get("/queue/stats")
async def queue_stats():
    """获取队列统计信息"""
    return {
        "success": True,
        "stats": task_queue.stats
    }


@app.delete("/queue/cache")
async def clear_queue_cache():
    """清理队列缓存"""
    task_queue.clear_cache()
    return {"success": True, "message": "队列缓存已清理"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)