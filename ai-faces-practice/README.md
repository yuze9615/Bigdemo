# AI Faces - 工业互联网边缘计算之人脸识别系统

> 基于 FastAPI + OpenCV DNN/LBPH 的高性能人脸识别服务，专为工业互联网边缘计算场景设计。

------

## 📋 项目简介

本项目是一个基于**工业互联网边缘计算架构**的人脸识别原型系统，使用 **FastAPI** 框架构建 RESTful API，结合 **OpenCV DNN (SSD)** 深度学习模型进行高精度人脸检测，并基于 **LBPH (局部二值模式直方图)** 算法实现实时人脸比对与识别。

系统针对边缘计算场景进行了深度性能优化，包括模型缓存、图片压缩、异步任务队列等，确保在低功耗设备上也能流畅运行。

## ✨ 功能特性

| 功能模块              | 说明                                                         |
| :-------------------- | :----------------------------------------------------------- |
| 🎯 **高精度人脸检测**  | 基于 OpenCV DNN (ResNet-SSD) 深度学习模型，支持 Haar 级联降级 |
| 🔍 **实时人脸比对**    | 基于 LBPH 算法，支持多尺度预测和数据增强训练                 |
| 📝 **人脸注册管理**    | 支持手动注册 + 摄像头自动检测注册，中文路径兼容              |
| 📷 **摄像头实时检测**  | Web 端直接调用摄像头，实时画框标注人脸位置与姓名             |
| 🖼️ **图片上传检测**    | 支持 jpg/jpeg/png 格式，自动压缩优化                         |
| ⚡ **性能优化**        | 模型缓存预热、图片压缩、异步任务队列、并发控制               |
| 🌐 **响应式 Web 界面** | 单页应用，支持图片检测、摄像头检测、人脸比对三大模式         |
| 📊 **健康监控**        | `/health` 和 `/queue/stats` 接口实时查看服务状态             |

------

## 🏗️ 项目结构

```plain
face_detection_app/
├── app.py                  # FastAPI 主应用入口
├── face_detector.py        # 人脸检测核心模块 (DNN/Haar)
├── face_recognizer.py      # 人脸比对识别模块 (LBPH)
├── image_optimizer.py      # 图片压缩优化模块 (纯 OpenCV)
├── async_queue.py          # 异步任务队列模块
├── requirements.txt        # Python 依赖
│
├── static/                 # 静态文件目录
│   └── images/             # 上传/结果图片存储
├── known_faces/            # 已注册人脸数据库存储
├── models/                 # DNN 模型文件下载目录
│
├── templates/
│   └── index.html          # Web 前端界面
│
├── test_detector.py        # 检测器单元测试
├── test_full.py            # 完整功能测试脚本
└── test_performance.py     # 性能/并发压力测试
```

------

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Windows / Linux / macOS
- 摄像头（可选，用于实时检测功能）

### 安装部署

```bash
# 1. 克隆项目
git clone https://github.com/yuze9615/Bigdemo.git
# 进入项目代码目录
cd Bigdemo/ai-faces-practice

# 2. 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动服务
python app.py
```

服务启动后访问：

- 🌐 **Web 界面**: [http://127.0.0.1:8001](http://127.0.0.1:8001/)
- 📚 **API 文档**: http://127.0.0.1:8001/docs (Swagger UI)
- 📖 **ReDoc 文档**: http://127.0.0.1:8001/redoc

> **首次启动**会自动下载 DNN 模型文件（约 5MB），请确保网络畅通。

------

## 📡 API 接口

### 基础接口

| 方法   | 路径                 | 说明                             |
| :----- | :------------------- | :------------------------------- |
| GET    | `/`                  | 返回 Web 前端页面                |
| GET    | `/health`            | 健康检查，返回模型状态与队列统计 |
| GET    | `/queue/stats`       | 获取异步任务队列统计             |
| DELETE | `/queue/cache`       | 清理队列结果缓存                 |
| GET    | `/result/{filename}` | 获取处理结果图片                 |

### 人脸检测

| 方法 | 路径      | 说明                               |
| :--- | :-------- | :--------------------------------- |
| POST | `/detect` | 仅检测人脸坐标（内存处理，不落盘） |
| POST | `/upload` | 上传图片并检测，返回标记结果图     |

**参数说明：**

- `file`: 图片文件 (jpg/jpeg/png)
- `confidence`: 置信度阈值，范围 0.1~0.9，默认 0.5
- `optimize`: 是否启用图片压缩优化，默认 `true`
- `compress`: `/upload` 专用，是否压缩输出图片，默认 `true`

### 人脸识别

| 方法   | 路径              | 说明                       |
| :----- | :---------------- | :------------------------- |
| POST   | `/recognize`      | 检测人脸并与数据库比对身份 |
| POST   | `/register`       | 注册新的人脸到数据库       |
| GET    | `/persons`        | 获取已注册人员列表         |
| DELETE | `/persons/{name}` | 删除指定人员               |

**识别参数：**

- `recognize_threshold`: LBPH 距离阈值，默认 60，越小越严格

------

## 🧪 测试

### 1. 单元测试（检测器）

```bash
python test_detector.py
```

### 2. 完整功能测试

```bash
# 确保服务已启动: python app.py
# 另开终端执行:
python test_full.py
```

测试覆盖：

- ✅ 健康检查
- ✅ 单张人脸检测
- ✅ 图片上传检测
- ✅ 人脸注册
- ✅ 人脸识别
- ✅ 人员列表查询
- ✅ 队列统计
- ✅ 并发压力测试（20 请求 / 10 并发）
- ✅ 缓存验证

### 3. 性能压力测试

```bash
python test_performance.py
```

输出 QPS、平均耗时、最快/最慢响应等指标。

------

## 🎨 Web 界面使用指南

### 图片检测模式

1. 选择本地图片文件
2. 调节置信度阈值（越低越宽松）
3. 点击「检测人脸」
4. 查看检测结果与人脸标记图

### 摄像头检测模式

1. 点击「开启摄像头」授权
2. 系统自动开始实时检测
3. 视频画面上实时绘制人脸边界框与置信度
4. 可调节检测间隔控制 CPU 占用

### 人脸比对模式

1. **手动注册**：选择正脸照片 + 输入姓名 → 注册
2. **实时识别**：开启摄像头自动识别已注册人员
3. **自动注册**：未知人脸持续出现 2 秒，自动弹窗提示注册

------

## ⚙️ 核心配置

### 检测器配置 (`face_detector.py`)

```python
detector = FaceDetector(
    use_dnn=True,      # True: DNN (SSD), False: Haar 级联
    cache_model=True   # 启用模型缓存，多实例共享
)
```

### 识别器配置 (`face_recognizer.py`)

```python
recognizer = FaceRecognizer(
    known_faces_dir="known_faces",  # 人脸数据库目录
    face_size=(200, 200),           # 统一输入尺寸
    threshold=60                    # LBPH 距离阈值
)
```

### 图片优化配置 (`image_optimizer.py`)

```python
optimizer = ImageOptimizer(
    max_size=(1920, 1080),          # 最大尺寸限制
    max_file_size=2*1024*1024,      # 最大文件大小 2MB
    quality=85                      # JPEG 质量
)
```

### 异步队列配置 (`async_queue.py`)

```python
task_queue = get_task_queue(
    max_workers=4,      # 最大并发工作数
    default_timeout=30  # 默认超时时间（秒）
)
```

------

## 🔧 技术栈

| 技术             | 版本   | 用途                  |
| :--------------- | :----- | :-------------------- |
| FastAPI          | ≥0.104 | Web 框架              |
| Uvicorn          | ≥0.24  | ASGI 服务器           |
| OpenCV (contrib) | ≥4.8   | 图像处理 + DNN + LBPH |
| NumPy            | ≥1.24  | 数值计算              |
| python-multipart | ≥0.0.6 | 文件上传解析          |

------

## 📈 性能指标

在典型边缘设备（4 核 ARM）上的参考表现：

| 场景           | 平均耗时 | 说明                  |
| :------------- | :------- | :-------------------- |
| 单张人脸检测   | ~200ms   | DNN 模型，图片 1280px |
| 并发检测 (QPS) | ~5-8     | 10 并发，模型缓存命中 |
| 人脸识别       | ~300ms   | 含检测 + LBPH 比对    |
| 图片压缩       | ~50ms    | 1920×1080 → 压缩      |

------

## 📝 注意事项

1. **首次启动**会自动从 GitHub 下载 DNN 模型文件，请确保网络畅通或提前手动放置到 `models/` 目录
2. **LBPH 识别精度**与注册照片数量/质量正相关，建议每人注册 **3-5 张**不同角度/光线的照片
3. **中文路径**已做兼容处理，Windows 用户无需担心编码问题
4. **生产部署**建议使用 `gunicorn` + `uvicorn.workers.UvicornWorker` 多进程部署

------

## 📄 许可证

MIT License