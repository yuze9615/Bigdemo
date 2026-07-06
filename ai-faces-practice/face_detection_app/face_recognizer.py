"""
人脸比对模块 - 基于 OpenCV LBPH 人脸识别器
使用已知人脸数据库进行训练，实时返回匹配姓名
兼容 Windows 中文路径
"""
import cv2
import numpy as np
from pathlib import Path
import json
import shutil


class FaceRecognizer:
    def __init__(self, known_faces_dir="known_faces", face_size=(200, 200), threshold=60):
        """
        参数说明：
        - face_size=(200,200): 统一输入尺寸
        - threshold=60: LBPH 距离阈值，越小越严格
                        距离 < threshold 才认为是已知人员，否则为"未知"
                        建议值：40-70，根据实际测试调整
        """
        self.known_faces_dir = Path(known_faces_dir)
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        self.face_size = face_size
        self.threshold = threshold
        self.model_path = self.known_faces_dir / "model.yml"
        self.labels_path = self.known_faces_dir / "labels.json"

        self.recognizer = None
        self.label_map = {}
        self.trained = False

        self._init_recognizer()
        self._load_or_train()

    def _set_big_threshold(self):
        """强制设置极大阈值，让 predict 始终返回真实距离而不是 DBL_MAX"""
        import sys as _sys
        big = _sys.float_info.max
        try:
            self.recognizer.setThreshold(big)
        except Exception:
            pass

    def _init_recognizer(self):
        """
        LBPH 参数：
        - radius=1, neighbors=8：标准参数，平衡速度与精度
        - grid_x=8, grid_y=8：标准空间分块
        - threshold=DBL_MAX：关闭内置截断，始终返回真实距离
        """
        import sys as _sys
        big_threshold = _sys.float_info.max
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=big_threshold
            )
        except AttributeError:
            try:
                self.recognizer = cv2.face_LBPHFaceRecognizer.create(
                    radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=big_threshold
                )
            except AttributeError:
                self.recognizer = cv2.createLBPHFaceRecognizer()
        self._set_big_threshold()

    @staticmethod
    def _imread(path):
        """支持中文路径读取图片"""
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return None
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None

    @staticmethod
    def _imwrite(path, image):
        """支持中文路径保存图片"""
        try:
            path = Path(path)
            ext = path.suffix.lower() if path.suffix else ".jpg"
            ok, buf = cv2.imencode(ext, image)
            if ok:
                buf.tofile(str(path))
                return True
            return False
        except Exception:
            return False

    def _preprocess_face(self, face_img):
        if face_img is None or face_img.size == 0:
            return None
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        gray = cv2.equalizeHist(gray)
        resized = cv2.resize(gray, self.face_size, interpolation=cv2.INTER_LINEAR)
        return resized

    def _collect_training_data(self):
        images = []
        labels = []
        self.label_map = {}
        label_id = 0

        for person_dir in sorted(self.known_faces_dir.iterdir()):
            if not person_dir.is_dir() or person_dir.name.startswith("__") or person_dir.name.startswith("."):
                continue
            person_name = person_dir.name
            self.label_map[label_id] = person_name
            added = 0

            for img_path in sorted(list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.jpeg")) + list(person_dir.glob("*.png"))):
                img = self._imread(img_path)
                if img is None:
                    continue
                processed = self._preprocess_face(img)
                if processed is not None:
                    images.append(processed)
                    labels.append(label_id)
                    added += 1

                    # 数据增强：轻微旋转、亮度变化、尺度变化
                    # 1. 水平翻转
                    flipped = cv2.flip(processed, 1)
                    images.append(flipped)
                    labels.append(label_id)
                    added += 1

                    # 2. 轻微旋转（±5度）
                    for angle in [-5, 5]:
                        h, w = processed.shape
                        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                        rotated = cv2.warpAffine(processed, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                        images.append(rotated)
                        labels.append(label_id)
                        added += 1

                    # 3. 亮度变化
                    for alpha in [0.9, 1.1]:
                        bright = cv2.convertScaleAbs(processed, alpha=alpha, beta=0)
                        images.append(bright)
                        labels.append(label_id)
                        added += 1

            if added == 0:
                if label_id in self.label_map:
                    del self.label_map[label_id]
            else:
                label_id += 1

        return images, labels

    def train(self):
        images, labels = self._collect_training_data()
        if len(images) == 0:
            self.trained = False
            return False
        labels_arr = np.array(labels, dtype=np.int32)
        self._set_big_threshold()
        self.recognizer.train(images, labels_arr)
        self._save_model()
        self.trained = True
        return True

    def _save_model(self):
        """兼容中文路径保存模型：先保存到临时英文路径再复制"""
        temp_model = self.known_faces_dir / "_temp_model.yml"
        self.recognizer.save(str(temp_model))
        if temp_model.exists():
            shutil.copy2(str(temp_model), str(self.model_path))
            temp_model.unlink()
        with open(self.labels_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.label_map.items()}, f, ensure_ascii=False)

    def _load_model(self):
        """兼容中文路径加载模型"""
        if not self.model_path.exists() or not self.labels_path.exists():
            return False
        try:
            temp_model = self.known_faces_dir / "_temp_model.yml"
            shutil.copy2(str(self.model_path), str(temp_model))
            self.recognizer.read(str(temp_model))
            if temp_model.exists():
                temp_model.unlink()
            self._set_big_threshold()
            with open(self.labels_path, "r", encoding="utf-8") as f:
                self.label_map = {int(k): v for k, v in json.load(f).items()}
            self.trained = len(self.label_map) > 0
            return self.trained
        except Exception:
            return False

    def _load_or_train(self):
        if not self._load_model():
            self.train()

    def register_face(self, name, face_image):
        processed = self._preprocess_face(face_image)
        if processed is None:
            return False

        person_dir = self.known_faces_dir / name
        person_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.jpeg")) + list(person_dir.glob("*.png")))
        save_path = person_dir / f"{existing + 1:04d}.jpg"
        save_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        if not self._imwrite(save_path, save_img):
            return False

        return self.train()

    def register_from_image(self, name, image, faces=None, face_index=0):
        h_img, w_img = image.shape[:2]
        face_crop = None

        if faces is not None and len(faces) > face_index:
            face = faces[face_index]
            if hasattr(face, '__len__') and len(face) == 4:
                x, y, fw, fh = [int(v) for v in face]
                # 扩展 35% 边界，确保框住完整脸部（额头、下巴、耳朵）
                pad_x = int(fw * 0.05)
                pad_y = int(fh * 0.05)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w_img, x + fw + pad_x)
                y2 = min(h_img, y + fh + pad_y)
                face_crop = image[y1:y2, x1:x2]

        if face_crop is None or face_crop.size == 0:
            margin_x = int(w_img * 0.15)
            margin_y = int(h_img * 0.1)
            face_crop = image[margin_y:h_img - margin_y, margin_x:w_img - margin_x]

        if face_crop is None or face_crop.size == 0:
            return False

        return self.register_face(name, face_crop)

    def recognize_faces(self, image, faces):
        if not self.trained or len(faces) == 0:
            results = []
            for face in faces:
                x, y, w, h = [int(v) for v in face]
                results.append({
                    "name": "未知" if self.trained else "未训练",
                    "confidence": -1,
                    "box": [x, y, w, h]
                })
            return results

        h_img, w_img = image.shape[:2]
        results = []
        for face in faces:
            x, y, w, h = [int(v) for v in face]

            # === 扩展 35% 边界，确保框住完整脸部 ===
            # 包含额头、下巴、耳朵，不裁掉任何面部特征
            pad_x = int(w * 0.05)
            pad_y = int(h * 0.05)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)
            face_crop = image[y1:y2, x1:x2]

            # 用于画框的完整边界（扩展后的）
            box_x = x1
            box_y = y1
            box_w = x2 - x1
            box_h = y2 - y1

            base_processed = self._preprocess_face(face_crop)
            if base_processed is None:
                results.append({
                    "name": "未知",
                    "confidence": -1,
                    "box": [box_x, box_y, box_w, box_h]
                })
                continue

            # 多尺度预测，取最优结果
            best_confidence = float('inf')
            best_label = -1

            scales = [1.0, 0.9, 1.1]
            for scale in scales:
                if scale == 1.0:
                    proc = base_processed
                else:
                    h, w = base_processed.shape
                    new_h, new_w = int(h * scale), int(w * scale)
                    scaled = cv2.resize(base_processed, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    proc = cv2.resize(scaled, self.face_size, interpolation=cv2.INTER_LINEAR)

                try:
                    label_id, confidence = self.recognizer.predict(proc)
                    if confidence < best_confidence:
                        best_confidence = confidence
                        best_label = label_id
                except Exception:
                    continue

            # 关键：根据阈值判断是否为陌生人
            name = self.label_map.get(best_label, "未知") if best_label >= 0 else "未知"
            is_known = best_confidence <= self.threshold

            if not is_known:
                name = "未知"

            results.append({
                "name": name,
                "confidence": round(float(best_confidence), 1),
                "threshold": self.threshold,
                "is_known": is_known,
                # 返回扩展后的完整框，确保前端画框覆盖整张脸
                "box": [box_x, box_y, box_w, box_h]
            })

        return results

    def get_registered_names(self):
        return sorted(self.label_map.values())

    def delete_person(self, name):
        person_dir = self.known_faces_dir / name
        if person_dir.exists():
            shutil.rmtree(person_dir)
            self.train()
            return True
        return False