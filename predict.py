# import os
# import json
# import csv
# import glob
# import time
# import numpy as np
# from typing import List, Dict, Optional, Tuple
# from PIL import Image
# import tensorflow as tf
# from flask import Flask, request, jsonify, send_file
# from werkzeug.utils import secure_filename
# import tempfile
# import shutil
#
#
# # 全局TensorFlow优化配置
# def setup_tf_optimization():
#     try:
#         gpus = tf.config.experimental.list_physical_devices('GPU')
#         if gpus:
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#             print(f"GPU加速启用: {len(gpus)} 张GPU")
#         else:
#             print("未检测到GPU，使用CPU推理")
#     except RuntimeError as e:
#         print(f"GPU配置失败，自动切换CPU: {e}")
#
#     tf.config.optimizer.set_jit(True)
#     tf.get_logger().setLevel('ERROR')
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
#     cpu_count = os.cpu_count() or 4
#     tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
#     tf.config.threading.set_inter_op_parallelism_threads(cpu_count)
#
#
# setup_tf_optimization()
#
# # Flask 应用初始化
# app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 限制上传文件大小100MB
# app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()  # 临时上传目录
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#
# # 允许的图片扩展名
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
#
# # 模型配置（可根据实际情况修改）
# MODEL_CONFIG = {
#     "num_classes": 100,
#     "model_arch": "s",
#     "class_indices_path": "./class_indices.json",
#     "class_mapping_path": "./class.json",
#     "weights_path": '../model/efficientnetv2_gai1_cbam.ckpt',
#     "use_cbam": True
# }
#
#
# def allowed_file(filename):
#     """检查文件是否为允许的图片格式"""
#     return '.' in filename and \
#         filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
#
# class ImagePredictor:
#     def __init__(self,
#                  num_classes: int,
#                  model_arch: str = "s",
#                  class_indices_path: str = "./class_indices.json",
#                  class_mapping_path: str = "./class.json",
#                  weights_path: str = '../model/efficientnetv2_gai1_cbam.ckpt',
#                  use_cbam: bool = True):
#         self.num_classes = num_classes
#         self.model_arch = model_arch
#         self.use_cbam = use_cbam
#         self.class_indict = self._load_class_indices(class_indices_path)
#         self.name_to_category_id = self._load_class_mapping(class_mapping_path)
#         self.im_width, self.im_height = self._get_image_size()
#         self.model = self._load_model(weights_path)
#         self._optimize_model_inference()
#         self._warmup_model()
#
#     def _get_image_size(self) -> Tuple[int, int]:
#         img_sizes = {"s": 384, "m": 480, "l": 480}
#         size = img_sizes.get(self.model_arch, 384)
#         return size, size
#
#     def _load_class_indices(self, path: str) -> Dict[str, str]:
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"类别文件不存在: {os.path.abspath(path)}")
#         try:
#             with open(path, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#                 # 校验格式
#                 if not all(k.isdigit() for k in data.keys()):
#                     raise ValueError("class_indices.json 键必须是数字字符串")
#                 return data
#         except json.JSONDecodeError as e:
#             raise ValueError(f"类别文件JSON解析失败: {str(e)}")
#         except Exception as e:
#             raise RuntimeError(f"加载类别文件失败: {str(e)}")
#
#     def _load_class_mapping(self, path: str) -> Dict[str, str]:
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"class文件不存在: {os.path.abspath(path)}")
#         try:
#             with open(path, "r", encoding="utf-8") as f:
#                 class_data = json.load(f)
#             if not isinstance(class_data, dict):
#                 raise ValueError("class.json 必须是字典格式")
#             return {v: k for k, v in class_data.items()}
#         except json.JSONDecodeError as e:
#             raise ValueError(f"class文件JSON解析失败: {str(e)}")
#         except Exception as e:
#             raise RuntimeError(f"加载class文件失败: {str(e)}")
#
#     def _load_model(self, weights_path: str) -> tf.keras.Model:
#         try:
#             from model import efficientnetv2_s as create_model
#         except ImportError:
#             raise ImportError("未找到model模块，请确保efficientnetv2_s.py在model目录下")
#
#         weight_files = glob.glob(f"{weights_path}*")
#         if not weight_files:
#             raise FileNotFoundError(f"未找到权重文件: {os.path.abspath(weights_path)}")
#
#         try:
#             model = create_model(num_classes=self.num_classes, use_cbam=self.use_cbam)
#             model.load_weights(weights_path)
#             print("模型权重加载成功")
#             return model
#         except Exception as e:
#             raise RuntimeError(f"加载模型失败: {str(e)}")
#
#     def _optimize_model_inference(self):
#         @tf.function(jit_compile=True)
#         def predict_fn(inputs):
#             return self.model(inputs, training=False)
#
#         self.predict_fn = predict_fn
#         self.softmax_layer = tf.keras.layers.Softmax()
#
#     def _warmup_model(self):
#         print("模型预热中...")
#         try:
#             dummy_input = np.random.randn(1, self.im_height, self.im_width, 3).astype(np.float32)
#             dummy_input = (dummy_input / 255.0 - 0.5) / 0.5
#             for _ in range(3):  # 减少预热次数提升启动速度
#                 self.predict_fn(dummy_input)
#             print("模型预热完成")
#         except Exception as e:
#             raise RuntimeError(f"模型预热失败: {str(e)}")
#
#     def preprocess_image(self, image_path: str) -> np.ndarray:
#         try:
#             if not os.path.exists(image_path):
#                 raise FileNotFoundError(f"图片不存在: {os.path.abspath(image_path)}")
#             with Image.open(image_path).convert('RGB') as img:
#                 img = img.resize((self.im_width, self.im_height), Image.Resampling.LANCZOS)
#                 img_array = np.array(img, dtype=np.float32)
#                 img_array = (img_array / 255.0 - 0.5) / 0.5
#                 return img_array[np.newaxis, ...]
#         except Exception as e:
#             raise RuntimeError(f"图片预处理失败: {str(e)}")
#
#     def predict_single_image(self, image_path: str) -> Tuple[Dict[str, any], float]:
#         try:
#             img_array = self.preprocess_image(image_path)
#             start_time = time.perf_counter()
#             predictions = self.predict_fn(img_array)
#             inference_time = (time.perf_counter() - start_time) * 1000
#
#             probabilities = self.softmax_layer(predictions).numpy()[0]
#             predict_idx = np.argmax(probabilities)
#             confidence = float(probabilities[predict_idx])
#             class_name = self.class_indict.get(str(predict_idx), "未知类别")
#             category_id = self.name_to_category_id.get(class_name, "未知ID")
#
#             return {
#                 'filename': os.path.basename(image_path),
#                 'category_id': category_id,
#                 'confidence': round(confidence, 6),
#                 'class_name': class_name,
#                 'inference_time_ms': round(inference_time, 2)
#             }, inference_time
#         except Exception as e:
#             raise RuntimeError(f"单图预测失败: {str(e)}")
#
#     def predict_multiple_images(self, image_dir: str) -> Tuple[List[Dict[str, any]], float, float]:
#         if not os.path.isdir(image_dir):
#             raise NotADirectoryError(f"目录不存在: {os.path.abspath(image_dir)}")
#
#         image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
#         image_files = []
#         for root, _, files in os.walk(image_dir):  # 递归遍历子目录
#             for filename in files:
#                 ext = os.path.splitext(filename)[1].lower()
#                 if ext in image_extensions:
#                     image_files.append(os.path.join(root, filename))
#
#         if not image_files:
#             print(f"警告: 目录 {os.path.abspath(image_dir)} 中未找到图片")
#             return [], 0.0, 0.0
#
#         print(f"找到 {len(image_files)} 张图片，开始处理...")
#         results = []
#         total_inference_time = 0.0
#         total_process_time = time.time()
#
#         for i, img_path in enumerate(image_files, 1):
#             try:
#                 result, infer_time = self.predict_single_image(img_path)
#                 results.append(result)
#                 total_inference_time += infer_time
#             except Exception as e:
#                 print(f"处理 {os.path.basename(img_path)} 失败: {str(e)}")
#
#         total_process_time = time.time() - total_process_time
#         num_success = len(results)
#         avg_inference_time = total_inference_time / num_success if num_success else 0.0
#         avg_process_time = total_process_time / num_success * 1000 if num_success else 0.0
#
#         return results, total_process_time, avg_process_time
#
#     def save_predictions_to_csv(self, results: List[Dict[str, any]], output_file: str):
#         if not results:
#             raise ValueError("无预测结果可保存")
#         try:
#             output_dir = os.path.dirname(output_file)
#             if output_dir:
#                 os.makedirs(output_dir, exist_ok=True)
#             with open(output_file, 'w', newline='', encoding='utf-8') as f:
#                 fieldnames = ['filename', 'category_id', 'confidence']
#                 writer = csv.DictWriter(f, fieldnames=fieldnames)
#                 writer.writeheader()
#                 filtered_results = [
#                     {
#                         'filename': res['filename'],
#                         'category_id': res['category_id'],
#                         'confidence': res['confidence']
#                     }
#                     for res in results
#                 ]
#                 writer.writerows(filtered_results)
#             print(f"结果已保存到: {os.path.abspath(output_file)}")
#         except Exception as e:
#             raise RuntimeError(f"保存CSV失败: {str(e)}")
#
#
# def calculate_statistics(self, results: List[Dict[str, any]]) -> Dict[str, any]:
#     """计算预测结果的统计数据"""
#     if not results:
#         return {}
#
#     # 基础统计
#     confidences = [r['confidence'] for r in results]
#     categories = [r['category_id'] for r in results]
#
#     # 计算统计值
#     from collections import Counter
#     category_counts = Counter(categories)
#     most_common = category_counts.most_common(1)
#
#         return {
#             'total_samples': len(results),
#             'success_samples': len(results),
#             'avg_confidence': round(sum(confidences) / len(confidences), 6) if confidences else 0.0,
#             'confidence_std': round(np.std(confidences).item(), 6) if confidences else 0.0,
#             'max_confidence': round(max(confidences), 6) if confidences else 0.0,
#             'min_confidence': round(min(confidences), 6) if confidences else 0.0,
#             'most_frequent_category': most_common[0][0] if most_common else None,
#             'most_frequent_count': most_common[0][1] if most_common else 0,
#             'high_confidence_ratio': round(sum(1 for c in confidences if c >= 0.6) / len(confidences) * 100,
#                                            2) if confidences else 0.0,
#             'category_distribution': {
#                 'labels': [str(k) for k, _ in category_counts.most_common(10)],
#                 'counts': [v for _, v in category_counts.most_common(10)]
#             },
#             'confidence_distribution': self._get_confidence_bins(confidences)
#         }
#
#
# def _get_confidence_bins(self, confidences: List[float]) -> Dict[str, List[any]]:
#     """计算置信度区间分布"""
#     bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     counts, _ = np.histogram(confidences, bins=bins)
#     labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
#
#     return {
#         'labels': labels,
#         'counts': counts.tolist()
#     }
#
#
# # 初始化预测器（全局单例）
# try:
#     predictor = ImagePredictor(
#         num_classes=MODEL_CONFIG["num_classes"],
#         model_arch=MODEL_CONFIG["model_arch"],
#         class_indices_path=MODEL_CONFIG["class_indices_path"],
#         class_mapping_path=MODEL_CONFIG["class_mapping_path"],
#         weights_path=MODEL_CONFIG["weights_path"],
#         use_cbam=MODEL_CONFIG["use_cbam"]
#     )
#     print("预测器初始化成功，Flask服务准备就绪")
# except Exception as e:
#     print(f"预测器初始化失败: {e}")
#     import traceback
#
#     traceback.print_exc()
#     exit(1)
#
#
# # API 路由
# @app.route('/health', methods=['GET'])
# def health_check():
#     """健康检查接口"""
#     return jsonify({
#         'status': 'healthy',
#         'model_loaded': True,
#         'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
#     }), 200
#
#
# @app.route('/predict/single', methods=['POST'])
# def predict_single():
#     """单图预测接口"""
#     try:
#         # 检查是否有文件上传
#         if 'image' not in request.files:
#             return jsonify({'error': '未上传图片文件'}), 400
#
#         file = request.files['image']
#
#         # 检查文件名
#         if file.filename == '':
#             return jsonify({'error': '文件名不能为空'}), 400
#
#         # 检查文件格式
#         if not allowed_file(file.filename):
#             return jsonify({'error': f'不支持的文件格式，仅支持 {ALLOWED_EXTENSIONS}'}), 400
#
#         # 保存临时文件
#         filename = secure_filename(file.filename)
#         temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(temp_file_path)
#
#         # 执行预测
#         result, _ = predictor.predict_single_image(temp_file_path)
#
#         # 删除临时文件
#         os.remove(temp_file_path)
#
#         return jsonify({
#             'status': 'success',
#             'result': result
#         }), 200
#
#     except Exception as e:
#         return jsonify({
#             'status': 'error',
#             'error': str(e)
#         }), 500
#
#
# @app.route('/predict/batch', methods=['POST'])
# def predict_batch():
#     """批量预测接口（支持多文件上传）"""
#     try:
#         # 检查是否有文件上传
#         if 'images' not in request.files:
#             return jsonify({'error': '未上传图片文件'}), 400
#
#         files = request.files.getlist('images')
#
#         if not files:
#             return jsonify({'error': '未选择任何图片'}), 400
#
#         # 创建临时目录保存上传的图片
#         batch_temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
#
#         # 保存所有上传的图片
#         valid_files = []
#         for file in files:
#             if file and allowed_file(file.filename):
#                 filename = secure_filename(file.filename)
#                 file_path = os.path.join(batch_temp_dir, filename)
#                 file.save(file_path)
#                 valid_files.append(file_path)
#
#         if not valid_files:
#             shutil.rmtree(batch_temp_dir)
#             return jsonify({'error': '没有有效的图片文件'}), 400
#
#         # 执行批量预测
#         results, total_time, avg_time = predictor.predict_multiple_images(batch_temp_dir)
#
#         # 计算统计信息
#         stats = predictor.calculate_statistics(results)
#
#         # 清理临时目录
#         shutil.rmtree(batch_temp_dir)
#
#         # 检查是否需要返回CSV文件
#         return_csv = request.args.get('return_csv', 'false').lower() == 'true'
#
#         if return_csv:
#             # 生成临时CSV文件
#             csv_temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, encoding='utf-8')
#             csv_path = predictor.save_predictions_to_csv(results, csv_temp_file.name)
#
#             # 返回CSV文件
#             return send_file(
#                 csv_path,
#                 mimetype='text/csv',
#                 as_attachment=True,
#                 download_name='prediction_results.csv'
#             )
#         else:
#             return jsonify({
#                 'status': 'success',
#                 'total_images': len(valid_files),
#                 'success_count': len(results),
#                 'total_process_time_s': round(total_time, 2),
#                 'avg_process_time_ms': round(avg_time, 2),
#                 'statistics': stats,
#                 'results': results
#             }), 200
#
#     except Exception as e:
#         return jsonify({
#             'status': 'error',
#             'error': str(e)
#         }), 500
#
#
# @app.route('/info', methods=['GET'])
# def model_info():
#     """获取模型信息接口"""
#     return jsonify({
#         'model_config': {
#             'num_classes': MODEL_CONFIG['num_classes'],
#             'model_arch': MODEL_CONFIG['model_arch'],
#             'image_size': f"{predictor.im_width}x{predictor.im_height}",
#             'use_cbam': MODEL_CONFIG['use_cbam']
#         },
#         'class_count': len(predictor.class_indict),
#         'categories': list(predictor.name_to_category_id.keys())
#     }), 200
#
#
# # 错误处理
# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({'error': '接口不存在'}), 404
#
#
# @app.errorhandler(413)
# def request_entity_too_large(error):
#     return jsonify({'error': '上传文件过大，最大支持100MB'}), 413
#
#
# @app.errorhandler(500)
# def internal_server_error(error):
#     return jsonify({'error': '服务器内部错误'}), 500
#
#
# if __name__ == '__main__':
#     # 启动Flask服务
#     # 注意：生产环境请使用gunicorn等WSGI服务器，不要使用app.run()
#     app.run(
#         host='0.0.0.0',  # 允许外部访问
#         port=5000,  # 端口号
#         debug=False,  # 生产环境禁用debug
#         threaded=True  # 启用多线程
#     )