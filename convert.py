from ultralytics import YOLO

# 1. 加载你的权重文件（可以是 yolov8n.pt，或者你自己训练出来的 best.pt）
model = YOLO("yolov8n.pt") 

# 2. 导出为兼容 RKNN 的静态 ONNX 模型
# 这里的 rknn=True 是核心参数，它会自动为瑞芯微 NPU 开启 9 输出优化
model.export(
    format="onnx", 
    rknn=True,        # 必须设为 True
    opset=12,         # 建议设为 12 获得最佳兼容性
    dynamic=False     # 必须设为 False（固定输入尺寸）
)