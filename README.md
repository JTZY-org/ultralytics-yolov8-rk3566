# YOLOv8 训练与模型导出指南

本文介绍本项目从环境配置、数据集制作、模型训练到 ONNX 模型导出的完整流程。以下命令均在项目根目录 `ultralytics_yolov8` 下执行。

## 1. 配置 Python 环境

建议使用 Python 3.8 或更高版本，并通过 Conda 创建独立环境：

```bash
conda create -n yolov8 python=3.10 -y
conda activate yolov8
```

根据本机 CUDA 版本安装 PyTorch，具体命令可参考 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)。然后以可编辑模式安装本项目及 ONNX 导出依赖：

```bash
pip install -e ".[export]"
```

安装完成后可执行以下命令检查环境：

```bash
python -c "import torch; import ultralytics; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 2. 制作数据集

数据集可参考项目中的 `dataset` 目录进行组织。每个数据集划分都包含一个 `images` 目录和一个 `labels` 目录：

```text
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

图片与标签文件必须同名。例如：

```text
images/example.jpg
labels/example.txt
```

检测任务的每一行标签采用 YOLO 格式：

```text
class_id x_center y_center width height
```

其中坐标和宽高均为相对于图片尺寸归一化后的数值，范围为 `0` 到 `1`；类别编号 `class_id` 从 `0` 开始。

### 配置 `data.yaml`

打开 `dataset/data.yaml`，修改训练集、验证集、测试集路径，以及类别数量和类别名称：

```yaml
train: D:/code/ultralytics_yolov8/dataset/train
val: D:/code/ultralytics_yolov8/dataset/valid
test: D:/code/ultralytics_yolov8/dataset/test

nc: 2
names: [cube, person]
```

配置时请注意：

- `train`、`val` 和 `test` 分别指向对应的数据集目录。
- `nc` 是需要识别的类别总数。
- `names` 按类别编号顺序填写，数量必须与 `nc` 一致。
- Windows 路径建议使用正斜杠 `/`，也可以使用完整绝对路径。
- `test` 为可选项；没有测试集时可以删除该配置。

## 3. 训练模型

项目根目录的 `train.py` 是训练入口。首先修改模型配置文件和数据集配置文件的路径：

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"D:\code\ultralytics_yolov8\ultralytics\cfg\models\v8\yolov8m.yaml")
    results = model.train(
        data=r"D:\code\ultralytics_yolov8\dataset\data.yaml",
        cache=False,
        imgsz=640,
        epochs=100,
        single_cls=False,
        batch=8,
        close_mosaic=10,
        workers=0,
        device="0",
        optimizer="SGD",
        project="runs/train",
        name="exp",
    )
```

如需训练不同规模的模型，可将 `YOLO(...)` 中的模型配置替换为：

| 模型规模 | 配置文件 |
| --- | --- |
| YOLOv8n | `ultralytics/cfg/models/v8/yolov8n.yaml` |
| YOLOv8s | `ultralytics/cfg/models/v8/yolov8s.yaml` |
| YOLOv8m | `ultralytics/cfg/models/v8/yolov8m.yaml` |

常用训练参数说明：

| 参数 | 说明 |
| --- | --- |
| `data` | `data.yaml` 的路径 |
| `imgsz` | 训练图片尺寸 |
| `epochs` | 训练轮数 |
| `batch` | 每个批次的图片数量，显存不足时可减小 |
| `device` | GPU 编号，例如 `"0"`；仅使用 CPU 时设为 `"cpu"` |
| `workers` | 数据加载进程数；Windows 下遇到多进程问题时可设为 `0` |
| `project` | 训练结果的保存目录 |
| `name` | 本次实验名称 |

根据实际数据量和硬件条件调整参数后，开始训练：

```bash
python train.py
```

默认情况下，训练结果保存在 `runs/train/exp`（若目录已存在，程序会自动生成新的实验目录）。模型权重位于实验目录的 `weights` 文件夹中：

```text
runs/train/exp/weights/best.pt
runs/train/exp/weights/last.pt
```

通常优先使用验证指标最好的 `best.pt` 进行部署。

## 4. 将 PT 模型导出为 ONNX

训练完成后，可以将生成的 `.pt` 权重转换为 `.onnx` 模型。

打开 `ultralytics/cfg/default.yaml`，至少修改以下两项：

```yaml
model: D:/code/ultralytics_yolov8/runs/train/exp/weights/best.pt
format: onnx
```

- `model` 设置为需要导出的 `.pt` 模型路径。
- `format` 必须设置为 `onnx`。本仓库当前默认值可能是其他格式，只修改模型路径无法保证生成 ONNX 文件。

如有需要，还可在该文件的 Export settings 部分调整 `imgsz`、`opset`、`simplify`、`dynamic` 等导出参数。

配置完成后，在项目根目录执行：

```bash
python ./ultralytics/engine/exporter.py
```

导出成功后，`.onnx` 文件将生成在所选 `.pt` 模型所在的目录中。例如：

```text
runs/train/exp/weights/best.onnx
```

也可以不修改全局默认配置，直接使用命令行导出：

```bash
yolo mode=export model="runs/train/exp/weights/best.pt" format=onnx
```

## 完整流程速查

```text
配置 Python 环境
        ↓
按照 YOLO 格式制作并划分数据集
        ↓
修改 dataset/data.yaml
        ↓
修改并运行 train.py
        ↓
获取 runs/train/.../weights/best.pt
        ↓
配置 default.yaml 或使用 yolo export 命令
        ↓
生成 best.onnx
```
