from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO(r'D:\code\ultralytics_yolov8\ultralytics\cfg\models\v8\yolov8m.yaml')
    results = model.train(data=r'D:\code\ultralytics_yolov8\dataset\data.yaml',  
                           cache=False,
                           imgsz=640,
                           epochs=100,
                           single_cls=False,  # 是否是单类别检测
                           batch=8,
                           close_mosaic=10,
                           workers=0,
                           device='0',
                           optimizer='SGD',
                           
                           project='runs/train',
                           name='exp',)