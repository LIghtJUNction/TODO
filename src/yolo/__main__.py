from ultralytics import YOLO
# model = YOLO("yolo11n.yaml")
model = YOLO("yolo11n.pt")

# 识别指定标签
results = model(source="0",show=True, classes=[0] )
