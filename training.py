from ultralytics import YOLO
model = YOLO("yolov8x.pt") # load the model
results = model.train(data="coco128.yaml", epochs=10)
model.val()  # run YOLOv5x on COCO val2017
model.save("trainedModel.pt")  # save model to trainedModel.pt