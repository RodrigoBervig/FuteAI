import ultralytics as ua 


model = ua.YOLO('../best.pt')

results = model.predict('jogo.mp4', save = True)