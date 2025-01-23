
import cv2
import torch

model = torch.hub.load('ultralytics/yolov5','yolov5s')

img = cv2.imread('street.jpg')
results =model(img)
print(results)
outputs =cv2.resize(results.render()[0],(img.shape[1],img.shape[0]))
print(outputs.shape)

cv2.imshow('output',outputs)
cv2.waitKey(0)
cv2.destroyAllWindows()

