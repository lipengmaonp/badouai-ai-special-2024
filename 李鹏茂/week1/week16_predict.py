import glob
import numpy as np
import torch
import cv2
from model.week16_unet_model import UNet

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "")
    net = UNet().to(device)
    net.to(device=device)
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    net = net.eval()
    tests_path = glob.glob('data/test/*.png')
    for path in tests_path:
        save_res_path = path.split('.')[0] + '_res.png'
        img = cv2.imread()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        imgtensor = torch.from_numpy(img)
        imgtensor = imgtensor.to(device=device, dtype=torch.float32)
        pred = net(imgtensor)
        print(pred.shape)
        pred = np.array(pred.data.cpu()[0])[0]
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        cv2.imwrite(save_res_path, pred)

