import cv2
import glob
import numpy as np
import os
import torch
from PIL import Image
from tqdm import tqdm

from facexlib.detection import init_detection_model

def main(args):
    det_net = init_detection_model('retinaface_resnet50')
    half = False
    det_net.to(args.device).eval()
    img = cv2.imread(args.img_path)
    with torch.no_grad():
        bboxes, warped_face_list = det_net.align_multi(img, 0.97, half=half)
    print(bboxes.shape,bboxes)
    print(len(warped_face_list))
    cv2.imwrite(args.save_path,warped_face_list[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='assets/test2.jpg')
    parser.add_argument('--save_path', type=str, default='test_alignment.png')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)
