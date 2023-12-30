import cv2
import numpy as np
import os
import torch
import argparse

from facexlib.detection import init_detection_model

def main(args):
    det_net = init_detection_model('retinaface_resnet50')
    det_net.to(args.device).eval()
    img = cv2.imread(args.img_path)
    with torch.no_grad():
        bboxes, warped_face_list = det_net.align_multi(img, 0.97)
    img = warped_face_list[0]
    img = cv2.resize(img, (1024, 1024))
    cv2.imwrite(args.save_path,img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='assets/test2.jpg')
    parser.add_argument('--save_path', type=str, default='test_alignment.png')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)
