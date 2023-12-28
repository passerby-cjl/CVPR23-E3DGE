import cv2
import torch

from facexlib.detection import init_detection_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

def main(args):

    input_img = args.img_path
    # initialize face helper
    face_helper = FaceRestoreHelper(
        upscale_factor=1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='jpg')

    face_helper.clean_all()

    det_net = init_detection_model('retinaface_resnet50', half=False)
    img = cv2.imread(input_img)
    with torch.no_grad():
        bboxes = det_net.detect_faces(img, 0.97)
        # x0, y0, x1, y1, confidence_score, five points (x, y)
    print(bboxes.shape)
    bboxes = bboxes[3]

    bboxes[0] -= 100
    bboxes[1] -= 100
    bboxes[2] += 100
    bboxes[3] += 100
    img = img[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2]), :]

    face_helper.read_image(img)
    # get face landmarks for each face
    face_helper.get_face_landmarks_5(only_center_face=True, pad_blur=False)
    # align and warp each face
    # save_crop_path = os.path.join(save_root, 'cropped_faces', img_name)
    save_crop_path = args.save_path
    face_helper.align_warp_face(save_crop_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='assets/test2.jpg')
    parser.add_argument('--save_path', type=str, default='test_alignment.png')
    args = parser.parse_args()

    main(args)
