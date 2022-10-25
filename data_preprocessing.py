"""
Code is taken and modified from
https://github.com/wtomin/Multitask-Emotion-Recognition-with-Incomplete-Labels/blob/master/MTCNN_alignment_with_video.py
Reading all videos in given dir (including subdirs and subsubdirs)
return the aligend face images of each video
"""

import argparse
import math
import os
import time

import cv2
import numpy as np
import pandas
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description="MTCNN video face preprocessing")
parser.add_argument("-i", "--input_dir", type=str, default="./data/content_dir")
parser.add_argument(
    "-o", "--output_dir", type=str, default="./data/content_dir_processed/"
)
parser.add_argument(
    "-l", "--landmarks_dir", type=str, default="./dataset/landmark/ALL/"
)
parser.add_argument(
    "--alignment", action="store_true", help="default: no face alignment"
)
parser.add_argument("--size", type=int, default=512, help="face size nxn")
parser.add_argument("--g_beta", type=float, default=2.0, help="face size nxn")
parser.add_argument(
    "--save_fl", action="store_true", help="default: do not save facial landmarks"
)
parser.add_argument(
    "--fill_black",
    action="store_true",
    help="default: do not fill black regions with white color",
)
parser.add_argument(
    "-q",
    "--quiet",
    action="store_true",
    help="whether to output face detection results",
)
parser.add_argument("--gpu_id", type=int, default=0, help="Choose gpu id for inference")
args = parser.parse_args()

gpu_id = args.gpu_id
rotate = args.alignment
size = args.size
quiet = args.quiet
save_landmarks = args.save_fl

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

detector = MTCNN(select_largest=False, device=device)


def image_rote(img, angle, elx, ely, erx, ery, mlx, mly, mrx, mry, expand=1):
    w, h = img.size
    img = img.rotate(angle, expand=expand)  # whether to expand after rotation
    if expand == 0:
        elx, ely = pos_transform_samesize(angle, elx, ely, w, h)
        erx, ery = pos_transform_samesize(angle, erx, ery, w, h)
        mlx, mly = pos_transform_samesize(angle, mlx, mly, w, h)
        mrx, mry = pos_transform_samesize(angle, mrx, mry, w, h)
    if expand == 1:
        elx, ely = pos_transform_resize(angle, elx, ely, w, h)
        erx, ery = pos_transform_resize(angle, erx, ery, w, h)
        mlx, mly = pos_transform_resize(angle, mlx, mly, w, h)
        mrx, mry = pos_transform_resize(angle, mrx, mry, w, h)
    return img, elx, ely, erx, ery, mlx, mly, mrx, mry


def calculate_angle(elx, ely, erx, ery):
    """
    Calculate image rotate angle
    :param elx: lefy eye x
    :param ely: left eye y
    :param erx: right eye x
    :param ery: right eye y
    :return: rotate angle
    """
    dx = erx - elx
    dy = ery - ely
    angle = math.atan(dy / dx) * 180 / math.pi
    return angle


def pos_transform_resize(angle, x, y, w, h):
    """
    After rotation, new coordinate with expansion
    :param angle:
    :param x:
    :param y:
    :param w:
    :param h:
    :return:
    """
    angle = angle * math.pi / 180
    matrix = [
        math.cos(angle),
        math.sin(angle),
        0.0,
        -math.sin(angle),
        math.cos(angle),
        0.0,
    ]

    def transform(x, y, matrix=matrix):
        (a, b, c, d, e, f) = matrix
        return a * x + b * y + c, d * x + e * y + f  # calculate output size

    xx = []
    yy = []
    for x_, y_ in ((0, 0), (w, 0), (w, h), (0, h)):
        x_, y_ = transform(x_, y_)
        xx.append(x_)
        yy.append(y_)
    ww = int(math.ceil(max(xx)) - math.floor(min(xx)))
    hh = int(math.ceil(max(yy)) - math.floor(min(yy)))
    # adjust center
    cx, cy = transform(w / 2.0, h / 2.0)
    matrix[2] = ww / 2.0 - cx
    matrix[5] = hh / 2.0 - cy
    tx, ty = transform(x, y)
    return tx, ty


def pos_transform_samesize(angle, x, y, w, h):
    """
    After rotation, new coordinate without expansion
    :param angle:
    :param x:
    :param y:
    :param w:
    :param h:
    :return:
    """
    angle = angle * math.pi / 180
    matrix = [
        math.cos(angle),
        math.sin(angle),
        0.0,
        -math.sin(angle),
        math.cos(angle),
        0.0,
    ]

    def transform(x, y, matrix=matrix):
        (a, b, c, d, e, f) = matrix
        return a * x + b * y + c, d * x + e * y + f

    cx, cy = transform(w / 2.0, h / 2.0)
    matrix[2] = w / 2.0 - cx
    matrix[5] = h / 2.0 - cy
    x, y = transform(x, y)
    return x, y


def PIL_image_convert(cv2_im):
    cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    return pil_im


def crop_face(
    image, file_name=None, rotate=True, size=512, quiet_mode=True, view_landmarks=False
):
    height, width, channels = image.shape  # cv2 image
    boxes, probs, landmarks = detector.detect(image, landmarks=True)
    image = PIL_image_convert(image)

    if boxes is None or boxes.shape[1] == 0:
        if not quiet_mode:
            print("***No Face detected. ***")
        return None, None
    if len(boxes) > 1:
        if not quiet_mode:
            print("*** Multi Faces ,get the face with largest confidence ***")
    detected_keypoints = landmarks[0, :]  # bounding_box = boxes[0, :]

    keypoints = {
        "left_eye": detected_keypoints[0, :],
        "right_eye": detected_keypoints[1, :],
        "nose": detected_keypoints[2, :],
        "mouth_left": detected_keypoints[3, :],
        "mouth_right": detected_keypoints[4, :],
    }

    lex, ley = keypoints["left_eye"]
    rex, rey = keypoints["right_eye"]
    rmx, rmy = keypoints["mouth_right"]
    lmx, lmy = keypoints["mouth_left"]
    nex, ney = keypoints["nose"]

    # rotation using PIL image

    if rotate:
        angle = calculate_angle(lex, ley, rex, rey)
        image, lex, ley, rex, rey, lmx, lmy, rmx, rmy = image_rote(
            image, angle, lex, ley, rex, rey, lmx, lmy, rmx, rmy
        )
    eye_width = rex - lex  # distance between two eyes
    ecx, ecy = (lex + rex) / 2.0, (ley + rey) / 2.0  # the center between two eyes
    # mouth_width = rmx - lmx
    mcx, mcy = (lmx + rmx) / 2.0, (lmy + rmy) / 2.0  # mouth center coordinate
    em_height = mcy - ecy  # height between mouth center to eyes center
    fcx, fcy = (ecx + mcx) / 2.0, (ecy + mcy) / 2.0  # face center
    # face
    if eye_width > em_height:
        alpha = eye_width
    else:
        alpha = em_height
    g_beta = args.g_beta
    g_left = fcx - alpha / 2.0 * g_beta
    g_upper = fcy - alpha / 2.0 * g_beta
    g_right = fcx + alpha / 2.0 * g_beta
    g_lower = fcy + alpha / 2.0 * g_beta
    g_face = image.crop((g_left, g_upper, g_right, g_lower))

    # Resize and save the face crop
    path_to_output = os.path.join(args.output_dir, file_name)

    # Fill black regions with white if needed
    if args.fill_black:
        bg = Image.new(
            "RGB",
            (
                int(max(g_right, g_left)) - int(min(g_right, g_left)),
                int(max(g_upper, g_lower)) - int(min(g_upper, g_lower)),
            ),
            (255, 255, 255),
        )
        bg.paste(image, (int(-g_left), int(-g_upper)))
        bg = bg.resize((size, size))
        bg.save(path_to_output)
    else:
        g_face = g_face.resize((size, size))
        g_face.save(path_to_output)

    # Redetect and save the landmarks
    new_boxes, new_probs, new_landmarks = detector.detect(g_face, landmarks=True)
    if new_boxes is None or new_boxes.shape[1] == 0:
        if not quiet_mode:
            print("***No Face detected. ***")
        return None, None
    if len(new_boxes) > 1:
        if not quiet_mode:
            print("*** Multi Faces ,get the face with largest confidence ***")
    new_detected_keypoints = new_landmarks[0, :]  # bounding_box = boxes[0, :]

    new_keypoints = {
        "left_eye": new_detected_keypoints[0, :],
        "right_eye": new_detected_keypoints[1, :],
        "nose": new_detected_keypoints[2, :],
        "mouth_left": new_detected_keypoints[3, :],
        "mouth_right": new_detected_keypoints[4, :],
    }

    new_lex, new_ley = new_keypoints["left_eye"]
    new_rex, new_rey = new_keypoints["right_eye"]
    new_nex, new_ney = new_keypoints["nose"]
    new_rmx, new_rmy = new_keypoints["mouth_right"]
    new_lmx, new_lmy = new_keypoints["mouth_left"]

    if view_landmarks:
        g_face = cv2.cvtColor(np.asarray(g_face), cv2.COLOR_RGB2BGR)
        cv2.circle(
            g_face,
            (int(new_lex), int(new_ley)),
            radius=5,
            color=(255, 0, 0),
            thickness=3,
        )
        cv2.circle(
            g_face,
            (int(new_rex), int(new_rey)),
            radius=5,
            color=(255, 0, 0),
            thickness=3,
        )
        cv2.circle(
            g_face,
            (int(new_nex), int(new_ney)),
            radius=5,
            color=(255, 0, 0),
            thickness=3,
        )
        cv2.circle(
            g_face,
            (int(new_rmx), int(new_rmy)),
            radius=5,
            color=(255, 0, 0),
            thickness=3,
        )
        cv2.circle(
            g_face,
            (int(new_lmx), int(new_lmy)),
            radius=5,
            color=(255, 0, 0),
            thickness=3,
        )
        cv2.imwrite(path_to_output, g_face)

    # Save the landmarks
    if save_landmarks:
        txt_name = ".".join((file_name.split(".")[0], "txt"))
        path_to_landmarks = os.path.join(args.landmarks_dir, txt_name)
        with open(path_to_landmarks, "w") as file:
            file.write(str(int(new_lex)) + " " + str(int(new_ley)) + "\n")
            file.write(str(int(new_rex)) + " " + str(int(new_rey)) + "\n")
            file.write(str(int(new_nex)) + " " + str(int(new_ney)) + "\n")
            file.write(str(int(new_lmx)) + " " + str(int(new_lmy)) + "\n")
            file.write(str(int(new_rmx)) + " " + str(int(new_rmy)))


def crop_folder(path_to_folder):
    list_names = os.listdir(path_to_folder)
    for name in tqdm(list_names):
        print(f"Processing the file {name}...")
        path_to_file = os.path.join(path_to_folder, name)
        img = cv2.imread(path_to_file)
        crop_face(img, file_name=name, rotate=rotate, size=size, quiet_mode=quiet)


def resize_img(image, file_name=None, size=size):
    height, width, channels = image.shape  # cv2 image
    print(height == width == 512)
    image = PIL_image_convert(image)
    # Resize and save the face crop
    path_to_output = os.path.join(args.output_dir, file_name)
    image = image.resize((size, size))
    image.save(path_to_output)


def resize_folder(path_to_folder):
    list_names = os.listdir(path_to_folder)
    for name in tqdm(list_names):
        print(f"Processing the file {name}...")
        path_to_file = os.path.join(path_to_folder, name)
        img = cv2.imread(path_to_file)
        resize_img(img, file_name=name, size=size)


if __name__ == "__main__":
    path_to_dir = args.input_dir
    crop_folder(path_to_dir)
