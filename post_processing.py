import argparse
import os
from pathlib import Path

import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Post processing")

parser.add_argument("-i", "--input_path", type=str, default="./output/model_output")
parser.add_argument("-o", "--output_path", type=str, default="./output/post_processed")
parser.add_argument("--otsu", action="store_true")
parser.add_argument("--thresh", type=int, default=127)

args = parser.parse_args()

input = args.input_path
output = args.output_path

output_dir = Path(args.output_path)
output_dir.mkdir(exist_ok=True, parents=True)


def binarize_image(input_path, output_path, file_name=None):
    # read the image file
    im_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Binarize using Otsu method
    if args.otsu:
        (thresh, im_bw) = cv2.threshold(
            im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
    else:
        thresh = args.thresh
        im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    path_to_output = os.path.join(output_path, file_name)
    # Save to disk
    cv2.imwrite(path_to_output, im_bw)


def binarize_folder(path_to_folder, output_path):
    list_names = os.listdir(path_to_folder)
    for name in tqdm(list_names):
        print(f"Processing the file {name}...")
        path_to_file = os.path.join(path_to_folder, name)
        binarize_image(path_to_file, output_path, name)


if __name__ == "__main__":
    binarize_folder(input, output)
