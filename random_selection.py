import argparse
import os
import random
import shutil

parser = argparse.ArgumentParser(
    description="Randomly select and copy data from one directory to another"
)
parser.add_argument("-i", "--input_dir", type=str)
parser.add_argument("-o", "--output_dir", type=str, default="./data/content_dir")
parser.add_argument("-n", "--number_files", type=int, default=1000)

args = parser.parse_args()


def select_and_copy(input_dir, output_dir):
    # Amount of random files you'd like to select
    print(f"Number of files in input: {len(os.listdir(input_dir))}")
    files = [
        file
        for file in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, file))
    ]
    random_amount = args.number_files
    for _ in range(random_amount):
        if len(files) == 0:
            print("No files to copy!")
            break
        else:
            file = random.choice(files)
            shutil.copy2(os.path.join(input_dir, file), output_dir)


def select_and_move(input_dir, output_dir):
    # Amount of random files you'd like to select
    print(f"Number of files in input: {len(os.listdir(input_dir))}")
    added_files = set()
    files = [
        file
        for file in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, file))
    ]
    random_amount = args.number_files
    for _ in range(random_amount):
        if len(files) == 0:
            print("No files to copy!")
            break
        else:
            file = random.choice(files)
            if file not in added_files:
                shutil.move(
                    os.path.join(input_dir, file), os.path.join(output_dir, file)
                )
                added_files.add(file)


if __name__ == "__main__":
    path_to_input = args.input_dir
    path_to_output = args.output_dir
    select_and_move(path_to_input, path_to_output)
