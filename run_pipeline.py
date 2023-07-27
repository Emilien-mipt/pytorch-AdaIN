import argparse
import subprocess
import sys
from collections import namedtuple

import yaml


def convert_dict_to_tuple(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_dict_to_tuple(value)
    return namedtuple("GenericDict", dictionary.keys())(**dictionary)


def main(args: argparse.Namespace):
    with open(args.cfg) as f:
        config_data = yaml.safe_load(f)
    config = convert_dict_to_tuple(config_data)

    print("Cropping the face...")
    if config.crop.apply:
        subprocess.run(
            [
                "python",
                "data_preprocessing.py",
                "--mode=crop",
                f"-i={config.crop.input}",
                f"-o={config.crop.output}",
                "--save_ratio",
                "--fill_black",
                f"--size={config.crop.crop_size}",
                f"--g_beta={config.crop.g_beta}",
            ]
        )
    print("Face crop is ready!")

    print("Removing the background...")
    if config.delete_bg.apply:
        subprocess.run(
            [
                "python",
                "data_preprocessing.py",
                "--mode=delete_bg",
                f"-i={config.delete_bg.input}",
                f"-o={config.delete_bg.output}",
            ]
        )
    print("Background removal completed!")

    print("Face superresolution...")
    if config.superres.apply:
        subprocess.run(
            [
                "python",
                "inference_codeformer.py",
                f"-i={config.superres.input}",
                f"-o={config.superres.output}",
                f"-s={config.superres.upscale}",
            ]
        )
    print("Face superresolution completed!")

    print("Style transfer...")
    subprocess.run(
        [
            "python",
            "test.py",
            f"--content_dir={config.adain.content_dir}",
            f"--style={config.adain.style}",
            f"--decoder={config.adain.decoder}",
            f"--content_size={config.adain.content_size}",
            f"--style_size={config.adain.style_size}",
            f"--output={config.adain.output}",
        ]
    )
    print("Style transfer completed!")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="./config/inference_config.yaml",
        help="Path to config file.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
