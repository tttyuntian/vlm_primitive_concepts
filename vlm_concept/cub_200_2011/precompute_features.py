import argparse
import os
import logging

import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image

from .cub_data import load_data

device = "cuda" if torch.cuda.is_available() else "cpu"


def parseArguments(): 
    parser = argparse.ArgumentParser()

    # Necessary variables
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report_step", type=int, default=100)
    parser.add_argument("--data_root", type=str, default="./data/CUB_200_2011")
    parser.add_argument("--output_path", type=str, default="./outputs/cub_200_2011/precompute_features")
    args = parser.parse_args()
    return args

def get_logger(args):
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./logs/cub_200_2011", exist_ok=True)
    os.makedirs("./logs/cub_200_2011/precompute_features", exist_ok=True)
    logging.basicConfig(level=logging.INFO, \
            format = "%(asctime)s %(levelname)s: %(message)s", \
            datefmt = "%m/%d %H:%M:%S %p", \
            filename = "./logs/cub_200_2011/precompute_features/precompute_features.log", \
            filemode = "w"
    )
    return logging.getLogger(__name__)

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

def main(args):
    # Load data
    logger.info("Load data...")
    data, _, _ = load_data(args, split="all")

    # Load the model
    logger.info("Load model...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Precompute image features and class activations
    logger.info("Start precomputing features!")
    image_features_list_train, class_activations_train = [],[]
    image_features_list_valid, class_activations_valid = [],[]

    # Prepare text inputs
    text_inputs_raw = [f"the bird is {class_name}" for class_name in data.class_name.unique()]
    text_inputs = torch.cat([clip.tokenize(text) for text in text_inputs_raw]).to(device)

    for row_id in range(len(data)):
        # Prepare image inputs
        image_id, image_name, is_training = data.iloc[row_id][["image_id", "filepath", "is_training_image"]]
        image_path = os.path.join(args.data_root, "CUB_200_2011", "images", image_name)
        image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        # Calculate featuress
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Pick k most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T)

        image_features = image_features.detach().cpu().numpy()
        class_activation = similarity.detach().cpu().numpy()

        if is_training:
            image_features_list_train.append(image_features)
            class_activations_train.append(class_activation)
        else:
            image_features_list_valid.append(image_features)
            class_activations_valid.append(class_activation)
        
        if args.verbose and (row_id % args.report_step == 0):
            logger.info(f"Progress: {row_id} / {len(data)}")
    
    image_features_list_train = np.concatenate(image_features_list_train)
    image_features_list_valid = np.concatenate(image_features_list_valid)
    class_activations_train = np.concatenate(class_activations_train)
    class_activations_valid = np.concatenate(class_activations_valid)

    # Output results
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./outputs/cub_200_2011", exist_ok=True)
    os.makedirs("./outputs/cub_200_2011/precompute_features", exist_ok=True)
    np.save(os.path.join(args.output_path, "image_features_train.npy"), image_features_list_train)
    np.save(os.path.join(args.output_path, "image_features_valid.npy"), image_features_list_valid)
    np.save(os.path.join(args.output_path, "class_activations_train.npy"), class_activations_train)
    np.save(os.path.join(args.output_path, "class_activations_valid.npy"), class_activations_valid)
    logger.info(f"Features output to {args.output_path}")
    logger.info("Done!")

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)

    main(args)
