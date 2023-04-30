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
    args = parser.parse_args()

    # I/O variables
    output_name, output_path = get_output_name(args)
    parser.add_argument("--output_name", type=str, default=output_name)
    parser.add_argument("--output_path", type=str, default=output_path)
    args = parser.parse_args()
    return args

def get_logger(args):
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./logs/cub_200_2011", exist_ok=True)
    os.makedirs(f"./logs/cub_200_2011/{args.output_name}", exist_ok=True)
    logging.basicConfig(level=logging.INFO, \
            format = "%(asctime)s %(levelname)s: %(message)s", \
            datefmt = "%m/%d %H:%M:%S %p", \
            filename = f"./logs/cub_200_2011/{args.output_name}/{args.output_name}.log", \
            filemode = "w"
    )
    return logging.getLogger(__name__)

def get_output_name(args):
    output_name = f"precompute_attribute_activations"
    output_path = f"./outputs/cub_200_2011/{output_name}"
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./outputs/cub_200_2011", exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    return output_name, output_path

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

def main(args):
    # Load data
    logger.info("Load data...")
    data, attributes, unique_attributes = load_data(args, split="all")

    # Load the model
    logger.info("Load model...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Precompute attribute activations
    logger.info("Start precomputing attribute activations!")
    attribute_activations_train, attribute_activations_valid = [],[]
    for row_id in range(len(data)):
        # Prepare image inputs
        image_id, class_id, image_name, is_training = data.iloc[row_id][["image_id", "class_id", "filepath", "is_training"]]
        image_path = os.path.join(args.data_root, "CUB_200_2011", "images", image_name)
        image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_attributes = attributes[attributes.image_id==image_id]
        
        curr_attr_actv = []
        for uni_attr in unique_attributes:
            curr_attr_df = image_attributes[image_attributes.attribute_template==uni_attr]

            # Prepare text inputs
            text_inputs_raw = [
                "a photo of bird whose {} is {}".format(attr.replace("has ", ""), label)
                for attr, label in zip(curr_attr_df["attribute_template"], curr_attr_df["attribute_label"])
            ]

            # Calculate attribute activations
            text_inputs = clip.tokenize(text_inputs_raw).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)

            # Pick k most similar labels for the image
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            similarity = similarity[0].detach().cpu().numpy()
            
            curr_attr_actv.append(similarity)
        
        curr_attr_actv = np.concatenate(curr_attr_actv)
        
        if is_training:
            attribute_activations_train.append(curr_attr_actv)
        else:
            attribute_activations_valid.append(curr_attr_actv)
        
        if args.verbose and (row_id % args.report_step == 0):
            logger.info(f"Progress: {row_id} / {len(data)}")

    # Output results
    attribute_activations_train = np.array(attribute_activations_train)
    attribute_activations_valid = np.array(attribute_activations_valid)
    np.save(os.path.join(args.output_path, f"attribute_activations_train.npy"), attribute_activations_train)
    np.save(os.path.join(args.output_path, f"attribute_activations_valid.npy"), attribute_activations_valid)
    logger.info(f"Features output to {args.output_path}")
    logger.info("Done!")

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)

    main(args)
