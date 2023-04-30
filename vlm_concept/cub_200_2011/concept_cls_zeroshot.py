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
    parser.add_argument("--split", type=str, choices=["train", "valid"])
    parser.add_argument("--data_root", type=str, default="./data/CUB_200_2011")
    parser.add_argument("--output_path", type=str, default="./outputs/cub_200_2011/concept_cls_zeroshot")
    args = parser.parse_args()
    return args

def get_logger(args):
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./logs/cub_200_2011", exist_ok=True)
    os.makedirs("./logs/cub_200_2011/concept_cls_zeroshot", exist_ok=True)
    logging.basicConfig(level=logging.INFO, \
            format = "%(asctime)s %(levelname)s: %(message)s", \
            datefmt = "%m/%d %H:%M:%S %p", \
            filename = f"./logs/cub_200_2011/concept_cls_zeroshot/concept_cls_zeroshot_{args.split}.log", \
            filemode = "w"
    )
    return logging.getLogger(__name__)

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

def main(args):
    # Load data
    logger.info("Load data...")
    data, attributes, unique_attributes = load_data(args, split=args.split)

    # Load the model
    logger.info("Load model...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Zero-shot concept classification - 312 retrieval task
    logger.info("Start zero-shot concept classification!")
    topk = 1

    image_id_list, class_id_list, attribute_list, certainty_list = [],[],[],[]
    label_value_list, label_rank_list, pred_list, label_list = [],[],[],[]
    for row_id in range(len(data)):
        # Prepare image inputs
        image_id, class_id, image_name = data.iloc[row_id][["image_id", "class_id", "filepath"]]
        image_path = os.path.join(args.data_root, "CUB_200_2011", "images", image_name)
        image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_attributes = attributes[attributes.image_id==image_id]
        
        num_presented_attributes = 0
        for uni_attr in unique_attributes:
            curr_attr_df = image_attributes[image_attributes.attribute_template==uni_attr]
            
            if any(curr_attr_df.is_present):
                # Only evaluate when an attribute is presented in the image
                num_presented_attributes += 1
                curr_attr_label = np.where(curr_attr_df.is_present)[0][0]
                
                # Prepare text inputs
                text_inputs_raw = [
                    "a photo of bird whose {} is {}".format(attr.replace("has ", ""), label)
                    for attr, label in zip(curr_attr_df["attribute_template"], curr_attr_df["attribute_label"])
                ]

                # Multiclass concept classification
                # Calculate features
                text_inputs = clip.tokenize(text_inputs_raw).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    text_features = model.encode_text(text_inputs)

                # Pick k most similar labels for the image
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].sort(descending=True)
                pred = indices[0].item()
                label_value = similarity[0][curr_attr_label].item()
                label_rank = (indices==curr_attr_label).nonzero()[0][0].item() + 1  # Index to 1
                certainty = curr_attr_df.certainty_id.mean()
                
                image_id_list.append(image_id)
                class_id_list.append(class_id)
                attribute_list.append(uni_attr)
                label_value_list.append(label_value)
                label_rank_list.append(label_rank)
                pred_list.append(pred)
                label_list.append(curr_attr_label)
                certainty_list.append(certainty)
        
        if args.verbose and (row_id % args.report_step == 0):
            logger.info(f"Progress: {row_id} / {len(data)}")
    
    # Output results
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./outputs/cub_200_2011", exist_ok=True)
    os.makedirs("./outputs/cub_200_2011/concept_cls_zeroshot", exist_ok=True)
    df = pd.DataFrame(
        {
            "image_id": image_id_list,
            "class_id": class_id_list,
            "attribute": attribute_list,
            "pred": pred_list,
            "label": label_list,
            "label_value": label_value_list,
            "label_rank": label_rank_list,
            "certainty_id": certainty_list,
        }
    )
    df.to_csv(os.path.join(args.output_path, f"predictions_{args.split}.csv"), index=False)
    logger.info(f"Predictions output to {args.output_path}")
    logger.info("Done!")

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)

    main(args)
