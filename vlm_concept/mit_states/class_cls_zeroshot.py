import argparse
import os
import logging

import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image

from .mit_states_data import load_data

device = "cuda" if torch.cuda.is_available() else "cpu"


def parseArguments(): 
    parser = argparse.ArgumentParser()

    # Necessary variables
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report_step", type=int, default=100)
    parser.add_argument("--data_root", type=str, default="./data/mit_states/release_dataset")
    parser.add_argument("--output_path", type=str, default="./outputs/mit_states/class_cls_zeroshot")
    args = parser.parse_args()
    return args

def get_logger(args):
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./logs/mit_states", exist_ok=True)
    os.makedirs("./logs/mit_states/class_cls_zeroshot", exist_ok=True)
    logging.basicConfig(level=logging.INFO, \
            format = "%(asctime)s %(levelname)s: %(message)s", \
            datefmt = "%m/%d %H:%M:%S %p", \
            filename = "./logs/mit_states/class_cls_zeroshot/class_cls_zeroshot.log", \
            filemode = "w"
    )
    return logging.getLogger(__name__)

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

def main(args):
    # Load data
    logger.info("Load data...")
    data, _ = load_data(args)

    # Load the model
    logger.info("Load model...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Zero-shot class classification - 312 retrieval task
    logger.info("Start zero-shot class classification!")

    image_id_list, adjective_list, adjective_id_list, noun_list, noun_id_list = [],[],[],[],[]
    label_value_list, label_rank_list, pred_list, label_list = [],[],[],[]

    # Prepare text inputs
    text_inputs_raw = [f"a photo of {noun}" for noun in data.noun.unique()]
    text_inputs = torch.cat([clip.tokenize(text) for text in text_inputs_raw]).to(device)

    for row_id in range(len(data)):
        # Prepare image inputs
        image_id, adjective_id, noun_id, image_path = data.iloc[row_id][["image_id", "adjective_id", "noun_id", "image_path"]]
        adjective, noun = data.iloc[row_id][["adjective", "noun"]]
        image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        
        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Pick k most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].sort(descending=True)
        pred = indices[0].item()
        label = noun_id
        label_value = similarity[0][label].item()
        label_rank = (indices==label).nonzero()[0][0].item() + 1  # Index to 1
        
        image_id_list.append(image_id)
        adjective_list.append(adjective)
        adjective_id_list.append(adjective_id)
        noun_list.append(noun)
        noun_id_list.append(noun_id)
        pred_list.append(pred)
        label_list.append(label)
        label_value_list.append(label_value)
        label_rank_list.append(label_rank)
        
        if args.verbose and (row_id % args.report_step == 0):
            logger.info(f"Progress: {row_id} / {len(data)}")
    
    # Output results
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./outputs/mit_states", exist_ok=True)
    os.makedirs("./outputs/mit_states/class_cls_zeroshot", exist_ok=True)
    df = pd.DataFrame(
        {
            "image_id": image_id_list,
            "adjective": adjective_list,
            "adjective_id": adjective_id_list,
            "noun": noun_list,
            "noun_id": noun_id_list,
            "pred": pred_list,
            "label": label_list,
            "label_value": label_value_list,
            "label_rank": label_rank_list,
        }
    )
    df.to_csv(os.path.join(args.output_path, "predictions.csv"), index=False)
    logger.info(f"Predictions output to {args.output_path}")
    logger.info("Done!")

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)

    main(args)
