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
    args = parser.parse_args()

    # I/O variables
    output_name, output_path = get_output_name(args)
    parser.add_argument("--output_name", type=str, default=output_name)
    parser.add_argument("--output_path", type=str, default=output_path)
    args = parser.parse_args()
    return args

def get_logger(args):
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./logs/mit_states", exist_ok=True)
    os.makedirs(f"./logs/mit_states/{args.output_name}", exist_ok=True)
    logging.basicConfig(level=logging.INFO, \
            format = "%(asctime)s %(levelname)s: %(message)s", \
            datefmt = "%m/%d %H:%M:%S %p", \
            filename = f"./logs/mit_states/{args.output_name}/{args.output_name}.log", \
            filemode = "w"
    )
    return logging.getLogger(__name__)

def get_output_name(args):
    output_name = f"concept_cls_adj_noun_separate"
    output_path = f"./outputs/mit_states/{output_name}"
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./outputs/mit_states", exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    return output_name, output_path

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

def main(args):
    # Load data
    logger.info("Load data...")
    data, antonym_dict = load_data(args)

    # Load the model
    logger.info("Load model...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Zero-shot concept classification - 312 retrieval task
    logger.info("Start zero-shot concept classification!")
    unique_noun_list = data.noun.unique()
    unique_adjective_list = data.adjective.unique()
    image_id_list, adjective_list, adjective_id_list, noun_list, noun_id_list = [],[],[],[],[]
    pred_adjective_list, pred_adjective_id_list, pred_noun_list, pred_noun_id_list = [],[],[],[]
    for row_id in range(len(data)):
        # Get necessary variables
        image_id, adjective_id, noun_id, image_path = data.iloc[row_id][["image_id", "adjective_id", "noun_id", "image_path"]]
        label_adj, label_noun = data.iloc[row_id][["adjective", "noun"]]
        
        # Prepare text inputs
        antonyms = antonym_dict[label_adj]
        if len(antonyms) == 0:
            # Only carry out retrieval task when there are antonyms
            continue
        else:
            adjective_candidates = antonyms + [label_adj]
            text_inputs_raw_adj = [
                f"this is {curr_adj}"
                for curr_adj in adjective_candidates
            ]
            text_inputs_raw_noun = [
                f"this is {curr_noun}"
                for curr_noun in unique_noun_list
            ]
        
        # Prepare image inputs
        image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            
        # Multiclass concept classification
        # Calculate features
        text_inputs_adj = clip.tokenize(text_inputs_raw_adj).to(device)
        text_inputs_noun = clip.tokenize(text_inputs_raw_noun).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features_adj = model.encode_text(text_inputs_adj)
            text_features_noun = model.encode_text(text_inputs_noun)

        # Adjective classification
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features_adj /= text_features_adj.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features_adj.T).softmax(dim=-1)
        values, indices = similarity[0].sort(descending=True)
        pred_id = indices[0].item()
        pred = text_inputs_raw_adj[pred_id]
        pred_adjective = pred.split()[-1]
        pred_adjective_id = np.where(unique_adjective_list==pred_adjective)[0][0]

        # Noun classification
        text_features_noun /= text_features_noun.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features_noun.T).softmax(dim=-1)
        values, indices = similarity[0].sort(descending=True)
        pred_id = indices[0].item()
        pred = text_inputs_raw_noun[pred_id]
        pred_noun = pred.split()[-1]
        pred_noun_id = np.where(unique_noun_list==pred_noun)[0][0]
        
        image_id_list.append(image_id)
        adjective_list.append(label_adj)
        adjective_id_list.append(adjective_id)
        noun_list.append(label_noun)
        noun_id_list.append(noun_id)
        pred_adjective_list.append(pred_adjective)
        pred_adjective_id_list.append(pred_adjective_id)
        pred_noun_list.append(pred_noun)
        pred_noun_id_list.append(pred_noun_id)
        
        if args.verbose and (row_id % args.report_step == 0):
            logger.info(f"Progress: {row_id} / {len(data)}")
    
    # Output results
    df = pd.DataFrame(
        {
            "image_id": image_id_list,
            "adjective": adjective_list,
            "adjective_id": adjective_id_list,
            "noun": noun_list,
            "noun_id": noun_id_list,
            "pred_adjective": pred_adjective_list,
            "pred_adjective_id": pred_adjective_id_list,
            "pred_noun": pred_noun_list,
            "pred_noun_id": pred_noun_id_list,
        }
    )
    df.to_csv(os.path.join(args.output_path, f"predictions.csv"), index=False)
    logger.info(f"Predictions output to {args.output_path}")
    logger.info("Done!")

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)

    main(args)
