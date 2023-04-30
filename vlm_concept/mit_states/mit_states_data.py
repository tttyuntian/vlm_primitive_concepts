import os
import numpy as np
import pandas as pd



def load_data(args, skip_adj=True):
    """
    Construct a dataframe of metadata of MIT States dataset.
    """
    adj_noun_pairs = os.listdir(os.path.join(args.data_root, "images"))
    adj_dict, noun_dict = {},{}
    adj_list, adj_id_list, noun_list, noun_id_list, image_path_list = [],[],[],[],[]

    for adj_noun_pair in adj_noun_pairs:
        # Collect path to every image under every adjective noun directory
        if "DS_Store" in adj_noun_pair:
            # Skip MacOS file in dataset
            continue

        adj, noun = adj_noun_pair.split()
        if (adj == "adj") and skip_adj:
            # Skip "adj" attribute representing a search of "<noun>" standalone
            continue
        if adj in adj_dict:
            adj_id = adj_dict[adj]
        else:
            adj_dict[adj] = len(adj_dict)
            adj_id = adj_dict[adj]
        
        if noun in noun_dict:
            noun_id = noun_dict[noun]
        else:
            noun_dict[noun] = len(noun_dict)
            noun_id = noun_dict[noun]
            
        image_names = os.listdir(os.path.join(args.data_root, "images", adj_noun_pair))
        for image_name in image_names:
            # Iterate through every image under this `Adjective Noun` folder
            image_path = os.path.join(args.data_root, "images", adj_noun_pair, image_name)
            adj_list.append(adj)
            adj_id_list.append(adj_id)
            noun_list.append(noun)
            noun_id_list.append(noun_id)
            image_path_list.append(image_path)

    data = pd.DataFrame(
        {
            "image_id": np.arange(0, len(adj_id_list)),
            "adjective": adj_list,
            "adjective_id": adj_id_list,
            "noun": noun_list,
            "noun_id": noun_id_list,
            "image_path": image_path_list,
        }
    )
    antonym_dict = load_antonym_data(args)
    return data, antonym_dict

def load_antonym_data(args):
    antonym_dict = {}
    antonym_path = os.path.join(args.data_root, "adj_ants.csv")
    with open(antonym_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            words = line.strip(",\n").split(",")
            antonym_dict[words[0]] = words[1:] if len(words) > 1 else []
    return antonym_dict

