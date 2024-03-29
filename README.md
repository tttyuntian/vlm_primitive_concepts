# Do Vision-Language Pretrained Models Learn Composable Primitive Concepts?
This repository provides the code for *Do Vision-Language Pretrained Models Learn Composable Primitive Concepts?* published at TMLR (2023).

## Table of Contents
1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Experiments](#experiments)
4. [How to Cite](#how-to-cite)


## Installation
- Install the necessary packages in `requirements.txt`.
- Install the CLIP API for computing features and activations. 
    - Note that CLIP requires at least pytorch=1.7.1.
    - For further instructions, see: https://github.com/openai/CLIP.
```
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

## Data Preparation

## MIT States 
- Download the images, labels, and annotations at http://web.mit.edu/phillipi/Public/states_and_transformations/index.html. Also, download the standard split from TMN.
    - After unzipping the dataset, move the images into the root directiory of MIT-States. 
    - After unzipping the standard split, move `compositional-split-natural`  to `data/mit_states`


```
# CREATE DATA DIRECTORIES
cd vlm_concept/
mkdir data/
mkdir data/mit_states/
cd data/mit_states/

# DOWNLOAD MIT-STATES DATASET
wget http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip
unzip -qq release_dataset.zip
mv release_dataset/* .
rename "s/ /_/g" /data/mit_states/images/*

# DOWNLOAD TMN SPLIT
wget https://www.cs.cmu.edu/~spurushw/publication/compositional/compositional_split_natural.tar.gz
tar -xzf compositional_split_natural.tar.gz
mv mit-states/* .
```


## CUB
- Download the images, labels, and annotations at http://www.vision.caltech.edu/visipedia/CUB-200-2011.html. 
    - Make sure to move `attributes.txt` into the extracted `CUB_200_2011/attributes/` folder. 
```
cd vlm_concept/
mkdir data/cub/

wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz

mv attributes.txt CUB_200_2011/attributes/attributes.txt
```



# Experiments
## MIT-States

1. Precompute CLIP/ViLT/ALBEF features and activations. 

```
python -m mit_states.clip_precompute_features --model_name "vit_b_32"

python -m mit_states.vilt_precompute_features

python -m mit_states.albef_precompute_features
```

2. Train CompMap with features from CLIP/ViLT/ALBEF.

```
python -m mit_states.train_retrieval_model.py --model_name "vit_b_32"

python -m mit_states.train_retrieval_model.py --model_name "vilt"

python -m mit_states.train_retrieval_model.py --model_name "albef"
```


## CUB

1. Precompute the features and activations. 
```
mkdir cub_200_2011/saved_activations/

python -m cub_200_2011.precompute_features --data_root ./data/ --output_path cub_200_2011/saved_activations/

python -m cub_200_2011.precompute_attribute_activations --data_root ./data/ --output_path cub_200_2011/saved_activations/
```

2. Run n-way k-shot shot ablation experiments. 
    - Specify the `n`, `k` for n-way k-shot classification with the arguments `-n` and `-k`. Specify the `--output_path` location of the saved activations from 2. with `--activations_root`.

```
python -m cub_200_2011.nway_kshot_cub_clip --data_root ./data/ --activations_root saved_activations/ -n 5 10 50 100 200 -k 1 5 
```


# How to Cite
```
@article{yun2023do,
    title={Do Vision-Language Pretrained Models Learn Composable Primitive Concepts?},
    author={Tian Yun and Usha Bhalla and Ellie Pavlick and Chen Sun},
    journal={Transactions on Machine Learning Research},
    year={2023},
}
```