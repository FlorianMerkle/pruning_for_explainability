#!/bin/bash
# script creates necessary folders, downloads imagenette and installs a new conda environment

mkdir pruned_models
mkdir trained_imgnette_models
mkdir data
echo "created folders"

wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz -p data
tar -xzf data/imagenette2.tgz -C data
echo "downloaded and unzipped imagenette"

conda update conda
conda config --append channels conda-forgeyn
conda create -n "pruning_for_xAI" --file requirements.txt
conda activate pruning_for_xAI
echo "ready to work with the repo"