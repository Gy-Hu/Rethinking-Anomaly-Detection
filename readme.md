# DSAA6000B Assignment

This is for the DSAA6000B assignment (Personal usage only). This repo update `KNN-based graph re-construction` method for GCN model, and `grid search hyperparameter tuning` for L and D. For software engineering side, I update `GPU support` and add more command line arguments for easy usage.

## Note
* `log/` contains all the log files for my experiments

## Usage
Put `tfinance` in `./dataset/` and run `python3 main.py`. Data can directly download the T-Finance dataset from [here](https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr?usp=sharing)
* usage: `python main.py [-h] [--dataset DATASET] [--train_ratio TRAIN_RATIO] [--hid_dim HID_DIM] [--num_layers NUM_LAYERS] [--epoch EPOCH] [--run RUN] [--knn-reconstruct-graph] [--knn-reconstruct-graph-approximate] [--alpha ALPHA] [--top-k TOP_K] [--save-model] [--model-path MODEL_PATH] [--device DEVICE] [--choose-model CHOOSE_MODEL] [--hyperparameter-tuning]`
* parameters:
    * `--dataset DATASET` : Dataset for this model (yelp/amazon/tfinance/tsocial, now only support tfinance)
    * `--train_ratio TRAIN_RATIO` : Training ratio, default is 0.4
    * `--hid_dim HID_DIM` : Hidden layer dimension in GCN model, default is 64
    * `--num_layers NUM_LAYERS` : Number of GCN layers, default is 3
    * `--epoch EPOCH` : The max number of epochs, default is 100
    * `--run RUN` : Running times. Default is 1, if run multiple times and want to archieve more stable result, set it to 3 or more.
    * `--knn-reconstruct-graph` : Reconstruct graph using KNN algorithm, use sklearn's NearestNeighbors
    * `--knn-reconstruct-graph-approximate` : Reconstruct graph using approximate KNN algorithm (Fast verision), use annoy
    * `--alpha ALPHA` : Propotion of threshold in KNN algorithm, default is 1.0
    * `--top-k TOP_K` : Top-k in KNN algorithm, default is 3
    * `--save-model` : Save model
    * `--model-path MODEL_PATH` : Path to save model, default is `./model`
    * `--device DEVICE` : Device to use, default is `cuda:0`
    * `--choose-model CHOOSE_MODEL` : Choose model to use (GCN/BWGNN/GAT/SAGE/Cheb), default is GCN
    * `--hyperparameter-tuning` : Hyperparameter tuning for L and D


### Run with example
* Run with L and D hyperparameter tuning
    * `python3 main.py --hyperparameter-tuning`
* Run with difference setting of KNN-graph re-construction
    * `python3 main.py --hid_dim 128 --knn-reconstruct-graph --alpha 0.5 --top-k 4 --run 3`