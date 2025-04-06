
# Get Started

For *object deteciton*, please go to the `detection/` directory. (or click [Here](detection/README.md))

The below guide is for *semantic segmentation*.

## Environment

Install dependencies (we test on python=3.9.19, pytorch==1.13.1, cuda==11.6)
```
# mamba install
cd libs/
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.1.3 
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
cd ..
git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout v1.1.4 
MAMBA_FORCE_BUILD=TRUE pip install .

cd iSphereFormer
pip install -r requirement.txt 
or
conda env create -f environment.yml
```

Install `sptr`
```
cd third_party/SparseTransformer && python setup.py install
```

Note: Make sure you have installed `gcc` and `cuda`, and `nvcc` can work (if you install cuda by conda, it won't provide nvcc and you should install cuda manually.)

## Datasets Preparation

### SemanticKITTI
Download the SemanticKIITI dataset from [here](http://www.semantic-kitti.org/dataset.html#download). Unzip and arrange it as follows. Then fill in the `data_root` entry in the .yaml configuration file.
```
dataset/
|--- sequences/
|------- 00/
|------- 01/
|------- 02/
|------- 03/
|------- .../
```
### Custom
Download the SemanticKIITI dataset from [here](https://github.com/ywh939/SemanticDatasetUtil). Unzip and arrange it as follows. Then fill in the `data_root` entry in the .yaml configuration file.
```
dataset/
|--- sub1/
|------- kitti_format/
|---------- 000.bin
|---------- 000.label
|--- sub2/
|------- kitti_format/
|--- .../
```

## Training

### SemanticKITTI
Run in the background
```
./run_ddp_backstage.sh
```
You can run `tail -f ddp_nohup.log` on Linux terminal to watch log of model's outputs.  

Run in the terminal
```
CUDA_VISIBLE_DEVICES=0 python train.py --config ******.yaml
```
Or use vscode.

## Validation
For validation, you need to modify the `.yaml` config file. (1) fill in the `weight` with the path of model weight (`.pth` file); (2) set `val` to `True`; (3) for testing-time augmentation, set `use_tta` to `True` and set `vote_num` accordingly. After that, run the following command. 
```
python train.py --config [YOUR_CONFIG_PATH]
```
