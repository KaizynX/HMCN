# Hierarchical Multi-Label Classification Networks

This repo is a implementation of HMCN-F refer to the paper "[Hierarchical Multi-Label Classification Networks](https://proceedings.mlr.press/v80/wehrmann18a.html)"

## Setup

```bash
pip install -r requirements.txt
```

## Run
```bash
usage: main.py [-h] --dataset {cellcycle_FUN,derisi_FUN,eisen_FUN,gasch2_FUN} [--seed SEED] [--device DEVICE]

Train neural network wutg train and validation set, and test it on the test set

options:
  -h, --help            show this help message and exit
  --dataset {cellcycle_FUN,derisi_FUN,eisen_FUN,gasch2_FUN}
                        dataset name, must end with: "_GO", "_FUN", or "_others"
  --seed SEED           random seed (default: 0)
  --device DEVICE       GPU (default:0)
```

## Reference

* https://github.com/EGiunchiglia/C-HMCNN
* https://github.com/mixiazhiyang/HMCN-F
* https://github.com/RAvontuur/hierarchical-multilabel-classification