#!/bin/bash

# create and activate env
conda init

# Automatically ships with cu121
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
echo "\n----Installing torch-cluster, this make take a while----"
pip3 install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
pip3 install torch_geometric==2.5.3
pip3 install argparse
pip3 install open3d==0.9.0
