conda create -n hgnn python=3.9

conda activate hgnn

# https://pytorch.org/get-started/locally/
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install pytorch==2.4.0  pytorch-cuda=12.4 -c pytorch -c nvidia

# https://www.dgl.ai/pages/start.html
conda install -c dglteam/label/th24_cu124 dgl


pip install  tqdm  pyyaml  numpy  pandas  scikit-learn