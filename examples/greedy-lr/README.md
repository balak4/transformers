## Environment Installation Instructions

1. git clone https://github.com/balak4/transformers/tree/main ~/balak4/transformers
2. conda env create -f git/balak4/transformers/examples/greedy-lr/conda/pytorch_p310_greedy_v2.yml
3. Install modified transformers fork in local env:
    a. source ~/.bashrc
    b. conda activate pytorch_p310_greedy_v2
    c. python3 -m pip install -e ~/balak4/transformers

