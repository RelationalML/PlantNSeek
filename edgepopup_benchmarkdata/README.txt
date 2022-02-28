This is an implementation and extension of edge-popup ("What's hidden in a randomly weighted neural network?" by Vivek Ramanujan, Mitchell Wortsman, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari) with support of predefined neural network architectures with planted tickets.
This source code is part of "Plant ’n’ Seek: Can You Find the Winning Ticket?: by <Anonymous Authors>.

We make this code available under the Apache License 2.0, parts of the original edge-popup source code are used and referenced to the original source, which was also licensed under Apache License 2.0.

All three example data sets (circle, ReLU, helix) are supported in combination with original edge-popup with scores for bias terms, and modifed version with annealing of sparsity levels.

USAGE:

Please refer to the helper (-h) option for all available parameters.
This version *only* supports planted ticket datasets. The implementation was tested in a CPU-only environment.
The datasets are specified by flags "--dataset circle" or "--dataset relu" or "--dataset helix".

To specify a network architecture, "--depth d" and "--width w" allow to specify the depth and width (all layers have same width for sake of use) as integral values. The depth d should be more than 2, we tested up to depth 10.

The sparsity flag allows to specify a desired target sparsity, e.g. "--sparsity 0.1" prunes away 90% of the available parameters. By setting it to -1, the sparsity of the ground truth ticket is used.

EXAMPLE:

Running a network of depth 5 and layer widths 50 on circle data, aiming for parameter (including bias) sparsity of 10% using the extension of edge-popup with reducing the sparsity stepwise to the target sparsity in each epoch.

python3 ticket_main.py --lr 0.1 --epochs 10 --width 50 --depth 5 --sparsity 0.1 --dataset circle --anneal True

