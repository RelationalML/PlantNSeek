This is a strongly modified implementation of https://github.com/ganguli-lab/Synaptic-Flow to support predefined neural network architectures with planted tickets.
This source code is part of "Plant ’n’ Seek: Can You Find the Winning Ticket?: by <Anonymous Authors>.

The original authors of the Synaptic Flow source code are Hidenori Tanaka, Daniel Kunin, Daniel L. K. Yamins, and Surya Ganguli.

All three example data sets (circle, ReLU, helix) are supported in combination with all implemented pruning methods (Synaptic-Flow, GraSP, SNIP, Magnitude, and Random pruning)

USAGE:

Please refer to the helper (-h) option for all available parameters.
This version *only* supports the planted ticket datasets. The implementation was tested in a CPU-only environment with "--no-cuda" flag.
The datasets are specified by using "--model-class lottery" along with the desired dataset "--dataset circle" or "--dataset relu" or "--dataset helix".

To specify a network architecture, "--depth d" and "--width w" allow to specify the depth and width (all layers have same width for sake of use) as integral values. The depth d should be more than 2, we tested up to depth 10.

Note also the difference to the original implementation, in our case the explicit sparsity parameter directly reflects the fraction of parameters that should be available after pruning, e.g. "--sparsity 0.1" masks off 90% of the parameters. By setting it to -1, the sparsity of the ground truth ticket is used.


EXAMPLE:

Running a network of depth 5 and layer widths 50 on circle data, aiming for parameter (including bias) sparsity of 10% using the synflow algorithm in singleshot mode. The default optimizer adam is used with a learning rate of 0.001, results are saves in "results/test/". Training after pruning is carried out for 10 rounds.

python3 main.py --dataset circle --model planted-model --model-class lottery --train-batch-size 32 --test-batch-size 512 --pruner synflow --prune-bias True --no-cuda --verbose --post-epochs 10 --sparsity 0.1 --lr 0.001 --result-dir results/test --depth 5 --width 50 --exp-suffix circle_test

The same setup, but with 5 rounds of multishot pruning with 10 epochs of training after each round, and 20 epochs of trainning after the final pruning round.

python3 main.py --dataset circle --model planted-model --model-class lottery --train-batch-size 32 --test-batch-size 512 --pruner synflow --prune-bias True --no-cuda --verbose --post-epochs 20 --sparsity 0.1 --lr 0.001 --result-dir results/test --depth 5 --width 50 --level 5 --pre-epochs 10 --exp-suffix circle_test_multishot