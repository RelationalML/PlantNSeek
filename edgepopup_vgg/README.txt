This is a designated implementation of edgepopup with support for loading a ticket hidden in VGG16.
We provide the ticket as modelBest.pt, corresponding to the ticket that we use in the paper experiment originally found as a weak ticket by the Synflow algorithm.

Executing runExp.sh will run the whole experiment from the paper, automatically loading and hiding the ticket in VGG16.

Relevant functions to hide your own ticket can be found in planting.py and initializers.py.
