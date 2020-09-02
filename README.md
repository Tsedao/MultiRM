# MultiRM: Attention-based multi-label neural networks for integrated prediction and interpretation of twelve widely occuring RNA modifications

## Prerequisites
* `python`: 3.7.6
* `CUDA`: 10.1
* `pytorch`: 1.2.0
## Installation
Our current release has been tested on Ubuntu 16.04.4 LTS

**Cloning the repository and downloading MultiRM**
```
git clone git@github.com:Tsedao/MultiRM.git
cd MultiRM
```

## Demo
Here is a simple demo which using `AGCTGCCCTCCTGCTCGGAGCTTAGACCACAAAAAAGCTTGAGTTGGGATCCCCCC`
RNA sequence as input to predict modifications
Usage:
```
python main.py -s [RNA sequence] --top [No. of top-k highlighted sites] --alpha [significant level] --gpu [which gpu to use]
```
Example:
```
python main.py -s AGCTGCCCTCCTGCTCGGAGCTTAGACCACAAAAAAGCTTGAGTTGGGATCCCCCC --top 3 --alpha=0.1 --gpu=0
```
Predicting the RNA modification of a singe RNA sequence (Minimum length:51-bp), the result generates as:
```
************************ Reporting************************
m6A is predict at 26 with p-value 0.0600 and alpha 0.100
Cm is predict at 27 with p-value 0.0933 and alpha 0.100
m5C is predict at 27 with p-value 0.0533 and alpha 0.100
There is no modification sites at 28
There is no modification sites at 29
There is no modification sites at 30
There is no modification sites at 31
```
You also can visualize the specific site where the modification lies in and
how attention make that decision by turn `verbose` into `True`:
```
python main.py -s AGCTGCCCTCCTGCTCGGAGCTTAGACCACAAAAAAGCTTGAGTTGGGATCCCCCC --top 3 --alpha=0.1 --gpu=0 --verbose=True
```
it will generates:
```
***************Visualize modification sites***************
************************* 0-51 bp*************************
Origin AGCTGCCCTCCTGCTCGGAGCTTAGACCACAAAAAAGCTTGAGTTGGGATC
Am     ---------------------------------------------------
Cm     ---------------------------C-----------------------
Gm     ---------------------------------------------------
Um     ---------------------------------------------------
m1A    ---------------------------------------------------
m5C    ---------------------------C-----------------------
m5U    ---------------------------------------------------
m6A    --------------------------C------------------------
m6Am   ---------------------------------------------------
m7G    ---------------------------------------------------
Psi    ---------------------------------------------------
AtoI   ---------------------------------------------------
*************************51-56 bp*************************
Origin CCCCC
Am     -----
Cm     -----
Gm     -----
Um     -----
m1A    -----
m5C    -----
m5U    -----
m6A    -----
m6Am   -----
m7G    -----
Psi    -----
AtoI   -----

******************* Visualize Attention*******************
************************* 0-51 bp*************************
Origin AGCTGCCCTCCTGCTCGGAGCTTAGACCACAAAAAAGCTTGAGTTGGGATC
Am     ---------------------------------------------------
Cm     ---------CCT-CTC--------GAC------------------------
Gm     ---------------------------------------------------
Um     ---------------------------------------------------
m1A    ---------------------------------------------------
m5C    ----GCC--CCT------------GAC------------------------
m5U    ---------------------------------------------------
m6A    --CTG-------------------GAC------------------GGG---
m6Am   ---------------------------------------------------
m7G    ---------------------------------------------------
Psi    ---------------------------------------------------
AtoI   ---------------------------------------------------
*************************51-56 bp*************************
Origin CCCCC
Am     -----
Cm     -----
Gm     -----
Um     -----
m1A    -----
m5C    -----
m5U    -----
m6A    -----
m6Am   -----
m7G    -----
Psi    -----
AtoI   -----
```
Finally, you also can save these results by turning `save` to `True`.
