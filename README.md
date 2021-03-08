# MultiRM: Attention-based multi-label neural networks for integrated prediction and interpretation of twelve widely occuring RNA modifications

## Prerequisites
* `python`: 3.7.6
* `CUDA`: 10.1
* `pytorch`: 1.2.0
## Installation
For command line version, our current release has been tested on Ubuntu 16.04.4 LTS.

For GUI version, please check our [Web-Server]( www.xjtlu.edu.cn/biologicalsciences/multirm).

**Cloning the repository and downloading MultiRM**
```
git clone https://github.com/Tsedao/MultiRM.git
cd MultiRM
```

## Demo
Here is a simple demo which using `GGGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGCCGCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGGATACCTGCTGATTCCCTTCCCCTCTGAACCCCCAACACTCTGGCCCATCGGGGTGACGGATATCTGCTTTTTAAAAATTTTCTTTTTTTGGCCCATCGGGGCTTCGGATA`
RNA sequence as input to predict modifications.

Usage:
```
cd Scripts
python main.py -s [RNA sequence] --top [No. of top-k highlighted sites] --alpha [significant level] --gpu [which gpu to use]
```

The following options are available:

* `seqs`: Input single RNA sequnce in string format. (Minimum length: 51-bp)
* `att_window`: Length of sliding window to aggregate attention weights on a single sequence. (default:3; recommended range: 2-6)
* `top`: Number of top consecutive nucleotides based on the summation of attention weights. (default:3; recommended range: 2-5)
* `alpha`:Significance level. (default:0.05)
* `verbose`: Whether to show the detailed predictions or not. (default:False)
* `save`: Whether to save the results to file or not. (default:False)
* `save_path`: (Optional) Path of desirable directory to store results. (default: current working directory)
* `save_id`: (Optional) JOBID for the use of web sever.

Example:
```
python main.py -s GGGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGCCGCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGGATACCTGCTGATTCCCTTCCCCTCTGAACCCCCAACACTCTGGCCCATCGGGGTGACGGATATCTGCTTTTTAAAAATTTTCTTTTTTTGGCCCATCGGGGCTTCGGATA --top 3 --att_window=3 --alpha=0.1 --gpu=0
```
Predicting the RNA modification of a singe RNA sequence (Minimum length:51-bp), the result generates as:
```
Note: MultiRM does not make predictions for the first and
last 25nt of the input sequence.

************************Reporting************************
***************Visualize modification sites**************
************************  1- 50 nt***********************
Origin GGGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGC
Am**** --------------------------------------------------
Cm**** --------------------------------------------------
Gm**** ------------------------------------------------G-
Um**** --------------------------------------------------
m1A*** --------------------------------------------------
m5C*** ---------------------------C----------------------
m5U*** --------------------------------------------------
m6A*** --------------------------------------------------
m6Am** --------------------------------------------------
m7G*** --------------------------------------------------
Psi*** --------------------------------------------------
AtoI** --------------------------------------------------
************************ 51-100 nt***********************
Origin CGCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGG
Am**** --------------------------------------------------
Cm**** C-C---------C---------------CC--------------------
Gm**** -G-G-------G------------------------------------GG
Um**** ---------------TTT-TTTTT--------------------------
m1A*** -----A------------A-------------------------------
m5C*** ---------C--C---------------CC--------------------
m5U*** -----------------T--TTTT---T----------------------
m6A*** --------------------------------------------------
m6Am** --------------------------------------------------
m7G*** --------------------------------------------------
Psi*** --------------------------------------------------
AtoI** --------------------------------------------------
************************101-150 nt***********************
Origin ATACCTGCTGATTCCCTTCCCCTCTGAACCCCCAACACTCTGGCCCATCG
Am**** --------------------------A-----------------------
Cm**** ------------------CCCC-C------CCC-----------------
Gm**** ------G--G-------------------------------G-------G
Um**** ----------------------T------------------------T--
m1A*** --------------------------A-----------------------
m5C*** -------------CCC--CC---C----C------C--------------
m5U*** ----------------------T-T----------------------T--
m6A*** --------------------------------------------------
m6Am** --------------------------------------------------
m7G*** --------------------------------------------------
Psi*** ----------------------T---------------------------
AtoI** --------------------------------------------------
************************151-200 nt***********************
Origin GGGTGACGGATATCTGCTTTTTAAAAATTTTCTTTTTTTGGCCCATCGGG
Am**** -----A---A------------A-A-------------------------
Cm**** ----------------C---------------------------------
Gm**** GGG-G--GG------G----------------------------------
Um**** ---T--------------TTTT-------T--------------------
m1A*** -----A---A-A----------AAAAA-----------------------
m5C*** -------------C--C---------------------------------
m5U*** ---T--------------TTTT-----TTTT-------------------
m6A*** --------------------------------------------------
m6Am** --------------------------------------------------
m7G*** --------------------------------------------------
Psi*** ---T----------------------------------------------
AtoI** -----A---A----------------------------------------
************************201-210 nt***********************
Origin GCTTCGGATA
Am**** ----------
Cm**** ----------
Gm**** ----------
Um**** ----------
m1A*** ----------
m5C*** ----------
m5U*** ----------
m6A*** ----------
m6Am** ----------
m7G*** ----------
Psi*** ----------
AtoI** ----------


*******************Visualize Attention*******************
************************  1- 50 nt***********************
Origin GGGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGC
Am**** --------------------------------------------------
Cm**** ------------------------------------TCGC--ATCGG-GC
Gm**** --------------------------------TTATTCGC-CAT--GGGC
Um**** --------------------------------------------CGGGGC
m1A*** ----------------------------------ATT-------CGG---
m5C*** --------GGA---------------TCT-TTT----CGC---TCG----
m5U*** --------------------------------------------CGGGGC
m6A*** --------------------------------------------------
m6Am** --------------------------------------------------
m7G*** --------------------------------------------------
Psi*** --------------------------------------------------
AtoI** --------------------------------------------------
************************ 51-100 nt***********************
Origin CGCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGG
Am**** --------------------------------------------------
Cm**** CGC----ACCTGC-------------TTCCTTAG------------TCGG
Gm**** CGCG-----CTG---------------------GCCCATCG-----TCGG
Um**** CGCG----------TTTTATTTTTT-TTC----------CGG-----CGG
m1A*** ---GGA-----------TAT--TTT----------CCA------------
m5C*** -------ACCTGC-------------TTCCTTAGCCC-TCGG----TCGG
m5U*** CGCGGATACC-----TTT-TTTTTT-TTC------------------CGG
m6A*** --------------------------------------------------
m6Am** --------------------------------------------------
m7G*** --------------------------------------------------
Psi*** -----------------------------------------------CGG
AtoI** --------------------------------------------------
************************101-150 nt***********************
Origin ATACCTGCTGATTCCCTTCCCCTCTGAACCCCCAACACTCTGGCCCATCG
Am**** --ACC-GCT---------------TGA-----CAA-----TGG------G
Cm**** --------TGATT-CCTTCCCCTCTGAACCCCCAA---------CCA-CG
Gm**** ----CTGCTG-TTC-------CTC-GAACCCCCAACACTCTGGCCC-TCG
Um**** --------TGA----------CTC-GAA-CCCCA-CAC--------ATCG
m1A*** ---CCT------------------TGA-------------TGGCCC----
m5C*** -----TGCTGATTCCCTTCCCCTCT--ACC----ACA----GGC-CATCG
m5U*** --ACC---TGAT---------CTCTGAA-CCCCA-CAC--------ATCG
m6A*** --------------------------------------------------
m6Am** --------------------------------------------------
m7G*** --------------------------------------------------
Psi*** --------TGA----------CTC-------CCA-CAC------------
AtoI** ----------------------------------ACA---TGG-------
************************151-200 nt***********************
Origin GGGTGACGGATATCTGCTTTTTAAAAATTTTCTTTTTTTGGCCCATCGGG
Am**** GGGTGA-GGA-----------TAAAA-TTT--------TGG---------
Cm**** G-------------TGC---------------------------------
Gm**** GGGTGACGG----CTG-----------------TTT--------------
Um**** GGGTGACG-ATA-CTG-TTTTTAA----TTT-------------------
m1A*** ---TGACGGATAT------TTTAAAAATTTTC---TTTTGGCCCAT----
m5C*** ------------TCTGC-TTT-----------------------------
m5U*** GGGT-ACGGATATCTGCTTTTTAA--ATTTT-TTT---------------
m6A*** --------------------------------------------------
m6Am** --------------------------------------------------
m7G*** --------------------------------------------------
Psi*** -GGT----------------------------------------------
AtoI** ---TGA-GGA-------------AAA-TTT--------------------
************************201-210 nt***********************
Origin GCTTCGGATA
Am**** ----------
Cm**** ----------
Gm**** ----------
Um**** ----------
m1A*** ----------
m5C*** ----------
m5U*** ----------
m6A*** ----------
m6Am** ----------
m7G*** ----------
Psi*** ----------
AtoI** ----------
```
You also can check detailed descriptions by turning `verbose` into `True`:
```
python main.py -s GGGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGCCGCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGGATACCTGCTGATTCCCTTCCCCTCTGAACCCCCAACACTCTGGCCCATCGGGGTGACGGATATCTGCTTTTTAAAAATTTTCTTTTTTTGGCCCATCGGGGCTTCGGATA --top 3 --att_window=3 --alpha=0.1 --gpu=0 --verbose=True
```
Additionally ,you can save these results by turning `save` to `True`.

## Train your own MultiRM from scratch
```
python train.py --mode='train' --use_embedding=True --epochs=10 --lr=0.001 --batch_size=64 --length=51
```
For reference, please check our [training data profie](https://github.com/Tsedao/MultiRM/blob/master/Notebooks/RMdata_profile.ipynb)
