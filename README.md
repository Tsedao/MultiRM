# MultiRM: Attention-based multi-label neural networks for integrated prediction and interpretation of twelve widely occuring RNA modifications

## Prerequisites
* `python`: 3.7.6
* `CUDA`: 10.1
* `pytorch`: 1.2.0
## Installation
Our current release has been tested on Ubuntu 16.04.4 LTS

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
python main.py -s [RNA sequence] --top [No. of top-k highlighted sites] --alpha [significant level] --gpu [which gpu to use]
```

The following options are available:

* `seqs`: Input single RNA sequnce in string format. (Minimum length: 51-bp)
* `att_window`: Length of sliding window to aggregate attention weights on a single sequence. (default:3; recommended range: 2-6)
* `top`: Number of top consecutive nucleotides based on the summation of attention weights. (default:3; recommended range: 2-5)
* `alpha`:Significant level. (default:0.05)
* `verbose`: Whether to show the visualization or not. (default:False)
* `save`: Whether to save the results to file or not. (default:False)
* `save_path`: (Optional) Path of desirable directory to store results. (default: current working directory)
* `save_id`: (Optional) JOBID for the use of web sever. 

Example:
```
python main.py -s GGGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGCCGCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGGATACCTGCTGATTCCCTTCCCCTCTGAACCCCCAACACTCTGGCCCATCGGGGTGACGGATATCTGCTTTTTAAAAATTTTCTTTTTTTGGCCCATCGGGGCTTCGGATA --top 3 --att_window=3 --alpha=0.1 --gpu=0
```
Predicting the RNA modification of a singe RNA sequence (Minimum length:51-bp), the result generates as:
```
************************ Reporting************************
...
There is no modification sites at 48 
Gm is predicted at 49 with p-value 0.0867 and alpha 0.100
There is no modification sites at 50 
Cm is predicted at 51 with p-value 0.0733 and alpha 0.100
Gm is predicted at 52 with p-value 0.0267 and alpha 0.100
Cm is predicted at 53 with p-value 0.0867 and alpha 0.100
Gm is predicted at 54 with p-value 0.0733 and alpha 0.100
There is no modification sites at 55 
m1A is predicted at 56 with p-value 0.0667 and alpha 0.100
There is no modification sites at 57 
There is no modification sites at 58 
There is no modification sites at 59 
m5C is predicted at 60 with p-value 0.0733 and alpha 0.100
...
```
You also can visualize the specific site where the modification lies in and
how attention make that decision by turn `verbose` into `True`:
```
python main.py -s GGGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGCCGCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGGATACCTGCTGATTCCCTTCCCCTCTGAACCCCCAACACTCTGGCCCATCGGGGTGACGGATATCTGCTTTTTAAAAATTTTCTTTTTTTGGCCCATCGGGGCTTCGGATA --top 3 --att_window=3 --alpha=0.1 --gpu=0 --verbose=True
```
it will generates:
```
***************Visualize modification sites***************
************************  1- 51 bp************************
Origin GGGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGCC
Am     ---------------------------------------------------
Cm     --------------------------------------------------C
Gm     ------------------------------------------------G--
Um     ---------------------------------------------------
m1A    ---------------------------------------------------
m5C    ---------------------------C-----------------------
m5U    ---------------------------------------------------
m6A    ---------------------------------------------------
m6Am   ---------------------------------------------------
m7G    ---------------------------------------------------
Psi    ---------------------------------------------------
AtoI   ---------------------------------------------------
************************ 52-102 bp************************
Origin GCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGGAT
Am     ---------------------------------------------------
Cm     -C---------C---------------CC----------------------
Gm     G-G-------G------------------------------------GG--
Um     --------------TTT-TTTTT----------------------------
m1A    ----A------------A---------------------------------
m5C    --------C--C---------------CC----------------------
m5U    ----------------T--TTTT---T------------------------
m6A    ---------------------------------------------------
m6Am   ---------------------------------------------------
m7G    ---------------------------------------------------
Psi    ---------------------------------------------------
AtoI   ---------------------------------------------------
************************103-153 bp************************
Origin ACCTGCTGATTCCCTTCCCCTCTGAACCCCCAACACTCTGGCCCATCGGGG
Am     ------------------------A--------------------------
Cm     ----------------CCCC-C------CCC--------------------
Gm     ----G--G-------------------------------G-------GGGG
Um     --------------------T------------------------T-----
m1A    ------------------------A--------------------------
m5C    -----------CCC--CC---C----C------C-----------------
m5U    --------------------T-T----------------------T-----
m6A    ---------------------------------------------------
m6Am   ---------------------------------------------------
m7G    ---------------------------------------------------
Psi    --------------------T------------------------------
AtoI   ---------------------------------------------------
************************154-204 bp************************
Origin TGACGGATATCTGCTTTTTAAAAATTTTCTTTTTTTGGCCCATCGGGGCTT
Am     --A---A------------A-A-----------------------------
Cm     -------------C-------------------------------------
Gm     -G--GG------G--------------------------------------
Um     T--------------TTTT-------T------------------------
m1A    --A---A-A----------AAAAA---------------------------
m5C    ----------C--C-------------------------------------
m5U    T--------------TTTT-----TTTT-----------------------
m6A    ---------------------------------------------------
m6Am   ---------------------------------------------------
m7G    ---------------------------------------------------
Psi    T--------------------------------------------------
AtoI   --A---A--------------------------------------------
************************205-210 bp************************
Origin CGGATA
Am     ------
Cm     ------
Gm     ------
Um     ------
m1A    ------
m5C    ------
m5U    ------
m6A    ------
m6Am   ------
m7G    ------
Psi    ------
AtoI   ------

******************* Visualize Attention*******************
************************  1- 51 bp************************
Origin GGGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGCC
Am     ---------------------------------------------------
Cm     ------------------------------------TCGC--ATCGG-GCC
Gm     --------------------------------TTATTCGC-CAT--GGGCC
Um     --------------------------------------------CGGGGCC
m1A    ----------------------------------ATT-------CGG----
m5C    --------GGA---------------TCT-TTT----CGC---TCG-----
m5U    --------------------------------------------CGGGGCC
m6A    ---------------------------------------------------
m6Am   ---------------------------------------------------
m7G    ---------------------------------------------------
Psi    ---------------------------------------------------
AtoI   ---------------------------------------------------
************************ 52-102 bp************************
Origin GCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGGAT
Am     ---------------------------------------------------
Cm     GC----ACCTGC-------------TTCCTTAG------------TCGG--
Gm     GCG-----CTG---------------------GCCCATCG-----TCGG--
Um     GCG----------TTTTATTTTTT-TTC----------CGG-----CGG--
m1A    --GGA-----------TAT--TTT----------CCA--------------
m5C    ------ACCTGC-------------TTCCTTAGCCC-TCGG----TCGG--
m5U    GCGGATACC-----TTT-TTTTTT-TTC------------------CGG--
m6A    ---------------------------------------------------
m6Am   ---------------------------------------------------
m7G    ---------------------------------------------------
Psi    ----------------------------------------------CGG--
AtoI   ---------------------------------------------------
************************103-153 bp************************
Origin ACCTGCTGATTCCCTTCCCCTCTGAACCCCCAACACTCTGGCCCATCGGGG
Am     ACC-GCT---------------TGA-----CAA-----TGG------GGGG
Cm     ------TGATT-CCTTCCCCTCTGAACCCCCAA---------CCA-CGG--
Gm     --CTGCTG-TTC-------CTC-GAACCCCCAACACTCTGGCCC-TCGGGG
Um     ------TGA----------CTC-GAA-CCCCA-CAC--------ATCGGGG
m1A    -CCT------------------TGA-------------TGGCCC-------
m5C    ---TGCTGATTCCCTTCCCCTCT--ACC----ACA----GGC-CATCG---
m5U    ACC---TGAT---------CTCTGAA-CCCCA-CAC--------ATCGGGG
m6A    ---------------------------------------------------
m6Am   ---------------------------------------------------
m7G    ---------------------------------------------------
Psi    ------TGA----------CTC-------CCA-CAC-------------GG
AtoI   --------------------------------ACA---TGG----------
************************154-204 bp************************
Origin TGACGGATATCTGCTTTTTAAAAATTTTCTTTTTTTGGCCCATCGGGGCTT
Am     TGA-GGA-----------TAAAA-TTT--------TGG-------------
Cm     -----------TGC-------------------------------------
Gm     TGACGG----CTG-----------------TTT------------------
Um     TGACG-ATA-CTG-TTTTTAA----TTT-----------------------
m1A    TGACGGATAT------TTTAAAAATTTTC---TTTTGGCCCAT--------
m5C    ---------TCTGC-TTT---------------------------------
m5U    T-ACGGATATCTGCTTTTTAA--ATTTT-TTT-------------------
m6A    ---------------------------------------------------
m6Am   ---------------------------------------------------
m7G    ---------------------------------------------------
Psi    T--------------------------------------------------
AtoI   TGA-GGA-------------AAA-TTT------------------------
************************205-210 bp************************
Origin CGGATA
Am     ------
Cm     ------
Gm     ------
Um     ------
m1A    ------
m5C    ------
m5U    ------
m6A    ------
m6Am   ------
m7G    ------
Psi    ------
AtoI   ------
```
Finally, you also can save these results by turning `save` to `True`.
