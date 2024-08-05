## The Cancer Genome Atlas Pan-Cancer Analysis Project

Pan-cancer analysis involves assessing frequently mutated genes and other genomic abnormalities common to many different cancers, regardless of tumor origin. Using next-generation sequencing (NGS), pan-tumor projects such as The Cancer Genome Atlas2 have made significant contributions to our understanding of DNA and RNA variants across many cancer types.
<div align="center">
  <img  height="200" src="https://drive.google.com/uc?id=1wyCTVxiPnUHJXHJ24RR5Qr5kGfHEL-D0" alt="pancancer"  width="200" />
</div>
*Image 

The objective of this project is to utilize regression techniques to predict a continuous value using data from The Cancer Genome Atlas (TCGA) Pan-Cancer analysis project. 

### Aim anf goals 
The goal of this project is to train regression models using this data to predict a continuous value representing a specific molecular characteristic of cancers. The aims are the following:
- Study the genomes of various cancers to better understand their molecular characteristics.
- Assist in developing new treatment strategies


### Dataset
The data, collected from different types of tumors, can be downloaded from the two links below:
- Data CSV: https://perso.univ-rennes1.fr/valerie.monbet/MachineLearning/TCGA-PANCAN-HiSeq-801x20531/data.csv
- Labels CSV: https://perso.univ-rennes1.fr/valerie.monbet/MachineLearning/TCGA-PANCAN-HiSeq-801x20531/labels.csv

### Models
This project implements four different regression models using cancer genome data:
- Simple Linear Regression (SLR)
- Multiple Linear Regression (MLR)
- Ridge Regression (RR)
- Lasso Regression (LR)
