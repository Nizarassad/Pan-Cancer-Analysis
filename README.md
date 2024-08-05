## The Cancer Genome Atlas Pan-Cancer Analysis Project

Pan-cancer analysis involves assessing frequently mutated genes and other genomic abnormalities common to many different cancers, regardless of tumor origin. Using next-generation sequencing (NGS), pan-tumor projects such as The Cancer Genome Atlas2 have made significant contributions to our understanding of DNA and RNA variants across many cancer types.
<div align="center">
  <img  height="200" src="https://drive.google.com/uc?id=1wyCTVxiPnUHJXHJ24RR5Qr5kGfHEL-D0" alt="pancancer"  width="300" />
</div>

The objective of this project is to utilize regression techniques to predict a continuous value using data from The Cancer Genome Atlas (TCGA) Pan-Cancer analysis project. 

### Goal and Aims:
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

### Evaluation
The models are evaluated using two different metrics: mean squared error (MSE) and cross-validation. 

| Model                    | Score | Cross-validation | Mean Squared Error                  |
|--------------------------|-------|------------------|-------------------------------------|
| Simple Linear Regression | 0.98  | 0.98             | 0.05                                |
| Multiple Linear Regression | 0.98  | 0.98             | 0.04         |
| Ridge Regression         | 0.97  | 0.96             | 0.05         |
| Lasso Regression         | 0.92  | 0.94             | 0.07         |

The results indicate relatively high performance across the four regression models, with accuracy scores ranging from 0.92 to 0.98. However, it is noteworthy that Lasso regression has a lower accuracy score and higher mean squared error compared to the other models.

In summary, simple and multiple linear regression models exhibit comparable performance, while Ridge and Lasso regression models have slightly lower performance.
