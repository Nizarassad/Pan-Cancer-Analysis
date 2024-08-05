# -*- coding: utf-8 -*-

# Adresse des deux fichiers de donnees
# https://perso.univ-rennes1.fr/valerie.monbet/MachineLearning/TCGA-PANCAN-HiSeq-801x20531/data.csv
# https://perso.univ-rennes1.fr/valerie.monbet/MachineLearning/TCGA-PANCAN-HiSeq-801x20531/labels.csv

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


print("N'oubliez pas de mettre votre numero d'etudiant")
etudiant = 23102934 # nombre à  remplacer par votre numéro d'etudiant
np.random.seed(etudiant)

# On lit la premiere ligne pour obtenir le nombre de colonnes
X = np.loadtxt("data.csv",max_rows=1,delimiter=",",dtype=str)
nvars=len(X)-1

# Lecture des donnees correpondant aux nvars premieres variables
# et a la moitie des individus.
print("Vous pouvez choisir le nombre de variables nvars.")
nvars=200
print("Sont lues les donnees correpondant aux nvars=",nvars,
      "premieres variables et a moitie des individus",
      "(tiree aleatoirement sur la base de votre numero d'etudiant)")
X = np.loadtxt("data.csv",skiprows=1,delimiter=",",usecols=np.arange(nvars)+1)
nech=X.shape[0]//2
y =np.loadtxt("labels.csv",delimiter=",",skiprows=1,dtype=str)
per=np.random.permutation(X.shape[0])[:nech]
X,y = X[per,:], y[per,1]
print("Nombre de lignes, nombre de colonnes : ",X.shape)

# Elimination des variables constantes
l=np.std(X,axis=0)>1.e-8
X=X[:,l]
print("Nombre de lignes et colonnes, apres elimination des variables constantes: ",X.shape)



# Importation des données: Les données sont importées à partir d'un fichier CSV nommé "data.csv"
data = pd.read_csv("data.csv")

# Chargement des étiquettes
labels = pd.read_csv("labels.csv")

# Encodage des données de catégories
le = LabelEncoder()

data['sample_id'] = le.fit_transform(data['sample_id'])
print("Check the encoded column: \n",data.head())
print("")



labels['sample_id'] = le.fit_transform(labels['sample_id'])
labels['Class'] = le.fit_transform(labels['Class'])

print("Check the labels table:\n",labels.head())
print("")


# Fusion des données et des étiquettes
merged_data = pd.merge(data, labels, on='sample_id')



# Performer une analyse basique
print("Data Shape:", data.shape)
print("Labels Shape:", labels.shape)
print("Merged Data Shape:", merged_data.shape)
print(merged_data.head())
print("")



# Performer une analyse de statistiques
print("Merged Data Description:")
print(merged_data.describe())#

print("")




# Séparation des données en fonctionnalités et cible
x = merged_data.drop("Class", axis=1)
y = merged_data["Class"]

print("Features shape: ",x.shape)
print("Target shape: ",y.shape)
print("")






# Division des données pour l'entraînement et le test
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2 , stratify=y, random_state=1)

print("shape of the x axis of the train set:",x_train.shape)
print("shape of the y axis of the train set:",y_train.shape)
print("shape of the x axis of the test set:",x_test.shape)
print("shape of the y axis of the test set:",y_test.shape)
print("")

# Régression linéaire simple
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
y_predict_SLR = lin_reg.predict(x_test)


# Multiple Linear Regression
multi_reg = LinearRegression()
multi_reg.fit(x_train, y_train)
y_predict_MLR = lin_reg.predict(x_test)

# Ridge Regression
ridge_reg = Ridge(alpha=0.5)
ridge_reg.fit(x_train, y_train)
y_predict_RR = lin_reg.predict(x_test)

# Lasso Regression
lasso_reg = Lasso(alpha=0.5)
lasso_reg.fit(x_train, y_train)
y_predict_LR = lin_reg.predict(x_test)

#mean squared error
lin_reg_mse = mean_squared_error(y_test, y_predict_SLR)
multi_reg_mse = mean_squared_error(y_test, y_predict_MLR)
ridge_reg_mse = mean_squared_error(y_test, y_predict_RR)
lasso_reg_mse = mean_squared_error(y_test, y_predict_LR)

# Cross-validation
lin_reg_scores = cross_val_score(lin_reg, x_train, y_train, cv=5)
multi_reg_scores = cross_val_score(multi_reg, x_train, y_train, cv=5)
ridge_reg_scores = cross_val_score(ridge_reg, x_train, y_train, cv=5)
lasso_reg_scores = cross_val_score(lasso_reg, x_train, y_train, cv=5)

# Print the scores
print("Simple Linear Regression:")
print("Score: ", lin_reg.score(x_test, y_test))
print("Cross-Validation Score: ", lin_reg_scores.mean())
print("Mean Squared Error:", lin_reg_mse)
print("")

print("Multiple Linear Regression:")
print("Score: ", multi_reg.score(x_test, y_test))
print("Cross-Validation Score: ", multi_reg_scores.mean())
print("Mean Squared Error:", multi_reg_scores)
print("")

print("Ridge Regression:")
print("Score: ", ridge_reg.score(x_test, y_test))
print("Cross-Validation Score: ", ridge_reg_scores.mean())
print("Mean Squared Error:", ridge_reg_scores)
print("")

print("Lasso Regression:")
print("Score: ", lasso_reg.score(x_test, y_test))
print("Cross-Validation Score: ", lasso_reg_scores.mean())
print("Mean Squared Error:",lasso_reg_scores)



