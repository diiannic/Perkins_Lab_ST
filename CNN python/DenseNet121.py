'''
pip install pandas
pip install scanpy
pip install --editable .
pip install anndata2ri
pip install tensorflow_datasets
'''

import os

# Load spot location csv
import pandas as pd

# Load barcode feature matrix
import scanpy as sc
from scipy.sparse import issparse
import numpy as np
import string
import random

# Load in spot images
import PIL
from PIL import Image
import matplotlib.pyplot as plt

# NN
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
# pretrained model
from keras.models import Model

# saving model
from datetime import datetime


def get_spot_locations(tissue_pos_path):
    spot_locations = pd.read_csv(tissue_pos_path, header=None, names = ["barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres_yValue", "pxl_row_in_fullres_xValue"])
    spot_locations = spot_locations[spot_locations['in_tissue']==1]
    spot_locations.reset_index(drop=True, inplace=True)
    return spot_locations


def load_bfm(dir, count_file, spot_locations, number_of_genes_to_predict):
    adata = sc.read_visium(dir, genome=None, count_file=count_file, library_id=None, load_images=True)

    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, max_counts=35000)

    #normalize the data with scRNA normalization
    # useful site: https://anndata.readthedocs.io/en/stable/anndata.AnnData.html#anndata.AnnData
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes = number_of_genes_to_predict)

    mean_expression = adata.var["means"]
    mean_expression.sort_values(ascending = False)
    adata = adata[:, mean_expression.index[:number_of_genes_to_predict]]

    #equivalent to the GetAssayData
    #1 expression matrix
    barcode_feature_matrix = adata.to_df()

    #filter number of barcodes to get a list of barcodes
    barcodes = list(barcode_feature_matrix.index.values)
    sorted_barcodes = barcodes
    sorted_barcodes.sort()
    print(2)
    if barcodes == sorted_barcodes:
      spot_locations.index = spot_locations["barcode"]
      spot_locations.sort_index(inplace = True)
      for i in spot_locations.index:
        if i in barcodes:
          continue
        else:
          spot_locations.drop([i], axis = 0, inplace = True)#inplace = True, overwrite the current dataframe when set to true, not creating a new variable here (False)
    else:
      print("something's wrong. The barcodes from the barcode feature matrix are not in the correct order")

    return barcode_feature_matrix, spot_locations


def find_pove(bfm, predictions, number_of_samples, number_of_genes):
  totalPOVE = []
  for index in range(number_of_genes): #for every gene we predict
    actual_data_vairance = bfm[:number_of_samples, index].var() #TV
    residuals = bfm[:number_of_samples, index] - predictions[:number_of_samples, index]
    residuals_variance = residuals.var() #RV
    explained_variance = actual_data_vairance - residuals_variance #EV
    if actual_data_vairance != 0:
      POVE = 100*(explained_variance) / actual_data_vairance
      totalPOVE.append(POVE)
  return sum(totalPOVE)/len(totalPOVE)


def main():
    np.random.seed(1998)
    cropped_img_width = 224
    number_of_genes_to_predict = 100
    number_of_samples_to_use = 1000  # 2938
    spot_locations = get_spot_locations("/Users/colten/Desktop/Perkins_Lab_ST/Identifying (usable) image spots/data/spatial/tissue_positions_list.csv")
    bfm, spot_locations = load_bfm("/Users/colten/Desktop/Perkins_Lab_ST/Identifying (usable) image spots/data", 'V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5', spot_locations, number_of_genes_to_predict)



if __name__ == "__main__":
    main()