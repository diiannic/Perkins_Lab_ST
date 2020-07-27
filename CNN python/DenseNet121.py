'''
pip install pandas
pip install scanpy
pip install --editable .
pip install anndata2ri
pip install tensorflow_datasets
'''

# Getting file paths
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


def get_file_path_by_substring(substring, base_path):
    file = list(filter(lambda x: substring in x, os.listdir(base_path)))
    if len(file) != 1:
        print("there are too many bfm files when looking in " + base_path)
        quit()
    file = file[0]

    return file


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

    if barcodes == sorted_barcodes:
        spot_locations.index = spot_locations["barcode"]
        spot_locations.sort_index(inplace=True)
        for i in spot_locations.index:
            if i in barcodes:
                continue
            else:
                spot_locations.drop([i], axis=0, inplace=True)
    else:
      print("something's wrong. The bar codes from the barcode feature matrix are not in the correct order")

    return barcode_feature_matrix, spot_locations


    # defining a function to get a cropped spot image given the x and y coordinates of the center of the spot
def Crop_Spot(img_array, x, y, cropped_img_width):
    cropped_img = img_array[int(y - (cropped_img_width / 2)):int(y + (cropped_img_width / 2)),
                  int(x - (cropped_img_width / 2)):int(x + (cropped_img_width / 2))]

    return cropped_img


def generate_train_test_indexes(number_of_samples_to_use, number_of_spots):
    # partition data indexes into train and test
    train_test_indexes = [i for i in range(number_of_spots)]
    random.shuffle(train_test_indexes)
    # train_test_indexes = train_test_indexes[:number_of_samples_to_use]
    train_indexes = train_test_indexes[:int(0.7 * len(train_test_indexes))]
    train_indexes.sort()
    test_indexes = train_test_indexes[int(0.7 * len(train_test_indexes)):]
    test_indexes.sort()
    for i in train_indexes:
        if i in test_indexes:
            print("something went wrong. training and testing indexes overlap")
    return train_indexes, test_indexes


def get_cropped_images(img_path, spot_locations, cropped_img_width, train_indexes, test_indexes):
    wsi_file_name = img_path
    wsi = Image.open(wsi_file_name)
    img_array = np.array(wsi)
    print("shape of wsi: ", img_array.shape)

    # crop image into train and test
    training_cropped_img_list = []
    for i in train_indexes:
        cropped_img = Crop_Spot(img_array, spot_locations["pxl_row_in_fullres_xValue"][i],
                                spot_locations["pxl_col_in_fullres_yValue"][i],
                                cropped_img_width)
        training_cropped_img_list.append(cropped_img)
    training_cropped_img_list = np.array(training_cropped_img_list)
    # training_cropped_img_list = (training_cropped_img_list - [training_cropped_img_list[:, :, :, 0].mean(), training_cropped_img_list[:, :, :, 1].mean(), training_cropped_img_list[:, :, :, 2].mean()])

    testing_cropped_img_list = []
    for i in test_indexes:
        cropped_img = Crop_Spot(img_array, spot_locations["pxl_row_in_fullres_xValue"][i],
                                spot_locations["pxl_col_in_fullres_yValue"][i],
                                cropped_img_width)
        testing_cropped_img_list.append(cropped_img)
    testing_cropped_img_list = np.array(testing_cropped_img_list)
    # testing_cropped_img_list = (testing_cropped_img_list - [training_cropped_img_list[:, :, :, 0].mean(), training_cropped_img_list[:, :, :, 1].mean(), training_cropped_img_list[:, :, :, 2].mean()])
    return training_cropped_img_list, testing_cropped_img_list


def partition_bfm(bfm, train_indexes, test_indexes):
    bfm = bfm.to_numpy()
    training_bfm = bfm[train_indexes]
    testing_bfm = bfm[test_indexes]
    return training_bfm, testing_bfm


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
    PIL.Image.MAX_IMAGE_PIXELS = 1000000000
    np.random.seed(1998)

    cropped_img_width = 224
    number_of_genes_to_predict = 100
    number_of_samples_to_use = 1000  # 2938
    train_indexes, test_indexes = generate_train_test_indexes(number_of_samples_to_use, number_of_samples_to_use)

    base_path = "/Users/colten/Desktop/Perkins_Lab_ST/Identifying (usable) image spots/data"
    #USE CURRENT WORKING DIRECTORY AS THE FIRST PART OF BASE_PATH
    cwd = str(os.getcwd())
    file = list(filter(lambda x: "ST_" in x, os.listdir(cwd)))
    if len(file) >= 1:
        print("Importing data from " + str(len(file)) + " datasets")
    quit()
    file = file[0]

    # setting file paths
    spot_location_path = base_path + "/spatial/" + get_file_path_by_substring("tissue_positions_list.csv", base_path + "/spatial")
    bfm_dir = base_path
    bfm_filename = get_file_path_by_substring("filtered_feature_bc_matrix.h5", base_path)
    image_path = base_path + "/" + get_file_path_by_substring("image.tif", base_path)

    # loading data
    spot_locations = get_spot_locations(spot_location_path)
    bfm, spot_locations = load_bfm(bfm_dir, bfm_filename, spot_locations, number_of_genes_to_predict)
    training_cropped_img_list, testing_cropped_img_list = get_cropped_images(image_path,
                                                                             spot_locations,
                                                                             cropped_img_width,
                                                                             train_indexes,
                                                                             test_indexes)
    training_bfm, testing_bfm = partition_bfm(bfm, train_indexes, test_indexes)

    # Neural Network
    print("made it to end")


if __name__ == "__main__":
    main()
