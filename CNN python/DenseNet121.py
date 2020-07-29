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
# from scipy.sparse import issparse
import numpy as np
# import string
import random

# Load in spot images
import PIL
from PIL import Image
# import matplotlib.pyplot as plt

# NN
import tensorflow as tf
# import tensorflow.compat.v2 as tf
# import tensorflow_datasets as tfds
# pretrained model
# from keras.models import Model

# saving model
# from datetime import datetime


def get_file_path_by_substring(substring, base_path):
    file = list(filter(lambda x: substring in x, os.listdir(base_path)))
    if len(file) != 1:
        print("there are too many bfm files when looking in " + base_path)
        quit()
    file = file[0]

    return file


def get_high_expression_genes(cwd, filenames):
    for i in filenames:
        bfm_dir = cwd + "/" + i
        bfm_filename = get_file_path_by_substring("filtered_feature_bc_matrix.h5", bfm_dir)

        adata = sc.read_visium(bfm_dir, genome=None, count_file=bfm_filename, library_id=None, load_images=True)

        adata.var_names_make_unique()
        sc.pp.filter_cells(adata, min_counts=5000)
        sc.pp.filter_cells(adata, max_counts=35000)

        # get expression as data frame
        barcode_feature_matrix = adata.to_df()
        barcode_feature_matrix = barcode_feature_matrix.sum(axis=0)

        if i == filenames[0]:
            total_bfm = barcode_feature_matrix
        else:
            total_bfm = pd.concat([total_bfm, barcode_feature_matrix], axis=1, sort=False)

    total_bfm = total_bfm.sum(axis=1)
    total_bfm.sort_values(inplace=True, ascending=False)
    return list(total_bfm.index)


def get_spot_locations(tissue_pos_path):
    spot_locations = pd.read_csv(tissue_pos_path, header=None, names=["barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres_yValue", "pxl_row_in_fullres_xValue"])
    spot_locations = spot_locations[spot_locations['in_tissue']==1]
    spot_locations.reset_index(drop=True, inplace=True)
    return spot_locations


def load_bfm(directory, count_file, spot_locations, number_of_genes_to_predict, high_expression_genes):
    adata = sc.read_visium(directory, genome=None, count_file=count_file, library_id=None, load_images=True)

    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, max_counts=35000)

    # normalize the data with scRNA normalization
    # useful site: https://anndata.readthedocs.io/en/stable/anndata.AnnData.html#anndata.AnnData
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=number_of_genes_to_predict)

    # mean_expression = adata.var["means"]
    # mean_expression.sort_values(ascending=False)
    # adata = adata[:, mean_expression.index[:number_of_genes_to_predict]]

    # get expression matrix
    barcode_feature_matrix = adata.to_df()
    barcode_feature_matrix = barcode_feature_matrix[high_expression_genes]

    #filter number of barcodes to get a list of barcodes
    barcodes = list(barcode_feature_matrix.index.values)
    sorted_barcodes = list(barcodes)
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
        quit()
    return barcode_feature_matrix, spot_locations


# defining a function to get a cropped spot image given the x and y coordinates of the center of the spot
def crop_spot(img_array, x, y, cropped_img_width):
    cropped_img = img_array[int(y - (cropped_img_width / 2)):int(y + (cropped_img_width / 2)),
                            int(x - (cropped_img_width / 2)):int(x + (cropped_img_width / 2))]

    return cropped_img


def generate_train_test_indexes(number_of_samples_to_use):
    # partition data indexes into train and test
    train_test_indexes = [i for i in range(number_of_samples_to_use)]
    random.shuffle(train_test_indexes)
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

    # crop image into train and test
    training_cropped_img_list = []
    for i in train_indexes:
        cropped_img = crop_spot(img_array, spot_locations["pxl_row_in_fullres_xValue"][i],
                                spot_locations["pxl_col_in_fullres_yValue"][i],
                                cropped_img_width)
        training_cropped_img_list.append(cropped_img) # todo - implement data manipulation of rotating and flipping image
    training_cropped_img_list = np.array(training_cropped_img_list)

    testing_cropped_img_list = []
    for i in test_indexes:
        cropped_img = crop_spot(img_array, spot_locations["pxl_row_in_fullres_xValue"][i],
                                spot_locations["pxl_col_in_fullres_yValue"][i],
                                cropped_img_width)
        testing_cropped_img_list.append(cropped_img)
    testing_cropped_img_list = np.array(testing_cropped_img_list)
    return training_cropped_img_list, testing_cropped_img_list


def partition_bfm(bfm, train_indexes, test_indexes):
    bfm = bfm.to_numpy()
    training_bfm = bfm[train_indexes]
    testing_bfm = bfm[test_indexes]
    return training_bfm, testing_bfm


def find_pove(bfm, predictions, number_of_samples, number_of_genes):
    total_pove = []
    for index in range(number_of_genes):
        actual_data_vairance = bfm[:number_of_samples, index].var()
        residuals = bfm[:number_of_samples, index] - predictions[:number_of_samples, index]
        residuals_variance = residuals.var()
        explained_variance = actual_data_vairance - residuals_variance
        if actual_data_vairance != 0:
            pove = 100*explained_variance / actual_data_vairance
            total_pove.append(pove)
    return sum(total_pove)/len(total_pove)


def train_network(training_bfm, training_cropped_img_list, testing_bfm, testing_cropped_img_list, number_of_genes_to_predict):
    # import DenseNet121 model
    model = tf.keras.applications.DenseNet121(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )

    # freezing layers
    # for layer in model.layers:
    #  layer.trainable = False

    # remove the last layer from the model
    model = tf.keras.Model(model.input, model.layers[-2].output)

    # create final model
    model3 = tf.keras.Sequential()
    model3.add(model)
    model3.add(tf.keras.layers.Dense(1024,
                                     activation='sigmoid'))
    model3.add(tf.keras.layers.Dense(number_of_genes_to_predict,
                                     activation='relu',
                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=3)))

    model3.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(1e-5),
        metrics=['mae'],
    )
    print(model.summary())
    print(model3.summary())

    # training
    pove_progression = []
    for i in range(10):
        model3.fit(
            training_cropped_img_list,
            training_bfm,
            batch_size=32,
            epochs=1,
            validation_split=0.2
        )
        predictions = model3.predict(testing_cropped_img_list)
        pove = find_pove(testing_bfm, predictions, 10, number_of_genes_to_predict)
        print("Average POVE for the " + str(i + 1) + " round of training: ", pove)
        pove_progression.append(pove)
    print("POVE progression")
    print(pove_progression)


def main():
    PIL.Image.MAX_IMAGE_PIXELS = 1000000000
    np.random.seed(1998)

    cropped_img_width = 224
    number_of_genes_to_predict = 10
    number_of_samples_to_use_per_dataset = 1000  # 2938
    use_max_samples = True  # This will overwrite the number_of_samples_to_use_per_dataset if True

    # Setting cwd and base_path to each data set folder in the same directory as this file
    cwd = str(os.getcwd())
    files = list(filter(lambda x: "ST_" in x, os.listdir(cwd)))
    print("Importing data from " + str(len(files)) + " data sets")
    high_expression_genes = get_high_expression_genes(cwd, files)
    high_expression_genes = high_expression_genes[:number_of_genes_to_predict]

    # initializing data bins
    training_cropped_img_list = "empty"
    testing_cropped_img_list = "empty"
    training_bfm = "empty"
    testing_bfm = "empty"

    # Training the CNN on all data sets except last one
    for i in files:
        base_path = cwd + "/" + i
        print(base_path)

        # setting file paths
        spot_location_path = base_path + "/spatial/" + get_file_path_by_substring("tissue_positions_list.csv", base_path + "/spatial")
        bfm_dir = base_path
        bfm_filename = get_file_path_by_substring("filtered_feature_bc_matrix.h5", base_path)
        image_path = base_path + "/" + get_file_path_by_substring("image.tif", base_path)

        # loading data
        spot_locations = get_spot_locations(spot_location_path)
        bfm, spot_locations = load_bfm(bfm_dir,
                                       bfm_filename,
                                       spot_locations,
                                       number_of_genes_to_predict,
                                       high_expression_genes)

        if use_max_samples:
            number_of_samples_to_use_per_dataset = len(bfm)
        train_indexes, test_indexes = generate_train_test_indexes(number_of_samples_to_use_per_dataset)

        # Getting cropped image np arrays
        training_cropped_img_list_i, testing_cropped_img_list_i = get_cropped_images(image_path,
                                                                                     spot_locations,
                                                                                     cropped_img_width,
                                                                                     train_indexes,
                                                                                     test_indexes)
        if type(training_cropped_img_list) == str and type(testing_cropped_img_list) == str:
            training_cropped_img_list, testing_cropped_img_list = training_cropped_img_list_i, testing_cropped_img_list_i
        else:
            training_cropped_img_list = np.concatenate((training_cropped_img_list, training_cropped_img_list_i))
            testing_cropped_img_list = np.concatenate((testing_cropped_img_list, testing_cropped_img_list_i))

        # Getting bfm np arrays
        training_bfm_i, testing_bfm_i = partition_bfm(bfm,
                                                      train_indexes,
                                                      test_indexes)
        if type(training_bfm) == str and type(testing_bfm) == str:
            training_bfm, testing_bfm = training_bfm_i, testing_bfm_i
        else:
            training_bfm = np.concatenate((training_bfm, training_bfm_i))
            testing_bfm = np.concatenate((testing_bfm, testing_bfm_i))

    # Neural Network
    train_network(training_bfm,
                  training_cropped_img_list,
                  testing_bfm,
                  testing_cropped_img_list,
                  number_of_genes_to_predict)


if __name__ == "__main__":
    main()
