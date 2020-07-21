#INSTALL PACKAGES
!pip install scanpy
!pip install --editable .
!pip install anndata2ri

#LOAD SPOT LOCATION
spot_locations = pd.read_csv("drive/Shared drives/Perkins_Lab_ST/data/spatial/tissue_positions_list.csv", header=None, names = ["barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres_yValue", "pxl_row_in_fullres_xValue"])
spot_locations = spot_locations[spot_locations['in_tissue']==1]
spot_locations.reset_index(drop=True, inplace=True)

cropped_img_width = 100 #272

#LOAD BARCODE FEATURE MATRIX

adata = sc.read_visium("/content/drive/Shared drives/Perkins_Lab_ST/data", genome=None,
                       count_file='V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5', library_id=None,
                       load_images=True)

adata.var_names_make_unique()
sc.pp.filter_cells(adata, min_counts=5000)
sc.pp.filter_cells(adata, max_counts=35000)

# normalize the data with scRNA normalization
# useful site: https://anndata.readthedocs.io/en/stable/anndata.AnnData.html#anndata.AnnData
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=100)

adata = adata[:, adata.var["highly_variable"]]

# equivalent to the GetAssayData
# 1 expression matrix
barcode_feature_matrix = adata.to_df()

# filter number of barcodes to get a list of barcodes
barcodes = list(barcode_feature_matrix.index.values)
sorted_barcodes = barcodes
sorted_barcodes.sort()

if barcodes == sorted_barcodes:
    spot_locations.index = spot_locations["barcode"]
    spot_locations.sort_index(inplace=True)
    spot_locations
    for i in spot_locations.index:
        if i in barcodes:
            continue
        else:
            spot_locations.drop([i], axis=0,
                                inplace=True)  # inplace = True, overwrite the current dataframe when set to true, not creating a new variable here (False)
else:
    print("something's wrong. The barcodes from the barcode feature matrix are not in the correct order")

#LOAD IN SPOT IMAGES

wsi_file_name = 'drive/Shared drives/Perkins_Lab_ST/data/V1_Breast_Cancer_Block_A_Section_1_image.tif'
PIL.Image.MAX_IMAGE_PIXELS = 1000000000
wsi = Image.open(wsi_file_name)
img_array = np.array(wsi)
print("shape of wsi: ", img_array.shape)

#defining a function to get a cropped spot image given the x and y coordinates of the center of the spot
def Crop_Spot(img_array, x, y):
  cropped_img = img_array[int(y-(cropped_img_width/2)):int(y+(cropped_img_width/2)),
                          int(x-(cropped_img_width/2)):int(x+(cropped_img_width/2))]
  #preprocessing
  #CHANGE EACH CHANNEL INDIVIDUALLY ONCE WE START USING PYCHARM
  #cropped_img[:, :, 0] = (cropped_img[:, :, 0] - cropped_img[:, :, 0].mean()) / 25500
  #cropped_img[:, :, 1] = (cropped_img[:, :, 1] - cropped_img[:, :, 1].mean()) / 25500
  #cropped_img[:, :, 2] = (cropped_img[:, :, 2] - cropped_img[:, :, 2].mean()) / 25500
  cropped_img = cropped_img / [cropped_img[:, :, 0].mean(), cropped_img[:, :, 1].mean(), cropped_img[:, :, 2].mean()]
  return cropped_img / 1000

#partition data indexes into train and test
np.random.seed(1998)
train_test_indexes = [i for i in range(len(spot_locations))]
random.shuffle(train_test_indexes)
train_indexes = train_test_indexes[:int(0.7*len(train_test_indexes))]
train_indexes.sort()
train_indexes = train_indexes[:10]
test_indexes = train_test_indexes[int(0.7* len(train_test_indexes)):]
test_indexes.sort()
test_indexes = test_indexes[:10]
for i in train_indexes:
  if i in test_indexes:
    print("something went wrong. training and testing indexes overlap")

#crop image into train and test
training_cropped_img_list = []
for i in train_indexes:
  cropped_img = Crop_Spot(img_array, spot_locations["pxl_row_in_fullres_xValue"][i], spot_locations["pxl_col_in_fullres_yValue"][i])
  training_cropped_img_list.append(cropped_img)
training_cropped_img_list = np.array(training_cropped_img_list)

testing_cropped_img_list = []
for i in test_indexes:
  cropped_img = Crop_Spot(img_array, spot_locations["pxl_row_in_fullres_xValue"][i], spot_locations["pxl_col_in_fullres_yValue"][i])
  testing_cropped_img_list.append(cropped_img)
testing_cropped_img_list = np.array(testing_cropped_img_list)

sample_img = Image.fromarray(training_cropped_img_list[0], "RGB")
plt.imshow(sample_img);

#NEURAL NETWORK
bfm = barcode_feature_matrix.to_numpy()
training_bfm = bfm[train_indexes]
testing_bfm = bfm[test_indexes]

#Creating the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(cropped_img_width, cropped_img_width, 3)),
  tf.keras.layers.Dense(100, activation='sigmoid'),
  tf.keras.layers.Dense(100, activation='sigmoid'),
  tf.keras.layers.Dense(len(testing_bfm[0]), activation='relu')
])
model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(0.0001),
    metrics=['mae'],
)
print(model.summary())

#training
model.fit(
    training_cropped_img_list,
    training_bfm,
    batch_size=1,
    epochs=1000,
    validation_split=0.2
)

predictions = model.predict(training_cropped_img_list)
#predictions = model.predict_on_batch(cropped_img_list[:100])
#predictions.max()

weights, biases = model.layers[2].get_weights()
print(weights[0])
print(weights[1])
print(train_indexes)
