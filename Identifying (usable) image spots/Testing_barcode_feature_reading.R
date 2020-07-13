library(Seurat)
feature_matrix <- Load10X_Spatial(data.dir = "/Users/colten/Desktop/Perkins_Lab_ST/Identifying (usable) image spots/data",
                         filename = "V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5",
                         assay = "Spatial",
                         slice = "slice1",
                         filter.matrix = TRUE,
                         to.upper = FALSE)

bf_matrix <- GetAssayData(object = feature_matrix, slot = "counts")
#image(bf_matrix)

names <- bf_matrix@Dimnames[1]
bf_matrix[,2]





# constants
data_dim <- 16
timesteps <- 8
num_classes <- 10

# define and compile model
# expected input data shape: (batch_size, timesteps, data_dim)
model <- keras_model_sequential() 
model %>% 
  layer_lstm(units = 32, return_sequences = TRUE, input_shape = c(timesteps, data_dim)) %>% 
  layer_lstm(units = 32, return_sequences = TRUE) %>% 
  layer_lstm(units = 32) %>% # return a single vector dimension 32
  layer_dense(units = 10, activation = 'softmax') %>% 
  compile(
    loss = 'categorical_crossentropy',
    optimizer = 'rmsprop',
    metrics = c('accuracy')
  )

# generate dummy training data
x_train <- array(runif(1000 * timesteps * data_dim), dim = c(1000, timesteps, data_dim))
y_train <- matrix(runif(1000 * num_classes), nrow = 1000, ncol = num_classes)
dim(y_train)
# generate dummy validation data
x_val <- array(runif(100 * timesteps * data_dim), dim = c(100, timesteps, data_dim))
y_val <- matrix(runif(100 * num_classes), nrow = 100, ncol = num_classes)

# train
model %>% fit( 
  x_train, y_train, batch_size = 64, epochs = 5, validation_data = list(x_val, y_val)
)

#test
score = model %>% evaluate(x_val, y_val, batch_size=64)

#show predictions
features <- model %>% predict(x_val)



#devtools::install_github("rstudio/keras", force = TRUE)
# library(keras)
# keras::install_keras()
# 
# 
# library("tensorflow")
# #tensorflow::use_condaenv("r-tensorflow")
# 
# # TF test
# sess = tf$Session()
# hello <- tf$constant('Hello, TensorFlow!')
# sess$run(hello)
