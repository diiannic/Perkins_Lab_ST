library(Seurat)
feature_matrix <- Load10X_Spatial(data.dir = "/Users/colten/Desktop/Perkins_Lab_ST/Identifying (usable) image spots/data",
                         filename = "V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5",
                         assay = "Spatial",
                         slice = "slice1",
                         filter.matrix = TRUE,
                         to.upper = FALSE)

bf_matrix <- as.matrix(GetAssayData(object = feature_matrix, slot = "counts"))






