barcodes <- read.table(file = "/Users/colten/Desktop/Perkins_Lab_ST/Identifying (usable) image spots/data/filtered_feature_bc_matrix/barcodes.tsv", sep = '\t')

features <- read.table(file = "/Users/colten/Desktop/Perkins_Lab_ST/Identifying (usable) image spots/data/filtered_feature_bc_matrix/features.tsv", sep = '\t')

library(Matrix)
#This should be read into a more easily understandable table
bf_matrix <- Matrix::readMM("/Users/colten/Desktop/Perkins_Lab_ST/Identifying (usable) image spots/data/filtered_feature_bc_matrix/matrix.mtx")

colnames(bf_matrix) = barcodes$V1
rownames(bf_matrix) = features$V1


# go through the bf_matrix and filter for each spot: (set barcode to rowname by finding barcode in barcodes)
# within the spot go through each gene and lookup the gene ID index from features.
# assign the gene column at the spot row to the value given



library(readtext)
text <- read.t

dat <- read.table(text="person1    12    15
    person2    15    18
    person3    20    14", stringsAsFactors=FALSE
)