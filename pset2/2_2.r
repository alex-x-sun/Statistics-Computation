
library(data.table)
library(Rtsne)


file = 'e:/MIT4/6.439/pset2/Trapnell/Trapnell.csv'
data = fread(file, sep = ",", header= FALSE)


# tSNE
set.seed(0)
data_tsne = tsne(as.matrix(data))


