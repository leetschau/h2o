library(h2o)
h2o.init(nthreads = -1)

datasrc <- "https://raw.githubusercontent.com/DarrenCook/h2o/bk/datasets/"
fullpath <- paste0(datasrc, "iris_wheader.csv")
data <- h2o.importFile(fullpath)
y <- "class"
x <- setdiff(names(data), y)
parts <- h2o.splitFrame(data, 0.8)
train <- parts[[1]]
test <- parts[[2]]

m <- h2o.deeplearning(x, y, train)
p <- h2o.predict(m, test)
h2o.performance(m, test)
