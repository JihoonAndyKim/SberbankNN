#########################################
# This file performs the  XGBoost model #
#########################################

#Read in the data
data <- read.csv("train.csv")
macro <- read.csv("macro.csv")
data = merge(data, macro, by = "timestamp")

newdat = data

#Treat all the NAs
for(i in 1:ncol(newdat)){
  if(!is.factor(newdat[,i])) {
    data[is.na(data[,i]), i] <- median(data[,i], na.rm = TRUE)
  }
  else {
    newdat[is.na(newdat[,i]), i] <- addNA(newdat[is.na(newdat[,i]), i])
    levels(newdat[is.na(newdat[,i]), i]) <- c(levels(newdat[is.na(newdat[,i]), i]), -1)
  }
}

library(drat)
library(xgboost)
library(Matrix)
#Split into training and dev sets
set.seed(123)
smp_size <- floor(0.90 * nrow(newdat))
train_ind <- sample(seq_len(nrow(newdat)), size = smp_size)
train = newdat[train_ind, ]
test = newdat[-train_ind, ]

#Force data.frame to be a matrix to input into our model
sparse.train = sparse.model.matrix(price_doc~.-1, data = train)
sparse.test = sparse.model.matrix(price_doc~.-1, data = test)
labels = train[complete.cases(train), "price_doc"]
labels_test = test[complete.cases(test), "price_doc"]

#Run XGBoost
bstSparse <- xgboost(data = sparse.train, label = labels, 
                     max.depth = 9, eta = 0.23, nthread = 8, 
                     nround = 100, lambda = 1.7,
                     objective = "reg:linear")

#Compute RMSLE
pred = predict(bstSparse, sparse.test)

library(Metrics)

pred[which(pred < 0)] = 0
linear.rmsle = rmsle(labels_test, pred)
print(linear.rmsle)

