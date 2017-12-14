######################################################
# This file performs the PCR, Ridge and LASSO models #
######################################################
library(Metrics)
library(pls)
library(glmnet)

#Read in the data
data <- read.csv("train.csv")
newdat = data
newdat[is.na(newdat)] <- 0

#Reduce to the variables that the linearReg.R file deemed as the most important to
#further reduce our feature space
variables = as.character(read.table(file = "important_variables.txt")[,1])

#Separate into training and test sets
set.seed(123)
smp_size <- floor(0.90 * nrow(newdat))
train_ind <- sample(seq_len(nrow(newdat)), size = smp_size)
train = newdat[train_ind, ]
test = newdat[-train_ind, ]

linear.data = train[, !sapply(train, is.factor)]
linear.test = test[, !sapply(train, is.factor)]
linear.data = linear.data[, variables]
linear.test = linear.test[, variables]

#Perform PCR
pcr.mod = pcr(log(price_doc) ~ ., data = linear.data)
pred <- predict(pcr.mod, newdata = linear.test)

#Compute RMSLE
linear.rmsle = rmsle(test[,"price_doc"], exp(pred)) 
linear.rmsle


#Create Ridge regression model and choose the RMSLE with the best Lambda
response = dim(linear.data)[2]
ridge.mod <- glmnet(as.matrix(linear.data[,-response]), 
                    log(linear.data[,response]), alpha = 0, 
                    lambda = seq(0.1, 300, 1))
ridge.pred <- predict(ridge.mod, as.matrix(linear.test[,-response]))

for(i in 1:dim(ridge.pred)[2]) {
  linear.rmsle = rmsle(test[,"price_doc"], exp(ridge.pred[,i]))
  print(linear.rmsle)
}

#Create LASSO regression modeland choose the RMSLE with the best lambda
lasso.mod <- glmnet(as.matrix(linear.data[,-response]), 
                    log(linear.data[,response]), alpha = 1, 
                    lambda = seq(0.01, 10, 0.1))
lasso.pred <- predict(lasso.mod, as.matrix(linear.test[,-response]))

for(i in 1:dim(lasso.pred)[2]) {
  linear.rmsle = rmsle(test[,"price_doc"], exp(lasso.pred[,i])) 
  print(linear.rmsle)
}