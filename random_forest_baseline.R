##############################################
# This file performs the Random Forest model #
##############################################

library(randomForest)
library(Metrics)

#Paralellize
library(doParallel) 
cl <- makeCluster(detectCores(), type='PSOCK')
registerDoParallel(cl)

#Read in the data and treat missing values
data <- read.csv("train.csv")
macro <- read.csv("macro.csv")
data = merge(data, macro, by = "timestamp")
newdat = data
newdat[is.na(newdat)] <- 0

#Separate into training and test sets
set.seed(123)
smp_size <- floor(0.90 * nrow(newdat))
train_ind <- sample(seq_len(nrow(newdat)), size = smp_size)
train = newdat[train_ind, ]
test = newdat[-train_ind, ]

linear.data = train[, !sapply(train, is.factor)]
linear.test = test[, !sapply(train, is.factor)]


#Run Random Forest
response = which(names(linear.test) == "price_doc")
rf.mod <- foreach(ntree=rep(3, 10), .combine=combine, .multicombine=TRUE,
                  .packages='randomForest') %dopar% {
                    randomForest(log(price_doc) ~ ., data = linear.data, ntree=10)
                  }

#Compute RMSLE
pred <- predict(rf.mod, linear.test[, -response])

linear.rmsle = rmsle(test[,"price_doc"], exp(pred))
print(linear.rmsle)