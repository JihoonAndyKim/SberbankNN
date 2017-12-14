######################################################################
# This file performs the Linear Regression model to choose variables #
######################################################################
##Load in the data
library(Metrics)
library(car)

data <- read.csv("train.csv")
#292 data columns
newdat = data
newdat[is.na(newdat)] <- 0


#Separate into training and test sets
set.seed(123)
smp_size <- floor(0.9 * nrow(newdat))
train_ind <- sample(seq_len(nrow(newdat)), size = smp_size)
train = newdat[train_ind, ]
test = newdat[-train_ind, ]

linear.data = train[, !sapply(train, is.factor)]
linear.test = test[, !sapply(train, is.factor)]
linear.mod <- lm(log(price_doc) ~ ., data = linear.data)

#This method shows that we should remove  ekder_all, x0_6_all, and x7_14_all
#alias(linear.mod)
multicollinear = c(which(is.na(linear.mod$coefficients))-1)
linear.data = linear.data[, -multicollinear]
linear.test = linear.test[, -multicollinear]

extreme_Vals = c(3528, 13547, 2119, 10090, 7458)
linear.data = linear.data[-extreme_Vals, ]
linear.mod <- lm(log(price_doc) ~ ., data = linear.data)



#We choose the critical alpha value to be 0.15
alpha_critical = 0.10
backward.data = linear.data
backward.mod = linear.mod

flag = TRUE
#Loop until we find the best model according to significance values
count = 0
while(flag) {
  print(count)
  #Find the pvalues and find the largest one
  p.vals = summary(backward.mod)$coefficients[-1,4]
  largest.p = order(p.vals)
  to.remove.ind = largest.p[length(largest.p)]
  to.remove = p.vals[to.remove.ind]
  #Check to see if this pvalue is bigger than the critical value
  #If it is, then remove the variable associated with this pvalue.
  #If not, then stop since we can't find pvalues greater than the
  #critical value.
  if(to.remove > alpha_critical){
    backward.data = backward.data[, -to.remove.ind]
    backward.mod = lm(price_doc ~ ., data = backward.data)
  }
  else {
    break
  }
  count = count + 1
}
summary(backward.mod)

#With the updated values, compute which variables have high VIF factors
update.mod = backward.mod

inflated = c(which(vif(update.mod) >= 10)-1)

#Reduce the number of variables that have VIF scores over 10
if(length(inflated) == 0) {
  break
}
linear.data = linear.data[, -inflated]
linear.test = linear.test[, -inflated]

#Update the model with these predictors and with the log-transformed price, perform 
#linear regression
update.mod = lm(log(price_doc) ~ ., data = linear.data)


pred <- predict(update.mod, newdata = linear.test)
pred[which(pred <= 0)] <- 0

#Save these new variables and 
variables <- names(linear.data)
write(variables, "important_variables")

#Compute RMSLE
linear.rmsle = rmsle(test[,"price_doc"], exp(pred)) 
linear.rmsle