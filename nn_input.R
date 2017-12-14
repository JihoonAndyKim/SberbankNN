##########################################################
# Cleaned dataset to be inputted into the Neural Network #
##########################################################
data <- read.csv("train.csv")
macro <- read.csv("macro.csv")
data = merge(data, macro, by = "timestamp")

newdat = data
for(i in 1:ncol(newdat)){
  if(!is.factor(newdat[,i])) {
    data[is.na(data[,i]), i] <- median(data[,i], na.rm = TRUE)
  }
  else {
    newdat[is.na(newdat[,i]), i] = addNA(newdat[is.na(newdat[,i]), i])
  }
  
  if(is.factor(newdat[,i]) || is.integer(newdat[,i])) {
    newdat[, i] = as.character(newdat[,i])
  }
}

write.csv(newdat, "house-macro.csv", col.names = TRUE)
