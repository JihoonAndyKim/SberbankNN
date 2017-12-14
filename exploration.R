######################################################################
# This file explores the data and takes a look at the missing values #
######################################################################
data <- read.csv("train.csv")

#Find all the variables with missing data
missing <- c()
for (i in 1:dim(data)[2]) {
  missing <- c(missing, sum(is.na(data[,i]))/length(data[,i]))
}

#Check how many are missing in each missing category and create a histogram
missing_name = names(data)[missing > 0]
missing = missing[missing > 0]
barplot(missing, names.arg = missing_name)

missing <- c()
for (i in 1:dim(macro)[2]) {
  missing <- c(missing, sum(is.na(macro[,i]))/length(macro[,i]))
}

missing_name = names(macro)[missing > 0]
missing = missing[missing > 0]
barplot(missing, names.arg = missing_name)

#Take a look at the log transform price vs log transformed square footage

plot(log(price_doc) ~ log(full_sq), data = data, xlab = "Log(square foot)",
     ylab = "Log(price)", main = "Log-log plot of square footage vs. price")

#Run linearReg.R before this module to load all the appropriate values

#Take a look at the histogram of price to see our price distribution.
data_col = 292
hist(log(data[,data_col]), breaks = seq(0, max(log(data[,data_col])), length.out = 400), 
     xlim=c(14, 18), xlab = "Price", main = "Histogram of Price")
