#Installing the required packages
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dbplyr)) install.packages("dbplyr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")

#Installing the required libraries
library(caret)
library(data.table)
library(dbplyr)
library(dplyr)
library(dslabs)
library(e1071)
library(caTools)
library(xgboost)

#Function to get the mode of a set of values
getmode <- function(v) {
       uniqv <- unique(v)
       uniqv[which.max(tabulate(match(v, uniqv)))] 
  }


#Loading the data
hr_train_set <- read.csv('D:/HARVARD CAPSTONE HR ANALYTICS PROJECT/hr_train.csv')
str(hr_train_set)

#Removing the employee_id column and duplicate values
hr_train_set$employee_id <- NULL
hr_train_set <- distinct(hr_train_set)

#Changing the class names from 1 and 0 to "yes" and "no"
for (i in 1:nrow(hr_train_set)) {
  hr_train_set$is_promoted[i] = if(hr_train_set$is_promoted[i] == 1) "yes" else "no"
}

#Viewing the first 6 rows of the training set
head(hr_train_set)

#Replacing the null values with the modes of the attributes

sapply(hr_train_set, function(x) sum(x=="" | is.na(x)))

#table that contains the number of missing values for each column
missing_values <- sapply(hr_train_set, function(x) sum(x=="" | is.na(x)))
missing_values <- (t(data.frame(missing_values)))

#table that contains the column names of the missing_values table
n <- names(data.frame(missing_values))

#barplot of number of missing values for all columns
barplot(missing_values, beside = T,
        col = rainbow(length(missing_values)), xaxt='n',main="Number of missing values for all columns",xlab="Column name",ylab="Number of values")

#legend for the above barplot
legend("topright", 
       legend = n, 
       fill = rainbow(length(missing_values)), ncol = 2,
       cex = 0.45)

#table without the null values of the hr_train_set
non_null_values <- na.omit(hr_train_set)

#number of non-null value records
print("Number of non-null values")
nrow(non_null_values)

#Assigning mode values to missing or NA values according to the appropriate subset of data by the getMode function
hr_train_set[hr_train_set$education == "" & hr_train_set$is_promoted == "no", grep("education", colnames(hr_train_set))] <- getmode(subset(hr_train_set, is_promoted == "no")$education) 
hr_train_set[hr_train_set$education == "" & hr_train_set$is_promoted == "yes", grep("education", colnames(hr_train_set))] <- getmode(subset(hr_train_set, is_promoted == "yes")$education)
hr_train_set[is.na(hr_train_set["previous_year_rating"]) & hr_train_set$is_promoted == "no", grep("previous_year_rating", colnames(hr_train_set))] <- getmode(subset(hr_train_set, is_promoted == "no")$previous_year_rating)
hr_train_set[is.na(hr_train_set["previous_year_rating"]) & hr_train_set$is_promoted == "yes", grep("previous_year_rating", colnames(hr_train_set))] <- getmode(subset(hr_train_set, is_promoted == "yes")$previous_year_rating)

print("Number of values in dataset")
nrow(hr_train_set)

#Balancing the classes

#barplot showing the proportion of promotions according to the dataset before undersampling
barplot(prop.table(table(hr_train_set$is_promoted)), col=c("orange","green"),main="Propn of people promoted/not promoted before undersampling")

#dataframe showing the proportion of promotions according to the dataset
print("Proportion of people with/without promotion")
data.frame(prop.table(table(hr_train_set$is_promoted))) 

#table showing the number of people promoted and not promoted
table(hr_train_set$is_promoted)

#Splitting the data into training and testing sets
set.seed(42)

temp_1 <- subset(hr_train_set, is_promoted == "yes")
temp_2 <- subset(hr_train_set, is_promoted == "no")

temp_2 <- temp_2[sample(nrow(temp_1)),]
#Creating the min_train data set
min_train <- rbind(temp_1, temp_2)

rows <- sample(nrow(min_train))
min_train <- min_train[rows,]

#Plotting the proportion of people in min_train who got promoted and the proportion who did not get promoted
barplot(prop.table(table(min_train$is_promoted)), col=c("orange","green"),main="Propn of people promoted/not promoted after undersampling")

#Data frame containing the proportion of people in min_train who got promoted and the proportion who did not get promoted
data.frame(prop.table(table(min_train$is_promoted)))

#Splitting min_train into training and testing data 
index = createDataPartition(min_train$is_promoted, p=4/5, list=FALSE)

#Training set has 4/5th of min_train data
training <- min_train[index,]

#Testing set has 4/5th of min_train data
testing <- min_train[-index,]

#Hyperparameters for training
params <- trainControl(method="cv",
                       number=5,
                       savePredictions=TRUE,
                       classProbs=TRUE)

#Training the Random Forest
random_forest <- train(as.factor(is_promoted)~.,
                       data=training,
                       method="rf",
                       trControl=params)
random_forest

#Calculating the confusion matrix for the random forest model
confusionMatrix(random_forest)

#Plotting the top 20 most important variables that affect the output according to random forest model
plot(varImp(random_forest, scale=FALSE), top=20,main="20 most important variables according to random forest model")

#Importance of the top 20 variables that affect the output according to random forest model
varImp(random_forest, scale=FALSE)

#Predicting for the testing data using random forest
rf_predict <- predict(random_forest, testing)

#Calculating the confusion matrix for the testing data using random forest
confusionMatrix(table(rf_predict, testing$is_promoted))

#Training the XGBoost tree

xgb_tree <- train(as.factor(is_promoted)~.,data=training,method="xgbTree",trControl=params)

xgb_tree

#Calculating the confusion matrix of xgboost tree
confusionMatrix(xgb_tree)
xgb_tree$pred

#Predicting for the testing data using xgboost
xgb_predict <- predict(xgb_tree, testing)

#Calculating the confusion matrix for the testing data using xgboost
confusionMatrix(table(xgb_predict, testing$is_promoted))

#Plotting the top 20 most important variables that affect the output according to xgboost model
plot(varImp(xgb_tree, scale=FALSE), top=20,main="20 most important variables according to XGBoost model")

#Importance of the top 20 variables that affect the output according to xgboost model
varImp(xgb_tree, scale=FALSE)

#Accuracy of random forest model is 81.99% and XGBoost is 83.76%,which is slightly better.
