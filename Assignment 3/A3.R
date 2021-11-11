## STA5076Z: ASSIGNMENT 3, 
## FRANCISCO RUTAYEBESIBWA: RTYFRA001

#### QUESTION 1 ####
# Clear Environment
rm(list = ls())

setwd("~/Masters/Coursework/STA5076Z-Supervised Learning/Assignment 3")

# Load lobraries
library(gridExtra)
library(ElemStatLearn)
library(tidyverse)
library(caret)
library(e1071)
library(factoextra)
library(h2o)

# Load the data
data("spam")
str(spam)

# checking if there are missing values
sum(is.na(spam)) #NO N/A values

#### EDA ####

# Check proportions of the data
prop.table(table(spam$spam))

# Checking if target can be split linearly.
p1 = ggplot(spam, aes(x=A.57, y=A.56, color=spam)) +
  geom_point()

p2 = ggplot(spam,aes(x=A.1,y=A.2,color=spam)) +
  geom_point()

p3 = ggplot(spam,aes(x=A.9,y=A.12,color=spam)) +
  geom_point()

p4 = ggplot(spam,aes(x=A.55,y=A.54,color=spam)) +
  geom_point()

# Put the plots in one place.
grid.arrange(p1,p2,p3,p4,nrow=2,ncol=2)

#### DATA PREPROCESSING ####
#scale the numerical features
scale_spam = data.frame(scale(x = spam[,-58]),spam$spam)
colnames(scale_spam)[58] = "spam"
spam = scale_spam

# Split the data
set.seed(2004)
idx = createDataPartition(spam$spam,1,.8)[[1]]
idx
Train = spam[idx,]
Test = spam[-idx,]
prop.table(table(Train$spam))
prop.table(table(Test$spam))

#### MODEL BUILDING ####

# All models ar ebuilt using the caret package.

# Want to perfrom 5-fold cross validation
trctrl = trainControl(method = "cv", number = 5)

#### Linear SVM ####
# Prepare grid search
grid.linear = expand.grid(C = seq(2^(-5),2^(4),by=2))
set.seed(2004)
# Train the model
svm_Linear_Grid =  train(spam ~., data = Train, method = "svmLinear",
                         trControl=trctrl,
                         tuneGrid = grid.linear)

svm_Linear_Grid
plot(svm_Linear_Grid,main="Accuracy error on the Training set")

# Make predictions on training and testing data.
linear_svm_pred_train = predict(svm_Linear_Grid, Train)
(mean(linear_svm_pred_train == Train$spam))*100

linear_svm_pred_test = predict(svm_Linear_Grid, Test)
(mean(linear_svm_pred_test == Test$spam))*100

#### Radial SVM ####
# Prepare grid search
grid.radial = expand.grid(C = seq(2^(-5),2^(4),by=2),
                          sigma = c(0.01,0.1,1,10))
# Train the model
set.seed(2004)
svm_Radial_Grid =  train(spam ~., data = Train, method = "svmRadial",
                         trControl=trctrl,
                         tuneGrid = grid.radial)
svm_Radial_Grid
plot(svm_Radial_Grid)

# Make predictions.
radial_svm_pred_train = predict(svm_Radial_Grid, Train)
(mean(radial_svm_pred_train == Train$spam))*100

radial_svm_pred_test = predict(svm_Radial_Grid, Test)
(mean(radial_svm_pred_test == Test$spam))*100


#### Polynomial SVM ####

# Prepare grid search
grid.poly = expand.grid(C = seq(2^(-5),2^(4),by=2),
                        degree = c(1,3,7,10),
                        scale  = 0.01)
set.seed(2004)
svm_Poly_Grid =  train(spam ~., data = Train, method = "svmPoly",
                       trControl=trctrl,
                       tuneGrid = grid.poly)

svm_Poly_Grid
plot(svm_Poly_Grid)

# Make predictions
poly_svm_pred_train = predict(svm_Poly_Grid, Train)
(mean(poly_svm_pred_train == Train$spam))*100

poly_svm_pred_test = predict(svm_Poly_Grid, Test)
(mean(poly_svm_pred_test == Test$spam))*100

#### RESULTS ####
# LINEAR
# Create a confusion matrix using CARET package
conf.linear.test  = confusionMatrix(linear_svm_pred_test,Test$spam)

# Extract metrics for test set from the confusion matrix above.
100*conf.linear.test[["overall"]][["Accuracy"]] # Accuracy
100*conf.linear.test[["byClass"]][["Recall"]] # Recall
100*conf.linear.test[["byClass"]][["Specificity"]] # Specificity
100*conf.linear.test[["byClass"]][["F1"]] # F1 Score

# RADIAL
# Create a confusion matrix using CARET package
conf.radial.test  = confusionMatrix(radial_svm_pred_test,Test$spam)

# Extract metrics for test set from the confusion matrix above.
100*conf.radial.test[["overall"]][["Accuracy"]] # Accuracy
100*conf.radial.test[["byClass"]][["Recall"]] # Recall
100*conf.radial.test[["byClass"]][["Specificity"]] # Specificity
100*conf.radial.test[["byClass"]][["F1"]] # F1 Score


# POLYNOMIAL
# Create a confusion matrix using CARET package
conf.poly.test  = confusionMatrix(poly_svm_pred_test,Test$spam)

# Extract metrics for test set from the confusion matrix above.
100*conf.poly.test[["overall"]][["Accuracy"]] # Accuracy
100*conf.poly.test[["byClass"]][["Recall"]] # Recall
100*conf.poly.test[["byClass"]][["Specificity"]] # Specificity
100*conf.poly.test[["byClass"]][["F1"]] # F1 Score

#### QUESTION 2 ####
## STA5076Z: ASSIGNMENT 3, 
## FRANCISCO RUTAYEBESIBWA: RTYFRA001

# Clear Environment
rm(list = ls())

setwd("~/Masters/Coursework/STA5076Z-Supervised Learning/Assignment 3")

# Load lobraries
library(gridExtra)
library(ElemStatLearn)
library(tidyverse)
library(caret)
library(e1071)
library(factoextra)
library(h2o)

# Read in the data
data = read.csv(file = "Train_Digits_2021.csv")
test = read.csv(file = "Test_Digits_2021.csv")

# Classify label as even or odd

# Create a function to do this.
is.even <- function(x) x %% 2 == 0
data$Digit =  as.factor(ifelse(test = is.even(data$Digit) == TRUE,
                               yes = "Even",no = "Odd"))
test$Digit =  as.factor(ifelse(test = is.even(test$Digit) == TRUE,
                               yes = "Even",no = "Odd"))

sum(is.na(data$Digit))
# No need to scale, there are no missing values.
# Split the data into training and validation set.

set.seed(2004)
idx = createDataPartition(data$Digit,1,.8)[[1]]
idx
Train = data[idx,]
Valid = data[-idx,]
prop.table(table(Train$Digit))
prop.table(table(Valid$Digit))

## Initialise H2O Connection
## Start a local H2O cluster directly from R
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE,min_mem_size = "1g")
localH2O = h2o.init()

# Fromat the datasets to h20 format
options("h2o.use.data.table"=FALSE)
h2oTrain = as.h2o(Train)
h2oValid = as.h2o(Valid)
h2oTest = as.h2o(test)

#### MODELLING ####

# Setup a grid search
hyper_params <- list(
  activation=c("RectifierWithDropout","TanhWithDropout"),
  rate = c(0.005,0.5,0.1),
  hidden=list(c(522), c(50), c(150), c(350), c(50,150), c(522,270)),
  input_dropout_ratio=0.5,
  l2=seq(0,1e-4,1e-6)
)
hyper_params

## Stop once the top 5 models are within 1% of each other (i.e., the windowed
## average varies less than 1%)
search_criteria = list(
  strategy = "RandomDiscrete", 
  max_runtime_secs = 360, 
  max_models = 100, 
  seed=2004,
  stopping_rounds=5, 
  stopping_tolerance=1e-2)

dl_random_grid = h2o.grid(
  algorithm="deeplearning",
  training_frame=h2oTrain,
  validation_frame=h2oValid, 
  x=2:785, 
  y=1,
  epochs=10,
  stopping_metric="misclassification",
  stopping_tolerance=1e-2,        ## stop when missclass does not improve 
  stopping_rounds=2,
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  max_w2=10,                      ## can help improve stability for Rectifier
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  nfolds = 5
)                                

# Obtain a summary of results
dl_random_grid@summary_table


#### Make predictions ####
predictions  = h2o.predict(nn.model1, h2oTrain)
predictions
pred_classes = as.factor(as.matrix(predictions$predict))

pred_classes = as.data.frame(pred_classes)
colnames(pred_classes) = "Digit"

# Write predictions as csv files
write.csv(x = pred_classes,file = "RTYFRA001_NN.csv",row.names = F)
