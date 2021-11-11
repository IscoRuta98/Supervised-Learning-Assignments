rm(list=ls())

#Load libraries
library(corrplot)
library(GGally)
library(ggplot2)
library(leaps)
library(glmnet)
library(stargazer)

setwd("C:/Users/Isco/Documents/Masters/Coursework/STA5076Z-Supervised Learning/Assignment 1")
boston.data = read.csv('my_boston.csv') #Read in all 400 observations

# head(boston.data)

#### PRE-PROCESSING ####
boston.data$chas = factor(x = boston.data$chas,levels = c(0,1),labels = c("otherwise","Tract Bounds"))
boston.data$rad = as.factor(boston.data$rad)

# boston.data$chas = as.factor(boston.data$chas)


#### Exploratory Data Analysis (EDA) ####

### Distribution of the dependent variable

# Layout to split the screen
layout(mat = matrix(c(1,2),2,1, byrow=TRUE),  height = c(1,8))

# Draw the boxplot and the histogram 
par(mfrow = c(1,1))
par(mar=c(0, 3.1, 1.1, 2.1))
boxplot(boston.data$medv , horizontal=TRUE, xaxt="n" , col=rgb(0.8,0.8,0,0.5) , frame=F)
par(mar=c(4, 3.1, 1.1, 2.1))
hist(boston.data$medv , breaks=40 , col=rgb(0.2,0.8,0.5,0.5) , border=F ,
     main="Distribution of Median value of owner-occupied homes" ,
     xlab="Median value of owner-occupied homes (in $1000's)",ylab = "Frequency")

# Continuous variables
# Descriptive statistics Table

capt <- "Continuous variables."
tableContinuous(vars = cont_var, cap = capt, lab = "tab: cont1",
                stats = c("min", "q1", "median", "mean", "q3", "max", 
                          "s"),longtable = FALSE)

par(mfrow = c(1,3))
boxplot(cont_var$crim,main = "crim", col = "blue")
boxplot(cont_var$zn, main = "zn", col = "green")
boxplot(cont_var$rm, main = "rm", col = "red")
par(mfrow = c(1,1))

# Correlation, scatterplots, and histograms.
ggpairs(boston.data[,-c(4,9)])

# Correlation Matrix
cont_var = boston.data[,-c(4,9)]
# corrplot(corr = cor(cont_var),method = "number")

par(mfrow=c(1,2))
plot(medv~lstat,data = boston.data)
plot(scale(log(medv))~scale(lstat,center = T,scale = T),data = boston.data)

plot(log(medv)~nox,data = boston.data)
par(mfrow = c(1,1))

### Categorical variable

#Boxplots
  
ggplot(boston.data, aes(x=chas, y=medv, fill=chas)) + 
  geom_boxplot(alpha=0.3) +
  theme(legend.position="none") +
  ggtitle("Median value of owner-occupied homes grouped by Charles River") +
  ylab("Median value of owner-occupied homes ($1000's)")

ggplot(boston.data, aes(x=rad, y=medv, fill=rad)) + 
  geom_boxplot(alpha=0.3) +
  theme(legend.position="none") + 
  ggtitle("Median value of owner-occupied homes grouped by radial highways.") +
  ylab("Median value of owner-occupied homes ($1000's)")

#### DATA PROCCESSING ####

# Standardisation
scale.data = data.frame(scale(x = cont_var),boston.data$chas,boston.data$rad)

# Split the data
set.seed(2021)
idx = sample(x = 1:nrow(boston.data),size = 0.8*nrow(boston.data),replace = FALSE)
train = scale.data[idx,]
test  = scale.data[-idx,] 


#### MLR MODEL ####
mlr.mod = lm(formula = medv~.,data = train)
summary(mlr.mod)

baseline = predict(object = mlr.mod,newdata = test)

baseline.mse = mean((baseline - test$medv)^2)
baseline.mse

#### QUESTION 2 ####

#### Variable Selection ####

### Best Subset Selection

regfit.full = regsubsets (medv ~ .,data=train ,nvmax =12)
reg.summary = summary(regfit.full)
#names(reg.summary)


# Plotting RSS, Adj R-sq, Cp, and BIC
par(mar =c(5.1, 4.1, 4.1, 2.1))
par(mfrow=c(2,2))
plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",
     type="l")
which.min(reg.summary$rss)
points (12, reg.summary$rss[12], col ="blue",cex =2, pch =20)

plot(reg.summary$adjr2,xlab="Number of Variables",
     ylab=" Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points (12, reg.summary$adjr2[12], col ="red",cex =2, pch =20)

plot(reg.summary$cp ,xlab =" Number of Variables ",ylab="Cp",
       type="l")
which.min (reg.summary$cp )
points (12, reg.summary$cp[12], col ="green",cex =2, pch =20)

which.min (reg.summary$bic )
plot(reg.summary$bic ,xlab=" Number of Variables ",ylab=" BIC",
       type="l")
points (8, reg.summary$bic[8], col ="yellow",cex =2, pch =20)

# Make predictions and calculate test mse

test.mat = model.matrix (medv~.,data=test)

coef.8 = coef(regfit.full ,8)
pred.8 = test.mat[,names(coef.8)]%*% coef.8
mean(( test$medv-pred.8)^2)

coef.12 = coef(regfit.full ,12)
pred.12 = test.mat[,names(coef.12)]%*% coef.12
mean(( test$medv-pred.12)^2)


#### Lasso Regularization, CV: 10 fold ####

grid = 10^seq(10,-2,length =100)
x = model.matrix(medv~.,scale.data)[,-11]
y = scale.data$medv

lasso.mod = glmnet(x[idx,],y[idx],alpha =1, lambda =grid,sstandardize = FALSE)
par(mfrow = c(1,1))
plot(lasso.mod,label = T)

# Implementing cross-validation 
set.seed (2021)
cv.out = cv.glmnet(x[idx,],y[idx],alpha =1,nfolds = 10)
plot(cv.out)
bestlam = cv.out$lambda.min
lasso.cv.pred = predict(object = cv.out ,s=bestlam,newx=x[-idx ,])
mean(( lasso.cv.pred - y[-idx])^2)

#### Interaction Term ####

inter.mod.1 = lm(medv ~zn+nox+rm+dis+ptratio+lstat*boston.data.chas,
               data = train)
summary(inter.mod.1)
predict.inter1 = predict(object = inter.mod.1,newdata = test)
mse(actual = test$medv,predicted = predict.inter1)

inter.mod.2 = lm(medv ~zn*boston.data.chas+nox+rm+dis+ptratio+lstat,
                 data = train)
summary(inter.mod.2)
predict.inter2 = predict(object = inter.mod.2,newdata = test)
mse(actual = test$medv,predicted = predict.inter2)

inter.mod.3 = lm(medv ~zn+nox*boston.data.chas+rm+dis+ptratio+lstat,
                 data = train)
summary(inter.mod.3)
predict.inter3 = predict(object = inter.mod.3,newdata = test)
mse(actual = test$medv,predicted = predict.inter3)

inter.mod.4 = lm(medv ~zn+nox+rm*boston.data.chas+dis+ptratio+lstat,
                 data = train)
summary(inter.mod.4)
predict.inter4 = predict(object = inter.mod.4,newdata = test)
mse(actual = test$medv,predicted = predict.inter4)


inter.mod.6 = lm(medv ~zn+nox+rm+dis+ptratio*boston.data.chas+lstat,
                 data = train)
summary(inter.mod.6)
predict.inter6 = predict(object = inter.mod.6,newdata = test)
mse(actual = test$medv,predicted = predict.inter6)

inter.mod.5 = lm(medv ~zn+nox+rm+dis*boston.data.chas+ptratio+lstat,
                 data = train)
summary(inter.mod.5)
predict.inter5 = predict(object = inter.mod.5,newdata = test)
mse(actual = test$medv,predicted = predict.inter5)


#### RESIDUAL DIAGNOSTICS ####

# Interaction term model
par(mfrow = c(1,2))
plot(inter.mod.3, which = 1, col = c("black"))
plot(inter.mod.3, which=2, col=c("black")) # Residuals vs Fitted Plot
par(mfrow = c(1,1))

plot(density(resid(inter.mod.3)), main="OLS Residuals", col=4)
hist(resid(inter.mod.3), freq=FALSE, add=TRUE, border=2)
