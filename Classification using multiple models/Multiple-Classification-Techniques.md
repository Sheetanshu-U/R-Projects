Multiple Classification Techniques
================
Sheetanshu
8/3/2020

``` r
getwd()
```

    ## [1] "C:/Users/Admin/Desktop/Stats/Markdown/R-Projects/Classification using multiple models"

``` r
data.cars = read.csv("Cars.csv", header = TRUE)

View(data.cars)
```

\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#Exploratory data analysis
and data preparation

``` r
############checking complete cases

sum(complete.cases(data.cars))
```

    ## [1] 443

``` r
colSums(is.na(data.cars))
```

    ##       Age    Gender  Engineer       MBA  Work.Exp    Salary  Distance   license 
    ##         0         0         0         1         0         0         0         0 
    ## Transport 
    ##         0

\#\#\#\#\#\#\#\#\#The data set has one incomplete case

``` r
############taking complete cases for analysis

data.cars2 = data.cars[complete.cases(data.cars), ]
```

``` r
#########converting categorical variables into factors

data.cars2$Engineer = as.factor(data.cars2$Engineer)
data.cars2$MBA = as.factor(data.cars2$MBA)
data.cars2$license = as.factor(data.cars2$license)
```

``` r
########checking whether the target variable(Transport) is balanced or not 

prop.table(table(data.cars2$Transport))
```

    ## 
    ##         2Wheeler              Car Public Transport 
    ##        0.1873589        0.1376975        0.6749436

\#\#\#\#\#\#\#\#\#\#\#\#\#The category cars make up 13.7% of the data
while other transport make up the remaining. Since we are interested in
only two categories - “Car” and “Non-Car” Balancing the target variable
is not required as both of these categories are over 10%

``` r
data.cars2$Transport = as.character(data.cars2$Transport)

data.cars2$Transport[data.cars2$Transport=="Public Transport"] = "0"
data.cars2$Transport[data.cars2$Transport=="2Wheeler"] = "0"
data.cars2$Transport[data.cars2$Transport=="Car"] = "1"


data.cars2$Transport = as.factor(data.cars2$Transport)
```

``` r
cor(data.cars2[, c(1, 5, 6, 7)])
```

    ##                Age  Work.Exp    Salary  Distance
    ## Age      1.0000000 0.9322510 0.8607652 0.3530563
    ## Work.Exp 0.9322510 1.0000000 0.9320081 0.3727857
    ## Salary   0.8607652 0.9320081 1.0000000 0.4422379
    ## Distance 0.3530563 0.3727857 0.4422379 1.0000000

``` r
library(reshape2)
library(ggplot2)
qplot(x = Var1, y = Var2,
      data = melt(cor(data.cars2[, c(1, 5, 6, 7)])),
      fill = value,
      geom = "tile")
```

![](Multiple-Classification-Techniques_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
View(data.cars2)

ggplot(data.cars2, aes(x = Transport, y = Salary)) +geom_boxplot()
```

![](Multiple-Classification-Techniques_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

``` r
ggplot(data.cars2, aes(x = Transport, y = Age)) +geom_boxplot()
```

![](Multiple-Classification-Techniques_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
library(caret)
```

    ## Loading required package: lattice

``` r
?createDataPartition
```

    ## starting httpd help server ...

    ##  done

``` r
part = createDataPartition(data.cars2$Transport, p = 0.7, list = FALSE)

train.cars3 = data.cars2[part, ]
test.cars3 = data.cars2[-part, ]
```

``` r
##############Creating duplicate train and test sets - one for each model

train.cars4 = train.cars3
train.cars5 = train.cars3
train.cars6 = train.cars3
train.cars7 = train.cars3

test.cars4 = test.cars3
test.cars5 = test.cars3
test.cars6 = test.cars3
test.cars7 = test.cars3
```

``` r
##############fitting logistic regression model

logit = glm(Transport~., data = train.cars3, family=binomial)

logit
```

    ## 
    ## Call:  glm(formula = Transport ~ ., family = binomial, data = train.cars3)
    ## 
    ## Coefficients:
    ## (Intercept)          Age   GenderMale    Engineer1         MBA1     Work.Exp  
    ##    -71.6291       2.2910      -2.2832      -0.1229      -3.0311      -1.2776  
    ##      Salary     Distance     license1  
    ##      0.2825       0.4946       2.3215  
    ## 
    ## Degrees of Freedom: 310 Total (i.e. Null);  302 Residual
    ## Null Deviance:       249.9 
    ## Residual Deviance: 43.03     AIC: 61.03

``` r
##checking overall validity of the model

library(lmtest)
```

    ## Warning: package 'lmtest' was built under R version 3.6.2

    ## Loading required package: zoo

    ## 
    ## Attaching package: 'zoo'

    ## The following objects are masked from 'package:base':
    ## 
    ##     as.Date, as.Date.numeric

``` r
lrtest(logit)
```

    ## Likelihood ratio test
    ## 
    ## Model 1: Transport ~ Age + Gender + Engineer + MBA + Work.Exp + Salary + 
    ##     Distance + license
    ## Model 2: Transport ~ 1
    ##   #Df   LogLik Df  Chisq Pr(>Chisq)    
    ## 1   9  -21.515                         
    ## 2   1 -124.959 -8 206.89  < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
summary(logit)
```

    ## 
    ## Call:
    ## glm(formula = Transport ~ ., family = binomial, data = train.cars3)
    ## 
    ## Deviance Residuals: 
    ##      Min        1Q    Median        3Q       Max  
    ## -1.85038  -0.03646  -0.00607  -0.00037   2.54516  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -71.6291    18.7109  -3.828 0.000129 ***
    ## Age           2.2910     0.6311   3.630 0.000283 ***
    ## GenderMale   -2.2832     1.0496  -2.175 0.029612 *  
    ## Engineer1    -0.1229     1.1459  -0.107 0.914603    
    ## MBA1         -3.0311     1.2298  -2.465 0.013711 *  
    ## Work.Exp     -1.2776     0.4505  -2.836 0.004566 ** 
    ## Salary        0.2825     0.1108   2.549 0.010812 *  
    ## Distance      0.4946     0.1701   2.908 0.003637 ** 
    ## license1      2.3215     1.0776   2.154 0.031210 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 249.92  on 310  degrees of freedom
    ## Residual deviance:  43.03  on 302  degrees of freedom
    ## AIC: 61.03
    ## 
    ## Number of Fisher Scoring iterations: 10

``` r
library(car)
```

    ## Loading required package: carData

``` r
vif(logit)
```

    ##       Age    Gender  Engineer       MBA  Work.Exp    Salary  Distance   license 
    ## 10.581237  1.752915  1.106333  1.919395 17.473778  5.714734  1.641745  1.770396

``` r
##########removing variables with p value > o.05

train.cars31 = train.cars3[, -c(3, 4)]

View(train.cars31)
```

``` r
##########fitting the model

logit = glm(Transport~., data = train.cars31, family=binomial)

summary(logit)
```

    ## 
    ## Call:
    ## glm(formula = Transport ~ ., family = binomial, data = train.cars31)
    ## 
    ## Deviance Residuals: 
    ##      Min        1Q    Median        3Q       Max  
    ## -2.04632  -0.06989  -0.01407  -0.00197   1.77682  
    ## 
    ## Coefficients:
    ##              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -61.75992   14.28249  -4.324 1.53e-05 ***
    ## Age           1.99389    0.48987   4.070 4.70e-05 ***
    ## GenderMale   -2.16205    0.94372  -2.291  0.02196 *  
    ## Work.Exp     -1.18217    0.36835  -3.209  0.00133 ** 
    ## Salary        0.21526    0.08785   2.450  0.01427 *  
    ## Distance      0.42979    0.14446   2.975  0.00293 ** 
    ## license1      2.03550    0.90826   2.241  0.02502 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 249.919  on 310  degrees of freedom
    ## Residual deviance:  51.113  on 304  degrees of freedom
    ## AIC: 65.113
    ## 
    ## Number of Fisher Scoring iterations: 9

``` r
##########checking vif of remaining variables

vif(logit)
```

    ##       Age    Gender  Work.Exp    Salary  Distance   license 
    ##  9.086122  1.605370 15.296503  4.320858  1.435882  1.508352

``` r
##########removing multi collinear variables

View(train.cars31)

train.cars32 = train.cars31[, -c(1, 2, 3)]

View(train.cars32)
```

``` r
##########fitting the model again

logit = glm(Transport~., data = train.cars32, family=binomial)

summary(logit)
```

    ## 
    ## Call:
    ## glm(formula = Transport ~ ., family = binomial, data = train.cars32)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.0066  -0.2618  -0.1641  -0.1027   2.7630  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -9.01261    1.41659  -6.362 1.99e-10 ***
    ## Salary       0.16220    0.02841   5.710 1.13e-08 ***
    ## Distance     0.27214    0.09719   2.800  0.00511 ** 
    ## license1     1.37036    0.61707   2.221  0.02637 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 249.919  on 310  degrees of freedom
    ## Residual deviance:  94.895  on 307  degrees of freedom
    ## AIC: 102.89
    ## 
    ## Number of Fisher Scoring iterations: 7

``` r
##########finally checking vif

vif(logit)
```

    ##   Salary Distance  license 
    ## 1.073298 1.011571 1.083362

``` r
##checking pseudo R square

library(pscl)
```

    ## Warning: package 'pscl' was built under R version 3.6.2

    ## Classes and Methods for R developed in the
    ## Political Science Computational Laboratory
    ## Department of Political Science
    ## Stanford University
    ## Simon Jackman
    ## hurdle and zeroinfl functions by Achim Zeileis

``` r
pR2(logit)
```

    ##          llh      llhNull           G2     McFadden         r2ML         r2CU 
    ##  -47.4474991 -124.9594799  155.0239616    0.6202969    0.3925402    0.7107569

``` r
###########preparing the test dataset

View(test.cars3)

test.cars32 = test.cars3[, -c(1, 2, 3, 4, 5)]

View(test.cars32)
```

``` r
##predicting the train and test data set

predicted.logit = predict(logit, train.cars32, type = "response")

predicted.logit = floor(predicted.logit+0.5)

predicted.logitest = predict(logit, test.cars32, type = "response")

predicted.logitest = floor(predicted.logitest+0.5)
```

``` r
##############preparing confusion matrix

predicted.logit = as.factor(predicted.logit)

predicted.logitest = as.factor(predicted.logitest)


confusionMatrix(predicted.logit, train.cars32$Transport, positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 265  11
    ##          1   3  32
    ##                                           
    ##                Accuracy : 0.955           
    ##                  95% CI : (0.9256, 0.9752)
    ##     No Information Rate : 0.8617          
    ##     P-Value [Acc > NIR] : 5.633e-08       
    ##                                           
    ##                   Kappa : 0.7951          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.06137         
    ##                                           
    ##             Sensitivity : 0.7442          
    ##             Specificity : 0.9888          
    ##          Pos Pred Value : 0.9143          
    ##          Neg Pred Value : 0.9601          
    ##              Prevalence : 0.1383          
    ##          Detection Rate : 0.1029          
    ##    Detection Prevalence : 0.1125          
    ##       Balanced Accuracy : 0.8665          
    ##                                           
    ##        'Positive' Class : 1               
    ## 

``` r
confusionMatrix(predicted.logitest, test.cars32$Transport, positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 112   5
    ##          1   2  13
    ##                                           
    ##                Accuracy : 0.947           
    ##                  95% CI : (0.8938, 0.9784)
    ##     No Information Rate : 0.8636          
    ##     P-Value [Acc > NIR] : 0.001692        
    ##                                           
    ##                   Kappa : 0.7579          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.449692        
    ##                                           
    ##             Sensitivity : 0.72222         
    ##             Specificity : 0.98246         
    ##          Pos Pred Value : 0.86667         
    ##          Neg Pred Value : 0.95726         
    ##              Prevalence : 0.13636         
    ##          Detection Rate : 0.09848         
    ##    Detection Prevalence : 0.11364         
    ##       Balanced Accuracy : 0.85234         
    ##                                           
    ##        'Positive' Class : 1               
    ## 

\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#Model 2 - Naive Bayes

``` r
library(e1071)
```

    ## Warning: package 'e1071' was built under R version 3.6.2

``` r
nb=naiveBayes(Transport~., data=train.cars4)

nb
```

    ## 
    ## Naive Bayes Classifier for Discrete Predictors
    ## 
    ## Call:
    ## naiveBayes.default(x = X, y = Y, laplace = laplace)
    ## 
    ## A-priori probabilities:
    ## Y
    ##         0         1 
    ## 0.8617363 0.1382637 
    ## 
    ## Conditional probabilities:
    ##    Age
    ## Y       [,1]     [,2]
    ##   0 26.51866 3.076373
    ##   1 35.25581 3.259368
    ## 
    ##    Gender
    ## Y      Female      Male
    ##   0 0.2985075 0.7014925
    ##   1 0.2558140 0.7441860
    ## 
    ##    Engineer
    ## Y           0         1
    ##   0 0.2723881 0.7276119
    ##   1 0.1627907 0.8372093
    ## 
    ##    MBA
    ## Y           0         1
    ##   0 0.7276119 0.2723881
    ##   1 0.8372093 0.1627907
    ## 
    ##    Work.Exp
    ## Y        [,1]     [,2]
    ##   0  4.798507 3.202948
    ##   1 15.069767 4.822543
    ## 
    ##    Salary
    ## Y       [,1]      [,2]
    ##   0 12.96269  5.029861
    ##   1 34.82093 11.844962
    ## 
    ##    Distance
    ## Y       [,1]     [,2]
    ##   0 10.59179 3.105463
    ##   1 15.19302 3.653139
    ## 
    ##    license
    ## Y           0         1
    ##   0 0.8917910 0.1082090
    ##   1 0.2790698 0.7209302

``` r
## Predicting the training and test data

nb_train_pred = predict(nb,train.cars4)

nb_test_pred = predict(nb,test.cars4)
```

``` r
### Confusion matrix on the train data

confusionMatrix(nb_train_pred,train.cars4$Transport, positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 259   9
    ##          1   9  34
    ##                                           
    ##                Accuracy : 0.9421          
    ##                  95% CI : (0.9101, 0.9653)
    ##     No Information Rate : 0.8617          
    ##     P-Value [Acc > NIR] : 4.396e-06       
    ##                                           
    ##                   Kappa : 0.7571          
    ##                                           
    ##  Mcnemar's Test P-Value : 1               
    ##                                           
    ##             Sensitivity : 0.7907          
    ##             Specificity : 0.9664          
    ##          Pos Pred Value : 0.7907          
    ##          Neg Pred Value : 0.9664          
    ##              Prevalence : 0.1383          
    ##          Detection Rate : 0.1093          
    ##    Detection Prevalence : 0.1383          
    ##       Balanced Accuracy : 0.8786          
    ##                                           
    ##        'Positive' Class : 1               
    ## 

``` r
### Confusion matrix on the test data

confusionMatrix(nb_test_pred,test.cars4$Transport, positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 110   3
    ##          1   4  15
    ##                                           
    ##                Accuracy : 0.947           
    ##                  95% CI : (0.8938, 0.9784)
    ##     No Information Rate : 0.8636          
    ##     P-Value [Acc > NIR] : 0.001692        
    ##                                           
    ##                   Kappa : 0.78            
    ##                                           
    ##  Mcnemar's Test P-Value : 1.000000        
    ##                                           
    ##             Sensitivity : 0.8333          
    ##             Specificity : 0.9649          
    ##          Pos Pred Value : 0.7895          
    ##          Neg Pred Value : 0.9735          
    ##              Prevalence : 0.1364          
    ##          Detection Rate : 0.1136          
    ##    Detection Prevalence : 0.1439          
    ##       Balanced Accuracy : 0.8991          
    ##                                           
    ##        'Positive' Class : 1               
    ## 

\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#Model 3 - KNN

``` r
train.cars4[, c(1, 5, 6, 7)] = scale(train.cars4[, c(1, 5, 6, 7)])

test.cars4[, c(1, 5, 6, 7)] = scale(test.cars4[, c(1, 5, 6, 7)])
```

``` r
train.cars4$Gender = as.character(train.cars4$Gender)

train.cars4$Gender[train.cars4$Gender=="Male"]  = 1

train.cars4$Gender[train.cars4$Gender=="Female"]  = 0

test.cars4$Gender = as.character(test.cars4$Gender)

test.cars4$Gender[test.cars4$Gender=="Male"]  = 1

test.cars4$Gender[test.cars4$Gender=="Female"]  = 0
```

``` r
?knn

library(DMwR)
```

    ## Warning: package 'DMwR' was built under R version 3.6.3

    ## Loading required package: grid

    ## Registered S3 method overwritten by 'xts':
    ##   method     from
    ##   as.zoo.xts zoo

    ## Registered S3 method overwritten by 'quantmod':
    ##   method            from
    ##   as.zoo.data.frame zoo

``` r
KNN = kNN(Transport~., train.cars4, test.cars4[, -9], norm = FALSE, k = 4)

KNN2 = kNN(Transport~., train.cars4, train.cars4[, -9], norm = FALSE, k = 4)
```

``` r
##############confusion matrix 

confusionMatrix(KNN, test.cars4$Transport, positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 111   3
    ##          1   3  15
    ##                                           
    ##                Accuracy : 0.9545          
    ##                  95% CI : (0.9037, 0.9831)
    ##     No Information Rate : 0.8636          
    ##     P-Value [Acc > NIR] : 0.0005559       
    ##                                           
    ##                   Kappa : 0.807           
    ##                                           
    ##  Mcnemar's Test P-Value : 1.0000000       
    ##                                           
    ##             Sensitivity : 0.8333          
    ##             Specificity : 0.9737          
    ##          Pos Pred Value : 0.8333          
    ##          Neg Pred Value : 0.9737          
    ##              Prevalence : 0.1364          
    ##          Detection Rate : 0.1136          
    ##    Detection Prevalence : 0.1364          
    ##       Balanced Accuracy : 0.9035          
    ##                                           
    ##        'Positive' Class : 1               
    ## 

``` r
##############confusion matrix train

confusionMatrix(KNN2, train.cars4$Transport, positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 268   6
    ##          1   0  37
    ##                                           
    ##                Accuracy : 0.9807          
    ##                  95% CI : (0.9585, 0.9929)
    ##     No Information Rate : 0.8617          
    ##     P-Value [Acc > NIR] : 1.849e-13       
    ##                                           
    ##                   Kappa : 0.914           
    ##                                           
    ##  Mcnemar's Test P-Value : 0.04123         
    ##                                           
    ##             Sensitivity : 0.8605          
    ##             Specificity : 1.0000          
    ##          Pos Pred Value : 1.0000          
    ##          Neg Pred Value : 0.9781          
    ##              Prevalence : 0.1383          
    ##          Detection Rate : 0.1190          
    ##    Detection Prevalence : 0.1190          
    ##       Balanced Accuracy : 0.9302          
    ##                                           
    ##        'Positive' Class : 1               
    ## 

\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#Model 3 - Baggging

``` r
library(caret) 

?trainControl

train.c = trainControl(method = "repeatedcv", number = 3)
               
modelrf = train(Transport ~., data = train.cars5, method = "rf", 
              trControl = train.c, tuneLength = 5)
```

``` r
predrf1 = predict(modelrf, train.cars5)
predrf2 = predict(modelrf, test.cars5)
```

``` r
##############confusion matrix train

confusionMatrix(predrf1, train.cars5$Transport, positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 268   0
    ##          1   0  43
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9882, 1)
    ##     No Information Rate : 0.8617     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##                                      
    ##  Mcnemar's Test P-Value : NA         
    ##                                      
    ##             Sensitivity : 1.0000     
    ##             Specificity : 1.0000     
    ##          Pos Pred Value : 1.0000     
    ##          Neg Pred Value : 1.0000     
    ##              Prevalence : 0.1383     
    ##          Detection Rate : 0.1383     
    ##    Detection Prevalence : 0.1383     
    ##       Balanced Accuracy : 1.0000     
    ##                                      
    ##        'Positive' Class : 1          
    ## 

``` r
##############confusion matrix test

confusionMatrix(predrf2, test.cars5$Transport, positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 111   0
    ##          1   3  18
    ##                                          
    ##                Accuracy : 0.9773         
    ##                  95% CI : (0.935, 0.9953)
    ##     No Information Rate : 0.8636         
    ##     P-Value [Acc > NIR] : 6.749e-06      
    ##                                          
    ##                   Kappa : 0.9098         
    ##                                          
    ##  Mcnemar's Test P-Value : 0.2482         
    ##                                          
    ##             Sensitivity : 1.0000         
    ##             Specificity : 0.9737         
    ##          Pos Pred Value : 0.8571         
    ##          Neg Pred Value : 1.0000         
    ##              Prevalence : 0.1364         
    ##          Detection Rate : 0.1364         
    ##    Detection Prevalence : 0.1591         
    ##       Balanced Accuracy : 0.9868         
    ##                                          
    ##        'Positive' Class : 1              
    ## 

\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#Model 4 - Boosting

``` r
train.x = trainControl(method = "repeatedcv", number = 3)


model.x = train(Transport ~., data = train.cars6, method = "xgbTree", 
              trControl = train.x, tuneLength = 5)
```

``` r
predrx1 = predict(model.x, train.cars6)
predrx2 = predict(model.x, test.cars6)
```

``` r
##################confusion matrix

confusionMatrix(predrx1, train.cars6$Transport, positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 267   1
    ##          1   1  42
    ##                                          
    ##                Accuracy : 0.9936         
    ##                  95% CI : (0.977, 0.9992)
    ##     No Information Rate : 0.8617         
    ##     P-Value [Acc > NIR] : <2e-16         
    ##                                          
    ##                   Kappa : 0.973          
    ##                                          
    ##  Mcnemar's Test P-Value : 1              
    ##                                          
    ##             Sensitivity : 0.9767         
    ##             Specificity : 0.9963         
    ##          Pos Pred Value : 0.9767         
    ##          Neg Pred Value : 0.9963         
    ##              Prevalence : 0.1383         
    ##          Detection Rate : 0.1350         
    ##    Detection Prevalence : 0.1383         
    ##       Balanced Accuracy : 0.9865         
    ##                                          
    ##        'Positive' Class : 1              
    ## 

``` r
##################confusion matrix

confusionMatrix(predrx2, test.cars6$Transport, positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 112   1
    ##          1   2  17
    ##                                          
    ##                Accuracy : 0.9773         
    ##                  95% CI : (0.935, 0.9953)
    ##     No Information Rate : 0.8636         
    ##     P-Value [Acc > NIR] : 6.749e-06      
    ##                                          
    ##                   Kappa : 0.9057         
    ##                                          
    ##  Mcnemar's Test P-Value : 1              
    ##                                          
    ##             Sensitivity : 0.9444         
    ##             Specificity : 0.9825         
    ##          Pos Pred Value : 0.8947         
    ##          Neg Pred Value : 0.9912         
    ##              Prevalence : 0.1364         
    ##          Detection Rate : 0.1288         
    ##    Detection Prevalence : 0.1439         
    ##       Balanced Accuracy : 0.9635         
    ##                                          
    ##        'Positive' Class : 1              
    ##
