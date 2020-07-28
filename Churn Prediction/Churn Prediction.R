

Cellphn.d = read.csv(file.choose(), header = T)
View(Cellphn.d)

##checking complete cases in dataset

sum(complete.cases(Cellphn.d))

##The dataset does not have any missing values 

summary(Cellphn.d)


##boxplot for univariate analysis

boxplot(Cellphn.d$AccountWeeks)
boxplot(Cellphn.d$DataUsage)
boxplot(Cellphn.d$DayMins)
boxplot(Cellphn.d$MonthlyCharge)

summary(Cellphn.d$DataUsage)

boxplot(Cellphn.d$DataUsage)


##scatter plot

plot(Cellphn.d$MonthlyCharge, Cellphn.d$DayMins)

plot(Cellphn.d$MonthlyCharge, Cellphn.d$DataUsage)

##histogram

hist(Cellphn.d$AccountWeeks)
hist(Cellphn.d$DataUsage)

##checking correlation

cor(Cellphn.d[, c(1:11)])


install.packages("ggpubr")

library("ggpubr")


cor(Cellphn.d[, c(2, 5, 7, 8, 9, 10,11)])


##Model building - dividing data into training and testing 70:30

set.seed(100)

indices = sample(1:nrow(Cellphn.d), .07*nrow(Cellphn.d))

test.data = Cellphn.d[indices, ]
train.data = Cellphn.d[-indices, ]

View(test.data)
View(train.data)


##Running logistic regression model and checking statistical significance of each independent variables:


logit = glm(Churn~AccountWeeks+ContractRenewal+DataPlan+DataUsage+CustServCalls+DayMins+DayCalls+MonthlyCharge+OverageFee+RoamMins, data = Cellphn.d)


##checking overall validity of the model

library(lmtest)
lrtest(logit)


##checking pseudo R square

pR2(logit)


##checking p values of variables

summary(logit)

##removing data usage and monthly charge variables and checking the p values of remaining variables

logit = glm(Churn~AccountWeeks+ContractRenewal+DataPlan+CustServCalls+DayMins+DayCalls+OverageFee+RoamMins, data = Cellphn.d)

summary(logit)

##removing accountweks variable and checking the p values of remaining variables

logit = glm(Churn~ContractRenewal+DataPlan+CustServCalls+DayMins+DayCalls+OverageFee+RoamMins, data = Cellphn.d)

summary(logit)

##removing dailycalls variable and checking the p values of remaining variables

logit = glm(Churn~ContractRenewal+DataPlan+CustServCalls+DayMins+OverageFee+RoamMins, data = Cellphn.d)

summary(logit)

##Checking multicollearity

library(car)
vif(logit)

##plotting residuals

plot(residuals(logit))

hist(residuals(logit))


##Checking pseudo R square

library(pscl)

pR2(logit)

##prediction using model

train.data$predict.Churn = predict(logit, train.data, type = "response")

train.data$predict.score = floor(train.data$predict.Churn+0.5)

View(train.data)


##Predicting test data

test.data$prediction = predict(logit, test.data, type = "response")
test.data$predict.score = floor(test.data$prediction+0.5)
View(test.data)

with(train.data, table(Churn, predict.Churn))


with(test.data, table(Churn, prediction))


table(Actual = train.data$Churn, Predicted = train.data$predict.score)
table(Actual = test.data$Churn, Predicted = test.data$predict.score)


##Area under curve

library(MLmetrics)

MLmetrics::AUC(train.data$predict.Churn, train.data$Churn)
MLmetrics::AUC(test.data$prediction,test.data$Churn)



