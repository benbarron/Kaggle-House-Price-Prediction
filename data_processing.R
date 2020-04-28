rm(list = ls())
setwd('~/Kaggle-House-Price-Prediction')

library(tidyr)
library(dplyr)
library(onehot)
library(stringr)

## using read.csv because read_csv doesn't have stringsAsFactors Option
train.data <- as_tibble(read.csv('./train.csv', stringsAsFactors = T, header = T))
pred.data <- as_tibble(read.csv('./test.csv', stringsAsFactors = T, header = T))

pred.data$SalePrice <- 0
pred.data$Prediction <- T
train.data$Prediction <- F

data <- rbind(train.data, pred.data)
data <- data[sample(nrow(data)),]

levels(data$Fence) = c(levels(data$Fence), 'NoFence')
levels(data$Alley) = c(levels(data$Alley), 'NoAlley')

data <- mutate(data,
               Alley = replace_na(Alley, 'NoAlley'),
               Fence = replace_na(Fence, 'NoFence'),
               GarageQual = replace_na(GarageQual, 'TA'),
               GarageCond = replace_na(GarageCond, 'TA'),
               GarageType = replace_na(GarageType, 'Attchd'),
               GarageYrBlt = replace_na(GarageYrBlt, 1980),
               GarageFinish = replace_na(GarageFinish, 'Unf'),
               FireplaceQu = replace_na(FireplaceQu, 'Gd'),
               BsmtFinType2 = replace_na(BsmtFinType2, 'Unf'),
               BsmtFinType1 = replace_na(BsmtFinType1, 'Unf'),
               BsmtExposure = replace_na(BsmtExposure, 'No'),
               MasVnrArea = replace_na(MasVnrArea, 103),
               MasVnrType = replace_na(MasVnrType, 'None'),
               LotFrontage = replace_na(LotFrontage, 70),
               BsmtCond = replace_na(BsmtCond, 'TA'),
               SaleType = replace_na(SaleType, 'WD'),
               GarageArea = replace_na(GarageArea, 472),
               Functional = replace_na(Functional, 'Typ'),
               GarageCars = replace_na(GarageCars, 2),
               BsmtHalfBath = replace_na(BsmtHalfBath, 0),
               BsmtFullBath = replace_na(BsmtFullBath, 0),
               BsmtFinSF1 = replace_na(BsmtFinSF1, 439),
               BsmtFinSF2 = replace_na(BsmtFinSF2, 52),
               BsmtUnfSF = replace_na(BsmtUnfSF, 554),
               TotalBsmtSF = replace_na(TotalBsmtSF, 1046),
               Exterior2nd = replace_na(Exterior2nd, 'VinylSd'),
               BsmtQual = replace_na(BsmtQual, 'Gd'),
               Exterior1st = replace_na(Exterior1st, 'VinylSd'),
               MSZoning = replace_na(MSZoning, 'RL'),
               Electrical = replace_na(Electrical, 'SBrkr'),
               KitchenQual = replace_na(KitchenQual, 'TA'))

data <- select(data, -c('MiscFeature', 'Utilities', 'PoolQC'))
##'BsmtExposure', 'BsmtFinType1','WoodDeckSF','OpenPorchSF','Fireplaces'

train.data <- filter(data, Prediction == F) %>% select(-c('Prediction'))
pred.data <- filter(data, Prediction == T) %>% select(-c('Prediction'))

set.seed(2345)
case.select <- sample(1:nrow(train.data), nrow(train.data)*0.9)
data.train <- train.data[case.select, ]
data.test <- train.data[-case.select, ]

## one hot encoded version of data
encoder <- onehot(data, max_levels = 30)
data.hot <- as_tibble(predict(encoder, data))
train.hot <- filter(data.hot, Prediction == F) %>% select(-c('Prediction'))
pred.hot <- filter(data.hot, Prediction == T) %>% select(-c('Prediction'))

ncols <- ncol(train.hot)
ncms <- ncols - 1
x.train <- as.matrix(train.hot[case.select, 2:ncms])
x.test <- as.matrix(train.hot[-case.select, 2:ncms])
y.train <- as.matrix(train.hot[case.select,  ncols])
y.test <- as.matrix(train.hot[-case.select, ncols])
pred.mat <- as.matrix(pred.hot[, 2:ncms])

cnames <- colnames(x.train)
num.cols <- which(is.na(str_locate(cnames, '=')[,2]))

m <- colMeans(x.train[,num.cols])
s <- apply(x.train[,num.cols], 2, sd)

x.test.scaled <- x.test
x.train.scaled <- x.train
pred.mat.scaled <- pred.mat

x.train.scaled[, num.cols] <- scale(x.train[, num.cols], center = m, scale = s)
x.test.scaled[, num.cols] <- scale(x.test[, num.cols], center = m, scale = s)
pred.mat.scaled[, num.cols] <- scale(pred.mat[, num.cols], center = m, scale = s)

save(x.train, x.train.scaled, x.test, x.test.scaled, y.train, y.test, data.train, 
     data.test, train.data, pred.data, pred.hot, pred.mat, train.hot, data.hot,
     pred.mat.scaled, file = './all_data.RData')

rm(list = ls())
