
#######################################It is experimental. Feel free to skip it ################################

source("code/utils.R")
source("code/model.R")
source("code/evaluation.R")

version_layer_1 = paste0("layer1_", version)

data.chunk2 <- readRDS("cleanedData/validation2.rds")
data.chunk1 <- readRDS("cleanedData/validation1.rds")

data_lst1 <- split_data(data.chunk1, validation = TRUE)
data_lst2 <- split_data(data.chunk2, validation = TRUE)
model.xgb <- get_xgboost(1)

model.xgboost.pred <- predict_xgboost(model.xgb, data_lst1$test_x, version_layer_1, save = F)
a <- get_accuracy(model.xgboost.pred, data_lst1$test_y, threshold = 0.52)

model.xgboost.pred <- predict_xgboost(model.xgb, data_lst2$test_x, version_layer_1, save = F)
get_accuracy(model.xgboost.pred, data_lst2$test_y, threshold = 0.52)

data.test <- readRDS("cleanedData/test_2000_features.rds")

model.xgboost.pred <- predict_xgboost(model.xgb, as.matrix(data.test), version_layer_1, save = F)
a <- get_submission(model.xgboost.pred, 0.52, 7)

a <- get_accuracy(model.xgboost.pred, data_lst1$test_y, threshold = 0.52)

a <- read.csv("submission/submission_7.csv")

library(glmnet)

model.sparse.lr.cv <- cv.glmnet(x = data_lst$train_x, y = data_lst$train_y, alpha = 1, family = 'binomial')
model.sparse.lr <- glmnet(x = data_lst$train_x, y = data_lst$train_y, alpha = 1, family = 'binomial')


model.xgboost.pred <- predict_xgboost(model.xgb, rbind(data_lst1$test_x, data_lst1$train_x), version_layer_1, save = F)
model.sparse.lr.pred <- predict(model.sparse.lr, rbind(data_lst1$test_x, data_lst1$train_x) , s = model.sparse.lr.cv$lambda.min, type = 'response')

c <- NULL
for (i in seq(0.3, 0.9, by=0.01)) {

  c <- c(c, get_accuracy(model.xgboost.pred, data_lst$test_y, 0.52))
}
c <- matrix(nrow = 2601, ncol = 3)
z = 1
for (i in seq(0.4, 0.9, by=0.01)) {
  result <- i * model.xgboost.pred + (1 - i) * model.sparse.lr.pred
  for (j in seq(0.4, 0.6, by=0.01)) {
  c[z, ] <- c(i, j, get_accuracy(result, data_lst$test_y, threshold = j))
  z <- z + 1
  }
}
idx <- which.max(c[,3])
c[idx, ]

get_accuracy(model.xgboost.pred, c(data_lst1$test_y, data_lst1$train_y), threshold = 0.52)

get_accuracy(model.xgboost.pred * 0.81 + model.sparse.lr.pred * 0.19, c(data_lst1$test_y, data_lst1$train_y), threshold = 0.52)


a <- model.pred <- predict(model.ens, data_lst$test_h2o)
df <- as.data.frame(a$pred$predict)
sum(df$predict == as.data.frame(data_lst$test_h2o$y)$y) / nrow(data_lst$test_h2o)



xgb <- readRDS("model/xgboost_1/xgboost_1.rds")
logit_cv <- readRDS("model/sparse_logit/sparse_cv.rds")
logit <- readRDS("model/sparse_logit/sparse_logit.rds")
ens <- h2o.load_ensemble("model/ensemble_1")

xgb.pred <- predict_xgboost(xgb, data_lst$test_x, version = 1)
logit.pred <- predict(logit, data_lst$test_x , s = logit_cv$lambda.min, type = 'response')
predict_ens(model.ens, data_lst$test_h2o)









