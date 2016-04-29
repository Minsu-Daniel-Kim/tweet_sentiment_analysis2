# import data
source("code/utils.R")
source("code/model.R")
source("code/evaluation.R")
# load("rawData/TrainTest.RData")

# load training data with 3000 features
data.train <- readRDS("cleanedData/train_300000.rds")

# load validation dataset to check prediction accuracy later
data.validation1 <- readRDS("cleanedData/validation1.rds")
data.validation2 <- readRDS("cleanedData/validation2.rds")

# version label
version <- 1

# subet data into sample to experiment
sample <- select(head(data.train, 200), V1:V20, y)
sample_validation1 <- select(head(data.validation, 200), V1:V20, y)
sample_validation1 <- select(head(data.validation, 200), V1:V20, y)

# split data into training, testing and aggregate them in a list
data_lst_validation1 <- split_data(sample_validation, validation = TRUE)
data_lst_validation2 <- split_data(sample_validation, validation = TRUE)

data_lst <- split_data(data.train)

# run random forest
model.rf <- run_randomforest(data_lst$train_h2o, version = version)

# run gradient boosting
model.gbm <- run_gbm(data_lst$train_h2o, version)

# run ensemble model
model.ens <- run_ensemble(data_lst$train_h2o, version)

# run xgboost
model.xgb <- run_xgboost(data_lst$train_x, data_lst$train_y, version)

# load models from cache
model.rf <- get_randomforest("DRF_model_R_1461545890217_299314", 1)
model.gbm <- get_gbm("GBM_model_R_1461545890217_299918", 1)
model.ens <- get_ensemble(1)
model.xgb <- get_xgboost(1)

# get predictions
version_layer_1 = paste0("layer1_", version)
model.rf.pred <- predict_rf(model.rf, data_lst$test_h2o, version_layer_1)
model.gbm.pred <- predict_gbm(model.gbm, data_lst$test_h2o, version_layer_1)
model.ensemble.pred <- predict_ens(model.ens, data_lst$test_h2o, version_layer_1)
model.xgboost.pred <- predict_xgboost(model.xgb, data_lst$test_x, version_layer_1)

# second layer data
models <- cbind(model.rf.pred, model.gbm.pred, model.ensemble.pred, model.xgboost.pred)
model.first.layer1 <- run_second_layer_xgb(models, data_lst, data_lst$test_y,version_layer_1)
model.first.layer2 <- run_second_layer_xgb(models, data_lst, data_lst$test_y,version_layer_1, depth = 7)
