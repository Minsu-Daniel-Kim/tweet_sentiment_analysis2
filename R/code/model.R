################################# split data into train, test #################################

# split data into training and testing
split_data <- function(data, validation=FALSE) {
  set.seed(11)
  index <- createDataPartition(y = data$y, p = 0.9, list = F)
  
  data.train <- as.data.frame(apply(data, 2, as.numeric))
  train <- data.train[index, ]
  test <- data.train[-index, ]
  
  train.x <- as.matrix(train[, which(names(train) != 'y' )])
  train.y <- train[, which(names(train) == 'y' )]
  
  test.x <- as.matrix(test[, which(names(test) != 'y' )])
  test.y <- test[, which(names(test) == 'y' )]
  
  if (validation) {
    return(list(train_x = train.x, train_y = train.y, test_x = test.x, test_y = test.y))
  } else {
    train$y <- as.factor(train$y)
    test$y <- as.factor(test$y)
    
    data.train.h2o <- as.h2o(train)
    data.test.h2o <- as.h2o(test)
    
    return(list(train_x = train.x, train_y = train.y, test_x = test.x, test_y = test.y, train_h2o = data.train.h2o, test_h2o = data.test.h2o))
    
  }
  

}

# second layer aggregator
run_second_layer_xgb <- function(models, data_lst, train_y, version, depth = 11) {

  train.x <- as.matrix(models)
  train.y <- train_y
  
  dir.create(paste0("model/second_layer_xgboost_", version, "_depth_", depth))
  set.seed(11)
  nround.cv = 3200
  nfold = 15
  
  param <- list("objective" = "binary:logistic",    # multiclass classification 
                "eval_metric" = "error",    # evaluation metric 
                "nthread" = 36,   # number of threads to be used 
                "max_depth" = depth,    # maximum depth of tree 
                "eta" = 0.05,    # step size shrinkage 
                "gamma" = 0,    # minimum loss reduction 
                "subsample" = 0.5,    # part of data instances to grow tree 
                "colsample_bytree" = 0.6
  )
  bst_pre.cv <- xgb.cv(param=param, data= train.x, label= train.y, nfold= nfold, nrounds=nround.cv, prediction=TRUE, verbose=TRUE, maximize = TRUE)
  min.auc.idx = which.min(bst_pre.cv$dt[, test.error.mean]) 
  model.xgb <- xgboost(param=param, data= train.x, label= train.y, nfold= nfold, nrounds = min.auc.idx, verbose = TRUE, prediction = TRUE, maximize = TRUE)
  
  
  saveRDS(model.xgb, paste0("model/", "second_layer_xgboost_", version, "_depth_", depth, "/xgboost_", version, ".rds"))
  return(model.xgb)  
  
}

# run random forest
run_randomforest <- function(data, version) {
  features <- names(data)[names(data) != "y"]
  model.rf <- h2o.randomForest(x = features, y = "y", training_frame = data,
                               build_tree_one_node = FALSE, ntrees = 1000, max_depth = 20, nbins = 20)
  h2o.saveModel(model.rf, path = paste0("model/", "randomforest_", version))
  return(model.rf)
}

# get random forest model
get_randomforest <- function(filename, version) {
  model <- h2o.loadModel(paste0("model/randomforest_", version, "/", filename))
  return(model)
}

# run gradient boosting machine
run_gbm <- function(data, version) {
  features <- names(data)[names(data) != "y"]
  model.gbm <- h2o.gbm(x = features, y = "y", training_frame = data,
                       ntrees = 1000, max_depth = 20)
  h2o.saveModel(model.gbm, path = paste0("model/", "gbm_", version))
  return(model.gbm)
}
# get gradient boosting machine model
get_gbm <- function(filename, version) {
  model <- h2o.loadModel(paste0("model/gbm_", version, "/", filename))
  return(model)
}

# run xgboost
run_xgboost <- function(train_x, train_y, version) {
  dir.create(paste0("model/xgboost_", version))
  set.seed(11)
  nround.cv = 3200
  nfold = 15
  
  train.x <- train_x
  train.y <- train_y
  param <- list("objective" = "binary:logistic",    # multiclass classification 
                "eval_metric" = "auc",    # evaluation metric 
                "nthread" = 36,   # number of threads to be used 
                "max_depth" = 11,    # maximum depth of tree 
                "eta" = 0.05,    # step size shrinkage 
                "gamma" = 0,    # minimum loss reduction 
                "subsample" = 0.5,    # part of data instances to grow tree 
                "colsample_bytree" = 0.6
  )
  bst_pre.cv <- xgb.cv(param=param, data= train.x, label= train.y, nfold= nfold, nrounds=nround.cv, prediction=TRUE, verbose=TRUE, maximize = TRUE)
  min.auc.idx = which.max(bst_pre.cv$dt[, test.auc.mean]) 
  model.xgb <- xgboost(param=param, data= train.x, label= train.y, nfold= nfold, nrounds = min.auc.idx, verbose = TRUE, prediction = TRUE, maximize = TRUE)
  

  saveRDS(model.xgb, paste0("model/", "xgboost_", version, "/xgboost_", version, ".rds"))
  return(model.xgb)
}

# get xgboost
get_xgboost <- function(version) {
  model <- readRDS(paste0("model/xgboost_", version, "/xgboost_", version, ".rds"))
  return(model)
}

# run grid seasrch on xgboost
run_xgboost_grid <- function(train_x, train_y, report = paste0("report/report_", now())) {
  set.seed(111)
  nround.cv = 2000
  nfold = 11
  
  train.x <- train_x
  train.y <- train_y
  
  ################### grid search ############################
  searchGridSubCol2 <- expand.grid(eta = c(0.01), gamma = c(0), max_depth = c(9, 11, 13), subsample = c(0.5), colsample_bytree = c(0.6))
  
  apply(searchGridSubCol2, 1, function(parameterList){
    dir.create(paste0("model/grid_xgboost2_", version))

    #Extract Parameters to test
    currentEta <- parameterList[["eta"]]
    currentGamma <- parameterList[["gamma"]]
    currentDepth <- parameterList[["max_depth"]]
    currentSubsampleRate <- parameterList[["subsample"]]
    currentColsampleRate <- parameterList[["colsample_bytree"]]
    xgboostModelCV <- xgb.cv(data =  train.x, label= train.y, nrounds = nround.cv, nfold = nfold, 
                             "eval_metric" = "auc", "subsample" = currentSubsampleRate, "colsample_bytree" = currentColsampleRate,
                             "objective" = "binary:logistic", "max_depth" = currentDepth, "eta" =currentEta, "gamma" = currentGamma, "nthread" = 36)
    selectedrow = which.max(xgboostModelCV$test.auc.mean)
    selected <- max(xgboostModelCV$test.auc.mean)
    
    param <- list("objective" = "binary:logistic",    # multiclass classification 
                  "eval_metric" = "auc",    # evaluation metric 
                  "nthread" = 36,   # number of threads to be used 
                  "max_depth" = currentDepth,    # maximum depth of tree 
                  "eta" = currentEta,    # step size shrinkage 
                  "gamma" = currentGamma,    # minimum loss reduction 
                  "subsample" = 0.5,    # part of data instances to grow tree 
                  "colsample_bytree" = 0.6
    )
    
    model.xgb <- xgboost(param=param, data= train.x, label= train.y, nfold= nfold, nrounds = selectedrow, verbose = TRUE, prediction = TRUE, maximize = TRUE)
    saveRDS(model.xgb, paste0("model/", "grid_xgboost2_", version, "/xgboost_depth_", currentDepth, ".rds"))
    
    tmp <- c(selectedrow, selected, currentSubsampleRate, currentColsampleRate, currentGamma, currentEta, currentDepth)
    write(tmp,file=report,append=TRUE)
  })
}

# h2o version of random forest, gradient boosting, deep learning.

h2o.glm.1 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.2 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha)

h2o.randomForest.1 <- function(..., ntrees = 100, nbins = 50, seed = 12) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.randomForest.2 <- function(..., ntrees = 300, sample_rate = 0.75, seed = 15) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.3 <- function(..., ntrees = 500, sample_rate = 0.85, seed = 120) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.4 <- function(..., ntrees = 400, nbins = 50, balance_classes = TRUE, seed = 144) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, balance_classes = balance_classes, seed = seed)
h2o.randomForest.5 <- function(..., ntrees = 150, sample_rate = 0.75, seed = 15) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.6 <- function(..., ntrees = 150, sample_rate = 0.85, seed = 120) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.9 <- function(..., ntrees = 130, sample_rate = 0.5, seed = 124) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.7 <- function(..., ntrees = 150, nbins = 50, seed = 100) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.randomForest.8 <- function(..., ntrees = 150, nbins = 50, seed = 120) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)

h2o.gbm.1 <- function(..., ntrees = 300, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.gbm.2 <- function(..., ntrees = 500, nbins = 50, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.gbm.3 <- function(..., ntrees = 100, max_depth = 10, seed = 111) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.4 <- function(..., ntrees = 350, col_sample_rate = 0.8, seed = 12) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.5 <- function(..., ntrees = 150, col_sample_rate = 0.7, seed = 13) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.6 <- function(..., ntrees = 150, col_sample_rate = 0.6, seed = 14) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.7 <- function(..., ntrees = 100, balance_classes = TRUE, seed = 15) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, seed = seed)
h2o.gbm.8 <- function(..., ntrees = 100, max_depth = 3, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.deeplearning.1 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.2 <- function(..., hidden = c(200,200,200), activation = "Tanh", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.3 <- function(..., hidden = c(500,500), activation = "RectifierWithDropout", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.4 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, balance_classes = TRUE, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, balance_classes = balance_classes, seed = seed)
h2o.deeplearning.5 <- function(..., hidden = c(100,100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.6 <- function(..., hidden = c(50,50), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.7 <- function(..., hidden = c(100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)

# bundle of models
learner <- c(
  "h2o.deeplearning.5",
  "h2o.gbm.2",
  "h2o.randomForest.3"
)

# meta learner to aggregated models
metalearner <- "h2o.glm.wrapper"

run_ensemble <- function(data, version) {
  

  features <- names(data)[names(data) != "y"]
  fit.binary <- h2o.ensemble(x = features, y = "y", 
                             training_frame = data,
                             family = "binomial", 
                             learner = learner, 
                             metalearner = metalearner,
                             cvControl = list(V = 12))
  
  h2o.save_ensemble(fit.binary, paste0("model/", "ensemble_", version), force = TRUE)
  return(fit.binary)
}

get_ensemble <- function(version) {
  
  model <- h2o.load_ensemble(paste0("model/ensemble_", version))
  return(model)
}

get_ensemble_single <- function(filename, version) {
  
  model <- h2o.loadModel(paste0("model/ensemble_", version, "/", filename))
  return(model)
}
