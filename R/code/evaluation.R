
predict_second_layer <- function(model1, model2, data_lst){
  
  pred1 <- predict(model1, data_lst$train_x)
  pred2 <- predict(model2, data_lst$train_x)
  
  return(list(pred1 = pred1, pred2 = pred2))
  
}

get_accuracy <- function(pred, obs, threshold=0.5) {
  
  result <- ifelse((pred > threshold), 1, 0)
  sum(result == obs) / length(obs)
  
  
}

optimizer <- function(model1, model2, obs) {
  
  for (i in seq(0.3, 0.8, by = 0.01)) {
    
    result <- i * model1 + (1 - i) * model2
    
    for (j in seq(0.3, 0.8, by = 0.01)) {
      
      z <- ifelse((result > j), 1, 0)
      sum(z == obs) / length(obs)
      
    }
    
  }
  
}

predict_rf <- function(model, data, version) {
  
  model.pred <- predict(model, data)
  df <- as.data.frame(model.pred$p1)
  file <- paste0("pred/randomforest_", version, "/randomforest_pred_", version, ".rds")
  
  if (dir.exists(paste0("pred/randomforest_", version))) {
    if(file.exists(file)) {
      df <- readRDS(file)
    } else {
      saveRDS(df, file)
    }
  } else {
    dir.create(paste0("pred/randomforest_", version))
    saveRDS(df, file)
  }
  
  return(df)
}

predict_gbm <- function(model, data, version) {
  
  model.pred <- predict(model, data)
  df <- as.data.frame(model.pred$p1)
  file <- paste0("pred/gbm_", version, "/gbm_pred_", version, ".rds")
  
  if (dir.exists(paste0("pred/gbm_", version))) {
    if(file.exists(file)) {
      df <- readRDS(file)
    } else {
      saveRDS(df, file)
    }
  } else {
    dir.create(paste0("pred/gbm_", version))
    saveRDS(df, file)
  }

  return(df)
}

predict_ens <- function(model, data, version) {
  
  model.pred <- predict(model, data)
  df <- as.data.frame(model.pred$pred$p1)
  file <- paste0("pred/ensemble_", version, "/ensemble_pred_", version, ".rds")
  
  if (dir.exists(paste0("pred/ensemble_", version))) {
    if(file.exists(file)) {
      df <- readRDS(file)
    } else {
      saveRDS(df, file)
    }
  } else {
    dir.create(paste0("pred/ensemble_", version))
    saveRDS(df, file)
  }
  
  return(df)
  
}

predict_xgboost <- function(model, data, version, save = TRUE) {

  model.pred <- predict(model, data)
  df <- as.data.frame(model.pred)
  file <- paste0("pred/xgboost_", version, "/xgboost_pred_", version, ".rds")
  if (save) {
    if (dir.exists(paste0("pred/xgboost_", version))) {
      if(file.exists(file)) {
        df <- readRDS(file)
      } else {
        saveRDS(df, file)
      }
    } else {
      dir.create(paste0("pred/xgboost_", version))
      saveRDS(df, file)
    }
  }
  return(df)
}




# model.ensemble.pred <- predict(model.rf, data.test.h2o)
# saveRDS(model.ensemble.pred, "pred/model_gbm.rds")
# 
# model.ensemble.pred.df <- as.data.frame(model.ensemble.pred$pred$predict)
# sum(model.ensemble.pred.df$predict == test$y) / nrow(test)
# 
# predict_rf(model.gbm, )
# 
# predict_rf <- function(model, data) {
#   data.h2o <- as.h2o(data)
#   model.pred <- predict(model, data.h2o)
#   model.pred.df <- as.data.frame(model.pred$pred$predict)
#   saveRDS(model.pred.df, "pred/randomforest.rds")
#   return(model.pred.df)
# }
# 
# model.xgb.pred <- predict(model.xgb, test.x)
# result <- ifelse((model.xgb.pred > 0.53), 1, 0)
# sum(result == test.y) / length(test.y)
# 
# 
# data <- readRDS("cleanedData/test.rds")
# data$y <- NULL
# model.xgb.pred <- predict(model.xgb, as.matrix(data))
# result <- ifelse((model.xgb.pred > 0.53), 1, 0)
# 
# last <- cbind(data.submission, result)
# names(last) <- c("id", "y")
# write.csv(last, "submission/submission6.csv", row.names = F)
# 
# saveRDS("")
# 
# predict_xgb <- function(model, data) {
#   model.xgb.pred <- predict(model, data)
#   saveRDS(model.xgb.pred, "pred/xgboost.rds")
#   return(model.xgb.pred)
# }
# 
# get_accuracy_xgb <- function(pred, obs, threshold = 0.5) {
#   result <- ifelse((model.xgb.pred > threshold), 1, 0)
#   sum(result == obs) / length(obs)
# }
# 
# get_accuracy_rf <- function(model, data) {
#   
# }



# for (i in seq(from = 0.1, to = 1, by = 0.1)) {
#   w1 <- i
#   w2 <- 1 - i
#   result <- w1 * en + w2 * xg
#   a <- ifelse((result > 0.5), 1, 0)
#   result <- sum(a == test.y) / length(test.y)
#   print(paste0(w1, w2, result))
# }
# 
# 
# 
# model.ensemble.pred <- predict(fit4, data.test.h2o)
# b <- as.data.frame(model.ensemble.pred$pred$p1)$p1
# 
# validation <- as.matrix(fread("validation/Fold02.csv"))
# validation.x <- validation[,-1001]
# validation.y <- validation[,1001]
# 
# model1 <- readRDS("model/xgb_auc.RDS")
# model2 <- readRDS("model/xgb_error.RDS")
# validation.x <- apply(validation.x, 2, as.numeric)
# model.xgb.pred1 <- predict(model1, validation.x)
# model.xgb.pred2 <- predict(model2, validation.x)
# 
# V1 <- model.xgb.pred1
# V2 <- model.xgb.pred2
# y <- validation.y
# 
# df <- data.frame(V1, V2, y)
# df$y <- as.factor(df$y)
# 
# 
# m <- matrix(ncol = 3, nrow = 441)
# 
# x <- 1
# for (i in 0.5) {
# 
#   r <- apply(df, 1, function(x){
#       
#       i * as.numeric(x[1]) + as.numeric(x[2]) * (1 - i)
#       
#   })
#   for (j in 0.5) {
#     
#     result <- ifelse((r > j), 1, 0)
#     # print(c(i, j, sum(result == y) / length(y)))
#     m[x, ] <- c(i, j, sum(result == y) / length(y))
#     x <- x + 1
#   }
# }
# idx <- which.max(m[,3])
# m[idx, ]
# max(m[,3])
# 
# 
# 
# fit_ens <- function(data.x) {
#   
#   data.x <- apply(data.x, 2, as.numeric)
#   
#   model.xgb.pred1 <- predict(model1, data.x)
#   model.xgb.pred2 <- predict(model2, data.x)
#   
#   result <- ifelse((model.xgb.pred1 * 0.5 + model.xgb.pred2 * 0.5 > 0.5), 1, 0)
#   # sum(result == data.y) / length(data.y)
# }
# 
# s <- fit_ens(Xtest)
# 
# sum(s == validation.y) / length(validation.y)
# 
# data.submission <- cbind(data.submission, s)
# names(data.submission) <- c("id", "y")
# write.csv(data.submission, "submission/submission_ens2.csv", row.names = F)
