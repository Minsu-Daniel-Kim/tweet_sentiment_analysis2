pkgs <- c("caret", "data.table","pROC", "dplyr", "mlbench", "lubridate","reshape2","tidyr","stringr","doMC","xgboost", "randomForest", "gbm", "kernlab", "h2o", "devtools", "e1071")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}

# library(devtools)
# install_github("h2oai/h2o-3/h2o-r/ensemble/h2oEnsemble-package")
# library(h2oEnsemble)
library(caret)
library(xgboost)
library(dplyr)
library(lubridate)
library(ggplot2)
library(doMC)
library(h2o)
library(pROC)
library(data.table)
library(tidyr)
library(mlbench)
library(e1071)
library(h2oEnsemble)
h2o.init(nthreads = -1) 
h2o.removeAll()
registerDoMC(8)

get_submission <- function(pred, threshold=0.5, version) {
  
  data.submission <- read.csv("rawData/SampleSubmission.csv")
  data.submission$y <- NULL
  
  result <- ifelse((pred > threshold), 1, 0)
  
  result <- cbind(data.submission, result)
  names(result) <- c("id", "y")
  write.csv(result, paste0("submission/submission_", version, ".csv"), row.names = F)
  
  return(NULL)
  
  
}

