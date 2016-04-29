library(dplyr)
library(caret)
library(data.table)
data <- fread("~/Desktop/sentiment_data/full_x.csv", stringsAsFactors = F)
data_y <- fread("~/Desktop/sentiment_data/full_y.csv", nrows = 459721, stringsAsFactors = F)
names(data_y) <- 'y'

data.x <- head(data, 209329)
data.y <- head(data_y, 209329)

data.merged <- cbind(data, data.y)

data.test <- filter(data.merged, y == -1)
data.train <- filter(data.merged, y != -1)

saveRDS(data.test, "cleanedData/test_2000_features.rds")
saveRDS(data.train, "cleanedData/train.rds")

saveRDS(data.train[1:300000, ], "cleanedData/train_300000.rds")
saveRDS(data.train[300001:350000, ], "cleanedData/validation1.rds")
saveRDS(data.train[350001:nrow(data.train), ], "cleanedData/validation2.rds")

# clean.data <- filter(merged, y != -1)
# str(clean.data$y)
# table(clean.data$y)
# 
# train <- clean.data[1:100000, ]
# rest <- clean.data[100001:nrow(clean.data), ]
# 
# folds <- createFolds(rest$y, k = 17)
# 
# 
# write.csv(train, "cleanedData/train100000.csv", row.names = F)
# saveRDS("cleanedData/train100000.csv")
# 
# for (name in names(folds)) {
#   write.csv(rest[folds[[name]], ], paste0("cleanedData/", name, ".csv"), row.names = F)
# }
# 
# for (name in c("Fold15", "Fold16", "Fold17")) {
#   write.csv(rest[folds[[name]], ], paste0("~/Desktop/", name, ".csv"), row.names = F)
# }
