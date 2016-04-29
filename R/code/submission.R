
get_submission <- function(pred) {
  
  data.submission <- read.csv("rawData/SampleSubmission.csv")
  data.submission$y <- NULL
  
  result <- cbind(data.submission, pred)
  names(result) <- c("id", "y")
  return(result)
  
}

