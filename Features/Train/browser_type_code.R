
library(caret)
library(xgboost)
library(magrittr)
library(jsonlite)
library(lubridate)
library(irlba)
library(knitr)
library(Rmisc)
library(scales)
library(tidyverse)
library(countrycode)
library(highcharter)
library(httr)
library(forcats)

device <- train$device
device1 <- as.data.frame(device)
#device1[[1]][[1]]
#str(device1[[1]][[1]], max.level = 1)
#dev1 <- device[[1]][[1]]
#levels(device1)
#dev1 <- fromJSON((device1, flatten = TRUE)
#dev1 <- str_c(device, collapse = ",") 
#is.list(dev1)
#typeof(dev1)
#device2 <- paste("[", 
#                   paste(train$device,
#                         collapse = ","),
#                   "]") %>% fromJSON(flatten = T)
#str(device1)
#glimpse(device1)
#library(listviewer)
#jsonedit(device1[[1]][[1]])
#for(i in 1:100) {
#  str(device1[[1]][[i]]) 
#}
#x <- fct_count(device1[[1]][[1]])
#x
#which(x$n == 1)



#******************************************************
#dev_shrt$browser_type <- as.character(NA)
#for(i in 1:20) {
# factors <- fct_count(dev_shrt[[1]][[i]])
# dev_shrt$browser_type[i] <- which(factors$n == 1)
#}
#dev_shrt$browser_type <- factor(dev_shrt$browser_type)
#******************************************************
browser_type <- as.character(NA)
for(i in 1:nrow(device1)) {
  factors <- fct_count(device1[[1]][[i]])
  browser_type[i] <- which(factors$n == 1)
}
browser_type <- data_frame(browser_type)
colnames(browser_type) <- "browser_type"
browser_type$id <- test$fullVisitorId
saveRDS(browser_type, "browser_type_tst")
#***************************************************
#***************************************************
#Now the same for test
device <- test$device
device1 <- as.data.frame(device)








