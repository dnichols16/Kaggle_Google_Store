library(tidyverse)
library(keras)
hepatitis <- read_csv("D:/Kaggle_Google_Store/Hepatitis/hepatitis.txt")
hep <- hepatitis
colnames(hep) <- c("class",
                   "age",
                   "sex",
                   "steroid",
                   "antivirals",
                   "fatigue",
                   "malaise",
                   "anorexia",
                   "bigliv",
                   "firmliv",
                   "splnpalp",
                   "spiders",
                   "ascites",
                   "varices",
                   "bili",
                   "alkphos",
                   "sgot",
                   "alb",
                   "ptt",
                   "histo")
glimpse(hep)
#*******************************************
zeroize <- function(x) {
  x <- as.numeric(x)
  case_when(
    x == 1 ~ 0,
    x == 2 ~ 1
  )
}

nas <- function(x) {
  y <- sum(is.na(x)) 
}
#*******************************************

ones_twos <- hep %>%
  select(
    class, sex, steroid, antivirals, fatigue, malaise,
    anorexia, bigliv, firmliv, splnpalp, spiders,
    ascites, varices, histo
  )

ones_twos <- as_data_frame(map(ones_twos, zeroize))
glimpse(ones_twos)

tabs <- map(ones_twos, table)
str(tabs)
yy <- map(tabs, pluck)
yy[[1]]


for(i in 1:length(yy)) {
  j = 1
  a <- yy[[i]][[j]]
  j = j + 1
  b <- yy[[i]][[j]]
  prop <- a/(a+b)
  print(names(yy[i]))
  print(prop)
  print("")
}

map_dbl(ones_twos, nas) 
#*******************************************

ones_twos$steroid <- ifelse(is.na(ones_twos$steroid), rbinom(1, 1, .49), ones_twos$steroid)
ones_twos$fatigue <- ifelse(is.na(ones_twos$fatigue), rbinom(1, 1, .49), ones_twos$fatigue)
ones_twos$malaise <- ifelse(is.na(ones_twos$malaise), rbinom(1, 1, .49), ones_twos$malaise)
ones_twos$anorexia <- ifelse(is.na(ones_twos$anorexia), rbinom(1, 1, .49), ones_twos$anorexia)
ones_twos$bigliv <- ifelse(is.na(ones_twos$bigliv), rbinom(1, 1, .49), ones_twos$bigliv)
ones_twos$firmliv <- ifelse(is.na(ones_twos$firmliv), rbinom(1, 1, .49), ones_twos$firmliv)
ones_twos$splnpalp <- ifelse(is.na(ones_twos$splnpalp), rbinom(1, 1, .49), ones_twos$splnpalp)
ones_twos$spiders <- ifelse(is.na(ones_twos$spiders), rbinom(1, 1, .49), ones_twos$spiders)
ones_twos$ascites <- ifelse(is.na(ones_twos$ascites), rbinom(1, 1, .49), ones_twos$ascites)
ones_twos$varices <- ifelse(is.na(ones_twos$varices), rbinom(1, 1, .49), ones_twos$varices)

sum(is.na(ones_twos))
#********************************************  
#********************************************
options(digits = 2)
bignums <- hep %>%
  select(bili, alkphos, sgot, alb, ptt)
glimpse(bignums)

makena <- function(x) {
 ifelse(x == "?", NA, x) 
}
bignums <- as.tibble(map(bignums, makena))

mnsd <- function(x) {
  mn <- mean(x, na.rm = TRUE)
  sd <- sd(x, na.rm = TRUE)
  return(list(mn, sd))
}
glimpse(bignums)
bignums <- bignums %>% 
  map(as.numeric) %>% 
  as_tibble()

bignums %>% 
  map(mnsd)

bignums$bili <- ifelse(is.na(bignums$bili), abs(rnorm(1, 1.43, 1.21)), bignums$bili)
bignums$alkphos <- ifelse(is.na(bignums$alkphos), abs(rnorm(1, 105, 52)), bignums$alkphos)
bignums$sgot <- ifelse(is.na(bignums$sgot), abs(rnorm(1, 86, 90)), bignums$sgot)
bignums$alb <- ifelse(is.na(bignums$alb), abs(rnorm(1, 3.8, .65)), bignums$alb)
bignums$ptt <- ifelse(is.na(bignums$ptt), abs(rnorm(1, 62, 23)), bignums$ptt)
sum(is.na(bignums))
#*************************************************
hep1 <- bind_cols(ones_twos, bignums)
hep2 <- as.matrix(hep1)
dim(hep2)
hep3 <- normalize(hep2)
#*************************************************
index <- sample(2,
                  nrow(hep3),
                  replace=TRUE,
                  prob=c(0.67, 0.33))
hep_train <- hep3[index == 1, 2:19]
hep_test <- hep3[index ==2, 2:19]

hep_train_target <- iris[index == 1, 1]
hep_test_target <- iris[index == 2, 1]



# One hot encode training target values
hep_train_labels <- to_categorical(hep_train_target)
hep_test_labels <- to_categorical(hep_test_target)

#*****************************************************















