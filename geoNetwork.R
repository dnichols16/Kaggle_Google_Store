tr_geo <- train %>% 
  select(geoNetwork, fullVisitorId)
rm(geo)
tr_geo1 <- paste("[", paste(train$geoNetwork, collapse = ","), "]") %>% fromJSON(flatten = T)
names(tr_geo1)
table(tr_geo1$networkLocation)
tr_geo1$networkLocation <- NULL

tr_geo1$id <- tr_geo$fullVisitorId
#**********************************************

te_geo <- test %>%
  select(geoNetwork, fullVisitorId)
te_geo1 <- paste("[", paste(test$geoNetwork, collapse = ","), "]") %>% fromJSON(flatten = T)
names(te_geo1)
te_geo1 <- te_geo1 %>% 
  select(continent, subContinent, country,
         region, metro, city, cityId, networkDomain)
te_geo1$id <- te_geo$fullVisitorId
#***********************************************
table(tr_geo1$continent)
str(tr_geo1)
str(te_geo1)
tr_geo1 <- as.tibble(tr_geo1)
te_geo1 <- as.tibble(te_geo1)
#************************************************
sum(tr_geo1$region == "not available in demo dataset")
# cityID all not available
tr_geo1 <- tr_geo1[ , -7]
names(tr_geo1)
names(te_geo1)
te_geo1 <- te_geo1[ , -7]
train_all <- bind_cols(train_all, tr_geo1)
test_all <- bind_cols(test_all, te_geo1)
saveRDS(train_all, "train_all.rds")
saveRDS(test_all, "test_all")
#**********************************************
#remove socialEngagementType - no variation
train <- train[ , -7]
names(train)
names(test)
test <- test[ , -7]















