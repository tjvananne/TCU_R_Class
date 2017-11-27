

# setup -------------------------------------------

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

getwd()             # print current working directory
list.files()        # list files of current working directory
list.files('../')   # list files of one directory up from current working directory


# dependencies
library(ggplot2)       # visualization tool
library(ggthemes)      # theme set for ggplot2
library(viridis)       # favorite color scale for continuous scales in ggplot2
library(MASS)          # utility package, using this for parallel coordinate plots
library(dplyr)         # SQL-style data manipulation
library(tidyr)         # data reshaping (using this for long-to-wide transformations)
library(Hmisc)         # using for cut2() function -- better binning system for continuous variables
#library(lubridate)    # best package for working with dates -- this is not needed for this script
library(caret)         # useful utilities for modeling
library(randomForest)  # randome forest modeling
library(xgboost)       # gradient boosted trees for modeling
library(assertthat)    # assertion for writing quick tests and validations



# if this file hasn't been cached yet, then download from source:
if(!file.exists("../cache/raw_data.rds")) {
    # download data straight from the source
    raw_data <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", sep=",", strip.white = T, stringsAsFactors = F)
    
    names(raw_data) <- c("age", "workclass", "fnlwgt", "education", "educ_numb", "mar_status", "occupation", "relationship", "race",
                         "gender", "cap_gain", "cap_loss", "hr_pr_wk", "country", "income_category")
    # cache it
    saveRDS(raw_data, file="../cache/raw_data.rds")
    write.csv(raw_data, file="../cache/raw_data.csv", row.names = F)
    
} else {
    # read it from cache
    raw_data <- readRDS(file="../cache/raw_data.rds")
}



# raw_data <- data.frame(lapply(raw_data, trimws), stringsAsFactors = F)
raw_data$target <- factor(ifelse(raw_data$income_category == "<=50K", 0, 1), levels=c(0, 1))
raw_data$income <- factor(ifelse(raw_data$income_category == "<=50K", "Under 50k", "Above 50k"), levels=c("Under 50k", "Above 50k"))
raw_data$target_num <- as.integer(ifelse(raw_data$income_category == "<=50K", 0, 1)) 
raw_data$income_category <- NULL





# color palette for exploratory data analysis
col_prim <- "#D45743"  # reddish color for above 50k income
col_seco <- "#3A8AE0"  # blueish color for below 50k income


# exploratory analysis ----------------------------------

# Education Years vs Income Category
# Finding first/third quartiles where target is FALSE and where target is TRUE
under_q1 <- as.numeric(quantile(raw_data$educ_numb[raw_data$target_num == 0], 0.25))
under_q3 <- as.numeric(quantile(raw_data$educ_numb[raw_data$target_num == 0], 0.75))
above_q1 <- as.numeric(quantile(raw_data$educ_numb[raw_data$target_num == 1], 0.25))
above_q3 <- as.numeric(quantile(raw_data$educ_numb[raw_data$target_num == 1], 0.75))

# Combine the data that is above or below the third and first quartile respectively
raw_data_sub <- rbind(
    filter(raw_data, target_num == 0) %>% filter(educ_numb > under_q3 | educ_numb < under_q1),
    filter(raw_data, target_num == 1) %>% filter(educ_numb > above_q3 | educ_numb < above_q1))

ggplot(data=raw_data, aes(x=income, y=educ_numb)) +
    geom_jitter(data=raw_data_sub, alpha=0.2, color='black') +
    geom_boxplot(data=raw_data, aes(fill=ifelse(target_num==0, col_prim, col_seco))) + #  col_prim, outlier.shape=NULL) +
    theme_hc() + 
    theme(legend.position="none") +
    ylab("Years of Education") + 
    xlab("Income Category (target variable)") + 
    ggtitle("Years of Education vs Income Category - Boxplot/Jitter")



# Parallel Coordinate Plot -- 2000 row sample at a time to not overwhelm graphing device
raw_data_sample <- raw_data[sample(1:nrow(raw_data), 2000), ]

parcoord(x = (raw_data_sample %>% dplyr::select(age, educ_numb, hr_pr_wk, target_num)), 
         col=ifelse(raw_data_sample$target_num==0, col_prim, col_seco),
         var.label=T, main="Parallel Coordinate Plot")





# heatmap: age grouped into deciles by years of education -- % high income
# heatmap -- % of <this> population that is high income

# cut the age field into deciles (10 equal frequency bins)
raw_data$age_decile <- cut2(raw_data$age, g=10)
raw_data$age_decile_char <- as.character(raw_data$age_decile)

raw_data_grp <- raw_data %>%
    group_by(age_decile, educ_numb) %>%
    dplyr::summarise(high_income = sum(target_num == 1),
                     low_income = sum(target_num == 0)) %>%
    ungroup() %>%
    dplyr::mutate(percent_high_income = (high_income / (high_income + low_income) * 100))

ggplot(data=raw_data_grp, aes(x=educ_numb, y=age_decile, fill=percent_high_income)) +
    geom_tile(color='White', size=0.1) +
    scale_fill_viridis(name="Percent High Income") +
    coord_equal() + 
    ylab("Age Deciles - Equal Frequency Bins") +
    xlab("Number of Years of Education") +
    ggtitle("Percentage of High Income Individuals within Age Decile / Education Years Groups")



# what is fnlwgt? final weight? lets plot it
head(raw_data)
ggplot(data=raw_data, aes(x=age, y=fnlwgt, color=income)) +
    geom_jitter(alpha=0.3) +
    theme_hc(base_size=16) +
    ggtitle("Scatter Plot - 'fnlwgt' by Age")
    
    
# density and histogram plots -- doesn't show count, just count relative to class/group
ggplot(data=raw_data, aes(x=fnlwgt, fill=income)) +
    geom_density(alpha=0.4) +
    theme_hc(base_size=16) +
    ggtitle("Density of 'fnlwgt' field by target variable class")


# density and histogram plots -- shows the counts
ggplot(data=raw_data, aes(x=fnlwgt, fill=income)) +
    geom_histogram(alpha=0.4, position='identity') +
    theme_hc(base_size=16) +
    ggtitle("Histogram of 'fnlwgt' field by target variable class")



# modeling (rf) --------------------------------------------


rm(list=ls()[grepl("^raw_data", ls())])  # remove items from environment that match my grep
rm(list=ls())                            # or you can remove everything in your environment
gc()                                     # manual garbage collection to free up memory

list.files('../cache') 

train_samp_size <- 0.8
all_data <- readRDS('../cache/raw_data.rds')
all_data$id <- 1:nrow(all_data)           # create an "id" field


# inspect and change classes of columns
sapply(all_data, class)
sapply(all_data[, sapply(all_data, class) == "character"], function(x) length(unique(x)))
all_data[, sapply(all_data, class) == "character"] <- lapply(all_data[, sapply(all_data, class) == "character"], as.factor)
sapply(all_data, class)


View(all_data)   # this is a sufficient format for working with randomForest models


# stratefied sampling -- split out a holdout set
set.seed(1776)  # for reproducibility
indx_holdout <- caret::createDataPartition(all_data$income_category, p=(1 - train_samp_size), list=F)
data_holdout <- all_data[indx_holdout, ]
data_train <- all_data[-indx_holdout, ]

    # assert that length of the intersection of these two dataset's "id" field is zero
    assert_that(length(intersect(data_holdout$id, data_train$id)) == 0)  


# cross validation:
rfcv <- randomForest::rfcv(trainx=data_train[, setdiff(names(data_train), c("id", "income_category"))],
                           trainy=data_train$income_category,
                           cv.fold=5)


    # cache -- takes a little while for this to run
    saveRDS(rfcv, '../cache/rfcv.rds')
    rfcv <- readRDS(file='../cache/rfcv.rds')
    


# training the model:
rf <- randomForest::randomForest(
    x=data_train[, setdiff(names(data_train), c("id", "income_category"))],
    y=data_train$income_category,
    importance=T,
    keep.forest=T)

    # cache -- takes a while for this to run
    saveRDS(rf, '../cache/rf.rds')
    rf <- readRDS('../cache/rf.rds')

    list.files('../cache')
    
# feature importance
plot(rf)
varImpPlot(rf)


# can't see anything from cap_gain AND cap_loss
ggplot(all_data, aes(x=cap_gain, y=cap_loss, color=income_category)) +
    geom_point(alpha=0.4)

# just stick to cap gain, and plot against a dummy index value
ggplot(all_data, aes(x=cap_gain, y=1:nrow(all_data), color=income_category)) +
    geom_point(alpha=0.4)
    

randomForest::getTree(rf, k=1, labelVar=TRUE)


# analysis of results
Y_holdout <- data_holdout$income_category
rf_preds <- predict(rf, data_holdout[, setdiff(names(data_holdout), c("id", "income_category"))])
caret::confusionMatrix(rf_preds, Y_holdout, positive=">50K")

rf_preds_prob <- predict(rf, data_holdout[, setdiff(names(data_holdout), c("id", "income_category"))], type='prob')
data_holdout$prob_prediction <- rf_preds_prob[, 2]
ggplot(data_holdout, aes(x=prob_prediction, fill=income_category)) +
    geom_density(alpha=0.4) +
    theme_hc(base_size=16) +
    xlab("Predicted Probability of High Income") + 
    ylab("Density") +
    ggtitle("Prediction Probability by Income Category")


ggplot(data_holdout, aes(x=prob_prediction, fill=income_category)) +
    geom_histogram(alpha=0.4, position='identity') +
    theme_hc(base_size=16) +
    xlab("Predicted Probability of High Income") +
    ggtitle("Prediction Probability by Income Category")



# modeling (rf) developing an ROC curve ------------------
cutoffs <- seq(0, 1, 0.01)
f1scores <- numeric(length(cutoffs))
TPR <- numeric(length(cutoffs))
FPR <- numeric(length(cutoffs))
f1scores_shuffled <- numeric(length(cutoffs))
MCC <- numeric(length(cutoffs))

yshuffle <- sample(data_holdout$income_category, size=length(data_holdout$income_category))
    assert_that(sum(data_holdout$income_category == ">50K") == sum(yshuffle == ">50K"))

data_holdout$income_category    

for(i in 1:length(cutoffs)) {
    
    # not shuffled
    predicted_pos <- as.integer(data_holdout$prob_prediction >= cutoffs[i])
    actual_pos <- as.integer(data_holdout$income_category == ">50K")
    total_predicted_pos <- sum(predicted_pos)
    total_actual_pos <- sum(actual_pos)
    total_actual_neg <- sum(actual_pos==0)
    
    TP <- as.numeric(sum(predicted_pos==1 & data_holdout$income_category==">50K"))
    FP <- as.numeric(sum(predicted_pos==1 & data_holdout$income_category=="<=50K"))
    TN <- as.numeric(sum(predicted_pos==0 & data_holdout$income_category=="<=50K"))
    FN <- as.numeric(sum(predicted_pos==0 & data_holdout$income_category==">50K"))
    MCC[i] <- ((TP * TN) - (FP * FN)) / 
        (sqrt( (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) ))
    
    precision <- TP / total_predicted_pos                
    recall <- TP / total_actual_pos
    TPR[i] <- recall
    FPR[i] <- FP / total_actual_neg 
    
    f1 <- 2 * ((precision * recall) / (precision + recall))
    f1scores[i] <- f1
    
    # shuffled
    # predicted_pos <- sum(data_holdout$prob_prediction >= cutoffs[i])
    actual_pos_shuffle <- as.integer(yshuffle == 1)
    TP <- sum(predicted_pos==1 & yshuffle==1)
    FP <- sum(predicted_pos==1 & yshuffle==0)
    precision <- TP / sum(predicted_pos)                
    recall <- TP / sum(actual_pos_shuffle)
    
    f1 <- 2 * ((precision * recall) / (precision + recall))
    f1scores_shuffled[i] <- f1
}




# MCC setup and plot
f1_MCC <- data.frame(cutoff=cutoffs, MCC = MCC)
ggplot(data=f1_MCC, aes(x=cutoff, y=MCC)) +
    theme_bw(base_size = 15) +
    geom_line(size=1.2, color='blue') +
    ggtitle("MCC") +
    scale_x_continuous(breaks = seq(0, 1, 0.1)) +
    geom_vline(xintercept = f1_MCC$cutoff[which.max(f1_MCC$MCC)], color='red') +
    geom_hline(yintercept = max(f1_MCC$MCC, na.rm=T), color='red')



best_MCC_cutoff <- f1_MCC$cutoff[which.max(f1_MCC$MCC)]
confusionMatrix(data = as.integer(data_holdout$prob_prediction >= best_MCC_cutoff),
                reference = ifelse(data_holdout$income_category == ">50K", '1', '0'), 
                positive='1')

# manual conf mat
confusionMatrix(data = as.integer(data_holdout$prob_prediction >= 0.35),
                reference = ifelse(data_holdout$income_category == ">50K", '1', '0'), 
                positive='1')



# calculate the holdout AUC:
this_roc <- pROC::roc(response=data_holdout$income_category,
                      predictor=data_holdout$prob_prediction)

this_auc <- round(pROC::auc(this_roc), 4)
print(this_roc)
print(this_auc)


# ROC setup and plot
ROC <- data.frame(cutoff=cutoffs, TPR=TPR, FPR=FPR)
ggplot(ROC, aes(x=FPR, y=TPR)) +
    theme_bw(base_size = 14) +
    geom_line(size=1.2, color='blue') +
    geom_line(color='gray', mapping = aes(x=cutoff, y=cutoff), size=1.2) +
    ggtitle(paste0("ROC (AUC = ", this_auc, ")" ))


# F1 setup
f1_results <- data.frame(cutoff=cutoffs, F1=f1scores, F1_shuffle=f1scores_shuffled)
best_f1 <- round(max(f1_results$F1, na.rm=T), 4)
best_f1_cutoff <- f1_results$cutoff[which.max(f1_results$F1)]
f1_results[is.na(f1_results)] <- 0   
f1_results <- gather(f1_results, metric, value, -cutoff)


ggplot(f1_results, aes(x=cutoff, y=value, color=metric)) +
    theme_bw(base_size = 15) +
    geom_line(size=1.2) +
    ggtitle(paste0("F1 Scores - Real and Shuffled (F1: ", best_f1, " - cutoff: ", best_f1_cutoff, ")")) +
    geom_vline(xintercept = best_f1_cutoff, color='red') +
    geom_hline(yintercept = best_f1, color='red') +
    scale_x_continuous(breaks = seq(0, 1, 0.1))

# modeling (rf) where are we MOST wrong? 
data_holdout$income_num <- ifelse(data_holdout$income_category == ">50K", 1, 0)
data_holdout$residual <- data_holdout$income_num - data_holdout$prob_prediction



# modeling (xgb) --------------------------

rm(list=ls())
gc()


list.files('../cache')

train_samp_size <- 0.8
all_data <- readRDS('../cache/raw_data.rds')
all_data$id <- 1:nrow(all_data)           # create an "id" field

# inspect and change classes of columns
sapply(all_data, class)
all_data[, sapply(all_data, class) == "character"] <- lapply(all_data[, sapply(all_data, class) == "character"], as.factor)
sapply(all_data, class)



all_data$income_category <- ifelse(all_data$income_category == ">50K", 1, 0)



# convert to model matrix -- dummy variable creation (also known as one-hot-encoding)
all_data_onehot <- as.data.frame(model.matrix(~ . - 1, data=all_data), stringsAsFactors = F)


# stratefied sampling -- split out a holdout set
set.seed(1776)  # for reproducibility
indx_holdout <- caret::createDataPartition(all_data_onehot$income_category, p=(1 - train_samp_size), list=F)
data_holdout <- all_data_onehot[indx_holdout, ]
data_train <- all_data_onehot[-indx_holdout, ]

# assert that length of the intersection of these two dataset's "id" field is zero
assert_that(length(intersect(data_holdout$id, data_train$id)) == 0)  

features <- setdiff(names(data_train), c("id", "income_category"))

dmat_train   <- xgb.DMatrix(as.matrix(data_train[, features]), 
                            label=data_train$income_category)
dmat_holdout <- xgb.DMatrix(as.matrix(data_holdout[, features]))



params <- list(
    "objective" = "binary:logistic",
    "eval_metric" = 'auc',  # <- I took "error" out, data is too imbalanced for that
    "eta" = 0.1,
    "max_depth" = 5,
    "subsample" = 0.8,
    "colsample_bytree" = 0.5,
    "lambda" = 0,
    "alpha" = 1,
    "gamma" = 0,
    "max_delta_step" = 0,
    "scale_pos_weight" = 3,
    "nthread" = 1)


# run cross validation to determine the optimal number of rounds of boosting
set.seed(1776)
xgbcv <- xgboost::xgb.cv(
    data=dmat_train,
    params=params,
    nfold=5,
    nrounds=2000,
    print_every_n = 1,
    early_stopping_rounds = 30)

    # cache
    saveRDS(xgbcv, '../cache/xgbcv.rds')
    xgbcv <- readRDS('../cache/xgbcv.rds')
    
    
best_nrounds <- which.max(xgbcv$evaluation_log$test_auc_mean)


# now train an actual model with those optimal number of rounds
xgbmod <- xgboost::xgboost(
    data=dmat_train,
    params=params,
    nrounds=best_nrounds)

    # cache
    saveRDS(xgbmod, '../cache/xgbmod.rds')
    xgbmod <- readRDS('../cache/xgbmod.rds')

# probability predictions
xgb_preds <- predict(xgbmod, dmat_holdout)
data_holdout$preds <- xgb_preds

ggplot(data=data_holdout, aes(x=xgb_preds, fill=as.factor(income_category))) +
    geom_density(alpha=0.5) +
    theme_hc(base_size=16) +
    ggtitle("xgboost prediction vs actual")

ggplot(data=data_holdout, aes(x=xgb_preds, fill=as.factor(income_category))) +
    geom_histogram(alpha=0.5) +
    theme_hc(base_size=16) +
    ggtitle("xgboost prediction vs actual")


# pick a split point for calculating the confusion matrix
data_holdout$prediction <- ifelse(data_holdout$preds > 0.7, 1, 0)

caret::confusionMatrix(data=data_holdout$prediction,
                       reference=as.factor(data_holdout$income_category))





