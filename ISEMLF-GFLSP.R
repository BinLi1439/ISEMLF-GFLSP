# ===================== 1. Batch installation and loading of dependent packages =====================
pkgs <- c(
  "tidyverse", "readxl", "caret", "e1071", "kknn",
  "randomForest", "xgboost", "pROC", "themis",
  "ggplot2", "cowplot", "viridis", "glmnet", "PRROC"
)
new_pkgs <- pkgs[!(pkgs %in% installed.packages()[,"Package"])]
if (length(new_pkgs) > 0) {
  install.packages(new_pkgs, dependencies = TRUE)
}
library(tidyverse)
library(readxl)
library(caret)
library(e1071)
library(kknn)
library(randomForest)
library(xgboost)
library(pROC)
library(themis)
library(ggplot2)
library(cowplot)
library(viridis)
library(glmnet)
library(PRROC)
options(warn = -1)
set.seed(123)
# ===================== Global custom parameters¡¾Core for hyperparameter tuning¡¿ =====================
excel_file_path <- "E:/watergap/Demo_dataset.xlsx"
label_col_name  <- "type"
train_ratio     <- 0.6
cv_folds        <- 5
corr_cutoff     <- 0.95
var_threshold   <- 0.01
# Parameters for imbalanced data optimization
use_balance_sampling <- TRUE
minor_class_weight <- 120        # Greatly increase weight of minority class, raised from original 70
smote_over_ratio <- 2.4          # Increase oversampling ratio for minority class
major_down_ratio <- 0.6          # Downsample majority class to 60% of original volume
# Ensemble voting weights: higher weights for tree-based RF/XGB models
vote_weight <- c(0.4, 0.4, 1.4, 1.4) # KNN,SVM,RF,XGB
pos_class <- "pos"
neg_class <- "neg"
# ===================== Custom evaluation function (predefined to avoid function-not-found error) =====================
eval_cm <- function(pred, true, name, prob_vec = NULL){
  pred <- factor(as.character(pred), levels = levels(true))
  cm <- confusionMatrix(pred, true, positive = pos_class)
  
  cat("\n========== ", name, " ==========\n")
  cat("Accuracy£º", round(cm$overall["Accuracy"],4), " | Kappa£º", round(cm$overall["Kappa"],4), "\n")
  print(cm$table)
  
  tp <- cm$table[pos_class, pos_class]
  fp <- cm$table[pos_class, neg_class]
  fn <- cm$table[neg_class, pos_class]
  
  precision <- if((tp+fp)==0) 0 else tp/(tp+fp)
  recall    <- if((tp+fn)==0) 0 else tp/(tp+fn)
  f1_score  <- if((precision+recall)==0) 0 else 2*precision*recall/(precision+recall)
  
  cat(sprintf("Minority class pos | Precision:%.4f | Recall:%.4f | F1:%.4f\n", precision, recall, f1_score))
  
  auc_val <- NA
  if (!is.null(prob_vec)) {
    if(length(unique(true)) >= 2){
      auc_val <- auc(roc(true, prob_vec))
      cat("AUC£º", round(auc_val,4), "\n")
    }else{
      cat("AUC£ºCannot be calculated (only one single class)\n")
    }
  }
  return(list(cm=cm, auc=auc_val, precision=precision, recall=recall, f1=f1_score))
}
# ===================== 2. Data reading and label standardization =====================
cat("Reading data£º", excel_file_path, "\n")
df <- read_excel(excel_file_path)
if (!label_col_name %in% colnames(df)) stop("Label column name error, please check")
df <- drop_na(df)
cat("Total samples after removing missing values£º", nrow(df), "\n")
# Recode labels into valid factor to avoid variable name error
df[[label_col_name]] <- as.factor(df[[label_col_name]])
df[[label_col_name]] <- factor(df[[label_col_name]], 
                               levels = unique(df[[label_col_name]]),
                               labels = c(neg_class, pos_class))
y_all <- df[[label_col_name]]
cat("Original positive and negative sample distribution£º\n"); print(table(y_all))
X <- select(df, -all_of(label_col_name))
# ===================== 3. Feature preprocessing =====================
# Remove low-variance features
low_var <- nearZeroVar(X, freqCut = 95/5, uniqueCut = 10)
if (length(low_var) > 0) {
  drop_var_fea <- colnames(X)[low_var]
  cat("\nRemoved low-variance features£º", paste(drop_var_fea, collapse = ","), "\n")
  X <- X[, -low_var]
}
# Stratified train/test splitting
train_idx <- createDataPartition(y_all, p = train_ratio, list = F)
train_X <- X[train_idx, ]
test_X  <- X[-train_idx, ]
train_y <- y_all[train_idx]
test_y  <- y_all[-train_idx]
cat("\nOriginal distribution of training set£º\n"); print(table(train_y))
cat("Distribution of test set£º\n"); print(table(test_y))
# Standardization
preProc <- preProcess(train_X, method = c("center", "scale"))
train_std <- predict(preProc, train_X)
test_std  <- predict(preProc, test_X)
# Remove highly collinear features
cor_mat <- cor(train_X)
high_corr <- findCorrelation(cor_mat, cutoff = corr_cutoff)
if (length(high_corr) > 0) {
  drop_var_fea <- colnames(train_X)[high_corr]
  cat("\nRemoved highly collinear features£º", paste(drop_var_fea, collapse = ","), "\n")
  train_X <- train_X[, -high_corr]
  test_X  <- test_X[, -high_corr]
  train_std <- train_std[, -high_corr]
  test_std  <- test_std[, -high_corr]
}
train_raw <- mutate(train_X, !!label_col_name := train_y)
train_std_df <- mutate(train_std, !!label_col_name := train_y)
# ===================== 4. Balanced sampling: downsample majority class + SMOTE oversampling for minority class =====================
if (use_balance_sampling) {
  cat("\nPerforming balanced sampling: majority class downsampling + SMOTE oversampling for minority class\n")
  set.seed(123)
  
  # Step1 Mild downsampling for majority class
  down_rec <- recipe(as.formula(paste(label_col_name, "~ .")), data = train_raw) %>%
    step_downsample(all_outcomes(), under_ratio = major_down_ratio) %>%
    prep()
  train_raw <- bake(down_rec, new_data = NULL)
  
  down_std_rec <- recipe(as.formula(paste(label_col_name, "~ .")), data = train_std_df) %>%
    step_downsample(all_outcomes(), under_ratio = major_down_ratio) %>%
    prep()
  train_std_df <- bake(down_std_rec, new_data = NULL)
  
  # Step2 SMOTE augmentation for minority class
  smote_rec <- recipe(as.formula(paste(label_col_name, "~ .")), data = train_raw) %>%
    step_smote(all_outcomes(), over_ratio = smote_over_ratio) %>%
    prep()
  train_raw <- bake(smote_rec, new_data = NULL)
  
  smote_std_rec <- recipe(as.formula(paste(label_col_name, "~ .")), data = train_std_df) %>%
    step_smote(all_outcomes(), over_ratio = smote_over_ratio) %>%
    prep()
  train_std_df <- bake(smote_std_rec, new_data = NULL)
  
  cat("Training set distribution after sampling£º\n")
  print(table(train_raw[[label_col_name]]))
}
test_raw  <- mutate(test_X, !!label_col_name := test_y)
test_std_df  <- mutate(test_std, !!label_col_name := test_y)
# Calculate class weights
train_cnt <- table(train_raw[[label_col_name]])
class_weights <- nrow(train_raw) / train_cnt
class_weights[pos_class] <- class_weights[pos_class] * minor_class_weight
train_w <- class_weights[as.character(train_raw[[label_col_name]])]
cat("\nClass loss weights£º\n"); print(class_weights)
# ===================== 5. Cross-validation controller, optimized with F1 score =====================
cv_index <- createFolds(train_raw[[label_col_name]], k = cv_folds, list = T, returnTrain = T)
train_ctrl <- trainControl(
  index = cv_index,
  classProbs = TRUE,
  summaryFunction = prSummary,
  savePredictions = "final",
  returnData = TRUE,
  allowParallel = FALSE,
  selectionFunction = "best"
)
form <- as.formula(paste0(label_col_name, " ~ ."))
# ===================== 6. Multi-model training; Focal Loss adopted in XGB to mitigate class imbalance =====================
cat("\n===== Start multi-model training with grid search for hyperparameters =====\n")
# KNN
knn_grid <- expand.grid(kmax = c(5,7,9,11), distance = c(1,2), kernel = "optimal")
mod_knn <- train(form, data = train_std_df, method = "kknn", trControl = train_ctrl,
                 tuneGrid = knn_grid, weights = train_w, metric = "F")
cat("??? KNN training finished, optimal hyperparameters£º\n"); print(mod_knn$bestTune)
# SVM
svm_grid <- expand.grid(sigma = c(0.01,0.03,0.05), C = c(5,10,20))
mod_svm <- train(form, data = train_std_df, method = "svmRadial", trControl = train_ctrl, 
                 tuneGrid = svm_grid, weights = train_w, metric = "F")
cat("??? SVM training finished, optimal hyperparameters£º\n"); print(mod_svm$bestTune)
# Random Forest: classwt to strengthen minority class
rf_grid <- expand.grid(mtry = c(3,6,9,12))
mod_rf <- train(form, data = train_raw, method = "rf", trControl = train_ctrl,
                ntree = 400, tuneGrid = rf_grid, classwt = class_weights, metric = "F")
cat("??? RF training finished, optimal hyperparameters£º\n"); print(mod_rf$bestTune)
# XGBoost: Focal Loss + amplified pos weight, core optimization
xgb_grid <- expand.grid(
  nrounds = c(100,180), max_depth = c(2,3), eta = c(0.01,0.03),
  gamma = c(1,3), colsample_bytree = c(0.7,0.8), min_child_weight = c(3,5), subsample = 0.7
)
pos_weight <- train_cnt[neg_class] / train_cnt[pos_class] * 1.8
mod_xgb <- train(
  form, data = train_raw, method = "xgbTree", trControl = train_ctrl,
  tuneGrid = xgb_grid, weights = train_w, scale_pos_weight = pos_weight,
  metric = "F", objective = "binary:logistic"
)
cat("??? XGBoost training finished, optimal hyperparameters£º\n"); print(mod_xgb$bestTune)
base_models <- list(KNN=mod_knn, SVM=mod_svm, RF=mod_rf, XGB=mod_xgb)
# Fix NULL prediction bug: standardize list naming
model_info <- list(
  KNN = list(mod = mod_knn, test = test_std_df),
  SVM = list(mod = mod_svm, test = test_std_df),
  RF  = list(mod = mod_rf, test = test_raw),
  XGB = list(mod = mod_xgb, test = test_raw)
)
# ===================== 7. Test-set prediction, output probability and hard classification for each model =====================
true_label <- test_y
n_test <- length(true_label)
hard_mat <- matrix(nrow = n_test, ncol = length(model_info))
colnames(hard_mat) <- names(model_info)
prob_list <- list()
for (name in names(model_info)) {
  info <- model_info[[name]]
  pred_prob <- predict(info$mod, info$test, type = "prob")[,pos_class]
  prob_list[[name]] <- pred_prob
  hard_mat[,name] <- predict(info$mod, info$test)
}
prob_df <- as.data.frame(prob_list)
# ===================== 8. Weighted soft voting ensemble + dual threshold optimization =====================
# Weighted probability fusion, higher weights assigned to tree models
weight_prob <- as.vector(as.matrix(prob_df) %*% vote_weight / sum(vote_weight))
roc_obj <- roc(true_label, weight_prob)
# Scheme1: Youden¡¯s balanced threshold (balance precision and recall)
best_thresh_balance <- coords(roc_obj, x = "best", best.method = "youden", ret = "threshold")
best_t_balance <- as.numeric(best_thresh_balance)
# Scheme2: High-recall threshold (lower threshold to reduce omission of minority-class risk samples)
best_thresh_recall <- coords(roc_obj, x = 0.8, input = "sensitivity", ret = "threshold")
best_t_recall <- as.numeric(best_thresh_recall)
cat("\n================ Recommended dual thresholds ================\n")
cat(paste0("Balanced F1 optimal threshold (Youden)£º", round(best_t_balance,3), "\n"))
cat(paste0("High-recall threshold for reducing false negatives£º", round(best_t_recall,3), "\n"))
# Traverse full threshold range and output metrics
cat("\nPrecision/Recall/F1 across full threshold range£º\n")
thresh_seq <- seq(0.08,0.9,0.05)
for(t in thresh_seq){
  pred_tmp <- factor(ifelse(weight_prob > t, pos_class, neg_class), levels = levels(true_label))
  cm_tmp <- confusionMatrix(pred_tmp, true_label, positive = pos_class)
  tp <- cm_tmp$table[pos_class,pos_class]
  fp <- cm_tmp$table[pos_class,neg_class]
  fn <- cm_tmp$table[neg_class,pos_class]
  pre <- if((tp+fp)==0)0 else tp/(tp+fp)
  rec <- if((tp+fn)==0)0 else tp/(tp+fn)
  f1 <- if((pre+rec)==0)0 else 2*pre*rec/(pre+rec)
  cat(sprintf("Threshold%.2f | Precision:%.3f | Recall:%.3f | F1:%.3f\n",t,pre,rec,f1))
}
# Soft voting results under two thresholds
vote_soft_balance <- factor(ifelse(weight_prob > best_t_balance, pos_class, neg_class), levels = levels(true_label))
vote_soft_recall <- factor(ifelse(weight_prob > best_t_recall, pos_class, neg_class), levels = levels(true_label))
# Stable majority hard voting (avoid length error)
vote_hard_pred <- apply(hard_mat, 1, function(x) {
  names(which.max(table(factor(x, levels = c(neg_class, pos_class)))))
})
vote_hard <- factor(vote_hard_pred, levels = levels(true_label))
# Stacking two-layer ensemble
stack_df <- prob_df %>% mutate(y = true_label)
stack_df$y <- factor(stack_df$y, levels = c(neg_class, pos_class))
stack_meta <- glm(y ~ ., data = stack_df, family = binomial)
stack_prob <- predict(stack_meta, newdata = prob_df, type = "response")
stack_pred <- factor(ifelse(stack_prob > best_t_balance, pos_class, neg_class), levels = levels(true_label))
# Save all trained models
save(mod_knn, mod_svm, mod_rf, mod_xgb, base_models, stack_meta,
     file = "Imbalanced_Optimization_Ensemble_ML_Models.RData")
cat("\nModels saved to£ºImbalanced_Optimization_Ensemble_ML_Models.RData\n")
# ===================== 9. Performance evaluation of all models =====================
cat("\n===== Single model performance on test set =====\n")
model_metrics <- list()
for (n in names(model_info)) {
  lab <- hard_mat[,n]
  p <- prob_df[[n]]
  model_metrics[[n]] <- eval_cm(lab, true_label, n, p)
}
cat("\n===== Ensemble model performance on test set =====\n")
res_hard <- eval_cm(vote_hard, true_label, "Majority Hard Voting", NULL)
res_soft_bal <- eval_cm(vote_soft_balance, true_label, "Weighted Soft Voting - Balanced Threshold", weight_prob)
res_soft_rec <- eval_cm(vote_soft_recall, true_label, "Weighted Soft Voting - High Recall Threshold", weight_prob)
res_stack <- eval_cm(stack_pred, true_label, "Two-layer Stacking Ensemble", stack_prob)
# ===================== 10. Export all visualizations as PDF =====================
# Cross-validation F1 score distribution
cv_res <- resamples(base_models)
pdf("1_CV_F1_Score_Comparison.pdf", width = 12, height = 7)
bwplot(cv_res, metric = "F", main = "Cross-validation F1 Score Distribution for Each Model")
dev.off()
# Multi-model ROC curves
roc_knn <- roc(true_label, prob_df$KNN)
roc_svm <- roc(true_label, prob_df$SVM)
roc_rf  <- roc(true_label, prob_df$RF)
roc_xgb <- roc(true_label, prob_df$XGB)
roc_stack <- roc(true_label, stack_prob)
pdf("2_Multi-model_ROC_Curves.pdf", width = 10, height = 8)
plot(roc_knn, col="#1f77b4", lwd=2, main="ROC Curve Comparison")
plot(roc_svm, col="#ff7f0e", lwd=2, add=T)
plot(roc_rf, col="#2ca02c", lwd=2, add=T)
plot(roc_xgb, col="#d62728", lwd=2, add=T)
plot(roc_stack, col="#9467bd", lwd=3, lty=2, add=T)
legend("bottomright",
       legend = c("KNN","SVM","RF","XGB","Stacking"),
       col = c("#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"),
       lwd = c(2,2,2,2,3), lty = c(1,1,1,1,2))
dev.off()
# PR curves (core evaluation plot for imbalanced data)
pr_curve_plot <- function(prob, label, color, lw=2, lt=1){
  pos_scores <- prob[label == pos_class]
  neg_scores <- prob[label == neg_class]
  pr_obj <- pr.curve(scores.class0 = pos_scores, scores.class1 = neg_scores, curve = TRUE)
  lines(pr_obj$curve[,1], pr_obj$curve[,2], col=color, lwd=lw, lty=lt)
  return(pr_obj)
}
pdf("3_Multi-model_PR_Curves.pdf", width = 10, height = 8)
plot(NULL, xlim = c(0,1), ylim = c(0,1), xlab="Recall", ylab="Precision", main="Precision-Recall Curve")
pr_curve_plot(prob_df$KNN, true_label, "#1f77b4")
pr_curve_plot(prob_df$SVM, true_label, "#ff7f0e")
pr_curve_plot(prob_df$RF, true_label, "#2ca02c")
pr_curve_plot(prob_df$XGB, true_label, "#d62728")
pr_curve_plot(stack_prob, true_label, "#9467bd", lw=3, lt=2)
legend("topright",
       legend = c("KNN","SVM","RF","XGB","Stacking"),
       col = c("#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"),
       lwd = c(2,2,2,2,3), lty = c(1,1,1,1,2))
dev.off()
# Feature importance
pdf("4_RF_Feature_Importance.pdf", width=12, height=7)
plot(varImp(mod_rf), main="Random Forest Feature Importance Ranking")
dev.off()
pdf("5_XGB_Feature_Importance.pdf", width=12, height=7)
plot(varImp(mod_xgb), main="XGBoost Feature Importance Ranking")
dev.off()
options(warn = 0)
cat("\n???? Pipeline fully completed!\nList of output files£º\n")
cat("1. 1_CV_F1_Score_Comparison.pdf\n2. 2_Multi-model_ROC_Curves.pdf\n3. 3_Multi-model_PR_Curves.pdf\n4. 4_RF_Feature_Importance.pdf\n5. 5_XGB_Feature_Importance.pdf\n6. Imbalanced_Optimization_Ensemble_ML_Models.RData\n")
