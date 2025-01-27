library(naivebayes)
library(caret)
library(pROC)

# Parts of this code are from the SNIPPER website (http://mathgene.usc.es/snipper/offline_snipper.html)

results <- data.frame(File = character(),
                      Accuracy = numeric(),
                      ROC_AUC = numeric(),
                      LogLoss = numeric(),
                      stringsAsFactors = FALSE)

all_predictions_continents <- data.frame(File = character(),
                              True = character(),
                              Predicted = character(),
                              Probability = numeric(),
                              stringsAsFactors = FALSE)

log_loss <- function(actual, predicted_probs) {
  eps <- 1e-15
  predicted_probs <- pmin(pmax(predicted_probs, eps), 1 - eps)
  -mean(log(predicted_probs[cbind(seq_along(actual), actual)]))
}

for (i in 1:10) {
  for (j in 1:5){
    file_name <- paste0("train_repeat_cont2_", i, "_fold_", j, ".csv")
    training <- read.csv(file_name)
    file_name <- paste0("test_repeat_cont2_", i, "_fold_", j, ".csv")
    test <- read.csv(file_name)
    training[] <- lapply(training, factor)
    test[] <- lapply(test, factor)
    nb_mod <- NaiveBayes(Population ~ ., data = training)
    pred <- suppressWarnings(predict(nb_mod, test))
    pred <- predict(nb_mod, test, type = "prob")  # This returns a matrix of probabilities
    prob_pred <- predict(nb_mod, test, type = "prob")
    tab <- table(pred$class, test$Population)
    acc <- sum(diag(tab)) / sum(tab)
    test$Population <- factor(test$Population)
    training$Population <- factor(training$Population)
    roc_curve <- multiclass.roc(test$Population, pred$posterior[, 2], levels = rev(levels(test$Population)))
    auc_val <- auc(roc_curve)
    logloss_val <- log_loss(as.numeric(test$Population), prob_pred$posterior)
    pred_data <- data.frame(File = file_name,
                            True = test$Population,
                            Predicted = pred$class,
                            Probability = prob_pred$posterior[, 2])
    
    all_predictions_continents <- rbind(all_predictions_continents, pred_data)
    results <- rbind(results, data.frame(File = file_name,
                                         Accuracy = acc,
                                         ROC_AUC = auc_val,
                                         LogLoss = logloss_val))
  }
}


write.csv(results, "results_SNIPPER_eur.csv", row.names = FALSE)
write.csv(all_predictions_continents, "all_predictions_eur_SNIPPER.csv", row.names = FALSE)
