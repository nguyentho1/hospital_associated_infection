# =============================================================================
#
#  ACADEMIC SIMULATION STUDY
#  Hospital-Acquired Infection (HAI) Surveillance
#  Using Machine Learning Risk Scoring + CUSUM Sequential Monitoring
#
#  PURPOSE:
#    This script simulates a hospital patient population, fits ML models
#    to predict individual HAI risk, and then applies CUSUM control charts
#    to detect ward-level outbreak signals in real time.
#
#    The goal is to understand — conceptually and statistically — how
#    combining ML and CUSUM solves problems neither method solves alone.
#
#  STRUCTURE:
#    SECTION 0  — Environment setup
#    SECTION 1  — Data simulation (patient population)
#    SECTION 2  — Exploratory Data Analysis (EDA)
#    SECTION 3  — ML risk models (Logistic Regression + Random Forest)
#    SECTION 4  — Model calibration & evaluation
#    SECTION 5  — CUSUM control chart (ward-level surveillance)
#    SECTION 6  — Visualisation (6 publication-quality plots)
#    SECTION 7  — Sensitivity analysis (h and k parameters)
#    SECTION 8  — Final performance report
#
#  PACKAGES NEEDED:
#    tidyverse, randomForest, pROC, ggplot2, gridExtra, scales, ResourceSelection
#
#  HOW TO RUN:
#    Rscript HAI_Surveillance_Academic.R
#    or open in RStudio and run line by line.
#
# =============================================================================


# =============================================================================
# SECTION 0 — ENVIRONMENT SETUP
# =============================================================================

# set.seed() makes all random number generation reproducible.
# Without this, every run would produce different patients and different plots.
# 2024 is an arbitrary seed; change it to verify results are seed-independent.
set.seed(2024)

# Define the list of packages this script needs.
# We use a character vector so we can loop over them programmatically.
required_packages <- c(
  "tidyverse",        # data wrangling (dplyr, tidyr) + ggplot2
  "randomForest",     # ensemble tree model for ML risk scoring
  "pROC",             # ROC curves and AUC computation
  "ggplot2",          # layered grammar-of-graphics plotting
  "gridExtra",        # arrange multiple ggplot objects on one page
  "scales",           # axis formatting helpers (percent_format, etc.)
  "ResourceSelection" # Hosmer-Lemeshow goodness-of-fit test for calibration
)

# Loop: if the package is not installed, install it; then load it.
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    # quietly = TRUE suppresses noisy startup messages during the check.
    install.packages(pkg, dependencies = TRUE)
  }
  library(pkg, character.only = TRUE)
  # character.only = TRUE is required because pkg is a string variable,
  # not a bare package name like library(ggplot2).
}

# Create an output directory for all plots and CSV files.
# showWarnings = FALSE prevents an error if the directory already exists.
output_dir <- "HAI_Academic_Output"
dir.create(output_dir, showWarnings = FALSE)

# A small helper to print section banners to the console clearly.
banner <- function(text) {
  cat("\n", strrep("=", 60), "\n", text, "\n", strrep("=", 60), "\n\n", sep = "")
}

banner("SECTION 0 COMPLETE: Environment ready")


# =============================================================================
# SECTION 1 — DATA SIMULATION
# =============================================================================
#
# RATIONALE:
#   We cannot use real patient data for a teaching simulation, so we generate
#   synthetic patients whose infection probabilities follow a logistic model
#   based on known clinical risk factors from the HAI literature.
#
#   The logistic model is:
#
#     log-odds(HAI) = β₀ + β₁·age + β₂·los + β₃·icu + ... + δ·outbreak
#
#   where:
#     β₀       = intercept (sets overall baseline rate ~2-3%)
#     βᵢ       = coefficient for risk factor i (estimated from HAI literature)
#     δ        = ln(outbreakOR) — this is the extra log-odds during an outbreak
#
#   True infection status is drawn from Bernoulli(p), where p = logistic(log-odds).
#   This mirrors how real binary outcomes are generated stochastically.
#
# SIMULATION DESIGN:
#   n = 3000 patients total, admitted sequentially over ~6 months.
#   An outbreak (Clostridium difficile, MRSA, etc.) is injected into Ward-3
#   starting at patient #2200. The outbreak triples the odds of HAI (OR = 3.0).
#   The ML models are TRAINED on patients 1–1500 (pre-outbreak, known labels).
#   CUSUM monitoring is applied to patients 1501–3000 (prospective period).
# =============================================================================

banner("SECTION 1: Simulating Patient Population")

# ── 1.1  Global simulation parameters ─────────────────────────────────────────

N_TOTAL         <- 3000   # Total patients to simulate
N_TRAIN         <- 1500   # Number of patients used for ML training
OUTBREAK_WARD   <- "Ward-3"   # Which ward experiences the outbreak
OUTBREAK_START  <- 2200   # Global patient index when outbreak begins
OUTBREAK_OR     <- 3.0    # Odds ratio of the outbreak effect (3× increased odds)
N_WARDS         <- 5      # Number of hospital wards

# ── 1.2  Simulate patient-level risk factors ──────────────────────────────────

# Total number of patients
n <- N_TOTAL

# AGE: Normally distributed around 62 years (reflecting an elderly hospital
# population). Truncated to [18, 95] to avoid physiologically impossible values.
age <- rnorm(n, mean = 62, sd = 18)
age <- pmax(18, pmin(95, round(age)))
# pmax(18, ...) replaces anything below 18 with 18 (lower truncation)
# pmin(95, ...) replaces anything above 95 with 95 (upper truncation)

# LENGTH OF STAY (days): Exponentially distributed because hospital stays are
# right-skewed — most are short but a few are very long.
# rate = 1/7 means average stay = 7 days (E[X] = 1/rate).
los <- round(rexp(n, rate = 1/7))
los <- pmax(1, pmin(60, los))   # Truncate to [1, 60] days

# BINARY RISK FACTORS: rbinom(n, 1, p) draws n Bernoulli(p) trials.
# Each returns 0 (absent) or 1 (present).
icu_admit    <- rbinom(n, 1, prob = 0.25)  # 25% of patients go to ICU
surgery      <- rbinom(n, 1, prob = 0.40)  # 40% undergo surgery
immunocomp   <- rbinom(n, 1, prob = 0.15)  # 15% are immunocompromised
antibiotics  <- rbinom(n, 1, prob = 0.50)  # 50% receive broad-spectrum antibiotics
central_line <- rbinom(n, 1, prob = 0.30)  # 30% have a central venous catheter
urinary_cath <- rbinom(n, 1, prob = 0.35)  # 35% have a urinary catheter
comorbidity  <- rbinom(n, 1, prob = 0.55)  # 55% have Charlson comorbidity index ≥ 2
diabetes     <- rbinom(n, 1, prob = 0.28)  # 28% have diabetes mellitus

# WARD ASSIGNMENT: Patients are randomly assigned to one of 5 wards.
# The probabilities are slightly unequal to mimic real hospital bed allocation.
ward <- sample(
  paste0("Ward-", 1:N_WARDS),              # Ward labels: "Ward-1" ... "Ward-5"
  size    = n,
  replace = TRUE,                           # Each patient independently sampled
  prob    = c(0.15, 0.20, 0.25, 0.20, 0.20) # Ward-3 is the largest (25%)
)

# ── 1.3  Compute true log-odds using the logistic model ───────────────────────

# The coefficients below represent the log-odds contribution of each unit
# increase in the predictor. Positive values increase infection risk.
#
# These are approximate values from published HAI risk models:
#   Magill SS et al. (2014), NEJM; Cassini A et al. (2016), PLoS Medicine.
#
# β₀ = -4.8:  Intercept. Sets baseline HAI rate ≈ 0.8% for a young, healthy
#              patient with no risk factors — clinically reasonable.

log_odds_true <- (
  -4.8                    # β₀: baseline intercept
  + 0.030 * age           # β₁: each extra year adds 0.030 log-odds
  + 0.065 * los           # β₂: each extra day of stay adds 0.065 log-odds
  + 0.950 * icu_admit     # β₃: ICU admission — strong risk factor
  + 0.700 * surgery       # β₄: post-operative infection risk
  + 1.150 * immunocomp    # β₅: immunocompromised — very strong predictor
  + 0.480 * antibiotics   # β₆: disrupts protective microbiome
  + 0.820 * central_line  # β₇: central-line-associated bloodstream infection
  + 0.620 * urinary_cath  # β₈: catheter-associated urinary tract infection
  + 0.720 * comorbidity   # β₉: comorbidity burden
  + 0.350 * diabetes      # β₁₀: impaired immune response
)

# ── 1.4  Inject outbreak effect ───────────────────────────────────────────────

# outbreak_flag is TRUE for each patient who is:
#   (a) in Ward-3, AND
#   (b) arrives after the outbreak starts (patient index >= OUTBREAK_START)
# Using seq_len(n) creates the sequential index 1, 2, 3, ..., n.
outbreak_flag <- (ward == OUTBREAK_WARD) & (seq_len(n) >= OUTBREAK_START)

# log(OUTBREAK_OR) converts the odds ratio to the log-odds scale.
# This is the δ term in our model: δ = ln(3.0) ≈ 1.099
# Adding this to log_odds_true effectively multiplies the odds by OUTBREAK_OR.
log_odds_true <- log_odds_true + log(OUTBREAK_OR) * as.integer(outbreak_flag)

# ── 1.5  Convert log-odds to probability via logistic (sigmoid) function ──────

# The logistic function maps any real number to (0, 1):
#   p = 1 / (1 + exp(-x))
# This is the standard inverse-logit transformation.
prob_true <- 1 / (1 + exp(-log_odds_true))

# ── 1.6  Draw actual infection outcomes ───────────────────────────────────────

# Each patient's infection status is a Bernoulli draw with their true probability.
# hai = 1 means the patient acquired a hospital infection; 0 means they did not.
hai <- rbinom(n, size = 1, prob = prob_true)

# ── 1.7  Assemble the master data frame ───────────────────────────────────────

# All patient-level variables collected into one data frame.
# Each row = one patient; each column = one attribute.
df <- data.frame(
  patient_id   = seq_len(n),    # Unique sequential patient identifier
  age, los,                     # Continuous risk factors
  icu_admit, surgery, immunocomp,
  antibiotics, central_line,
  urinary_cath, comorbidity, diabetes,  # Binary risk factors
  ward,                         # Ward assignment (character)
  prob_true,                    # True underlying infection probability (known only in simulation)
  hai,                          # Observed infection outcome (0 or 1)
  outbreak_flag                 # Was this patient affected by the simulated outbreak?
)

# ── 1.8  Print simulation summary ─────────────────────────────────────────────

cat("Simulation complete.\n\n")
cat(sprintf("Total patients simulated      : %d\n", nrow(df)))
cat(sprintf("Training cohort (patients 1–%d) : %d\n", N_TRAIN, N_TRAIN))
cat(sprintf("Monitoring cohort (%d–%d)  : %d\n",
            N_TRAIN+1, N_TOTAL, N_TOTAL - N_TRAIN))
cat(sprintf("Overall HAI rate              : %.2f%%\n", mean(df$hai) * 100))
cat(sprintf("Ward-3 HAI rate — PRE-outbreak: %.2f%%\n",
            mean(df$hai[df$ward == OUTBREAK_WARD & !df$outbreak_flag]) * 100))
cat(sprintf("Ward-3 HAI rate — POST-outbreak: %.2f%%\n",
            mean(df$hai[df$ward == OUTBREAK_WARD &  df$outbreak_flag]) * 100))
cat(sprintf("Patients in outbreak period   : %d\n", sum(df$outbreak_flag)))


# =============================================================================
# SECTION 2 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
#
# Before modelling, we characterise the data to ensure the simulation
# behaves as intended and to develop intuition about the risk structure.
# =============================================================================

banner("SECTION 2: Exploratory Data Analysis")

# ── 2.1  HAI rate by ward ─────────────────────────────────────────────────────

# group_by + summarise is the dplyr idiom for split-apply-combine.
# We compute n (count), n_hai (infections), and hai_rate (proportion).
ward_summary <- df %>%
  group_by(ward) %>%
  summarise(
    n_patients = n(),                           # count rows per ward
    n_hai      = sum(hai),                      # count infections
    hai_rate   = round(mean(hai) * 100, 2),     # infection rate as percentage
    mean_age   = round(mean(age), 1),
    mean_los   = round(mean(los), 1),
    pct_icu    = round(mean(icu_admit) * 100, 1),
    .groups    = "drop"                         # remove grouping after summarise
  )

cat("HAI Rate by Ward:\n")
print(ward_summary)

# ── 2.2  Risk factor association table ────────────────────────────────────────

# For each binary risk factor, compute HAI rate when factor is present vs absent,
# and the unadjusted odds ratio (OR).
risk_factors <- c("icu_admit","surgery","immunocomp","antibiotics",
                  "central_line","urinary_cath","comorbidity","diabetes")

association_table <- lapply(risk_factors, function(rf) {
  # Subset patients who have the risk factor (rf_present = 1) vs not
  p1 <- mean(df$hai[df[[rf]] == 1])   # HAI rate if factor present
  p0 <- mean(df$hai[df[[rf]] == 0])   # HAI rate if factor absent

  # Unadjusted odds ratio: OR = (p1/(1-p1)) / (p0/(1-p0))
  or <- (p1 / (1 - p1)) / (p0 / (1 - p0))

  data.frame(
    factor      = rf,
    rate_present = round(p1 * 100, 2),
    rate_absent  = round(p0 * 100, 2),
    crude_OR     = round(or, 2)
  )
}) %>% bind_rows()

# bind_rows stacks the list of data frames into one.
cat("\nRisk Factor Association (Unadjusted OR):\n")
print(association_table)

# Save EDA table
write.csv(ward_summary, file.path(output_dir, "EDA_ward_summary.csv"), row.names = FALSE)
write.csv(association_table, file.path(output_dir, "EDA_risk_factors.csv"), row.names = FALSE)


# =============================================================================
# SECTION 3 — MACHINE LEARNING RISK MODELS
# =============================================================================
#
# WHY DO WE NEED AN ML MODEL BEFORE CUSUM?
#   Raw infection counts are confounded by patient risk mix. A ward with sicker
#   patients will always have more infections — that is NOT an outbreak signal.
#   The ML model provides an EXPECTED infection probability for each patient
#   given their risk profile. CUSUM then monitors deviations from this expectation.
#
#   residual_i = observed_i - expected_i = hai_i - mlScore_i
#
#   If outcomes match expectations: residuals ≈ 0, CUSUM stays flat.
#   If outcomes exceed expectations (outbreak): residuals > 0, CUSUM rises.
#
# MODELS USED:
#   1. Logistic Regression  — interpretable, coefficient-based linear model
#   2. Random Forest        — ensemble of decision trees, handles non-linearity
#   3. Ensemble             — simple average of both scores (reduces variance)
# =============================================================================

banner("SECTION 3: Machine Learning Risk Models")

# ── 3.1  Define features and split data ───────────────────────────────────────

# Feature vector: all clinical risk factors used as ML predictors.
# Importantly, 'ward' is NOT a feature — we want ward to be the monitored unit,
# not a predictor. Including ward would allow the model to absorb the outbreak.
features <- c("age", "los", "icu_admit", "surgery", "immunocomp",
               "antibiotics", "central_line", "urinary_cath", "comorbidity",
               "diabetes")

# Temporal split: train on the FIRST 1500 patients (historical cohort),
# then predict prospectively on the remaining 1500 (monitoring cohort).
# This mimics real-world deployment — you train, then deploy.
train_df <- df[1:N_TRAIN, ]                     # rows 1 to 1500
test_df  <- df[(N_TRAIN + 1):N_TOTAL, ]         # rows 1501 to 3000

cat(sprintf("Training set  : %d patients, %.2f%% HAI rate\n",
            nrow(train_df), mean(train_df$hai) * 100))
cat(sprintf("Test set      : %d patients, %.2f%% HAI rate\n",
            nrow(test_df), mean(test_df$hai) * 100))

# ── 3.2  MODEL A: Logistic Regression ────────────────────────────────────────

# Build a formula object: hai ~ age + los + icu_admit + ... + diabetes
# paste() concatenates the feature names with " + " separator.
# as.formula() converts the string into an R formula object.
formula_lr <- as.formula(paste("hai ~", paste(features, collapse = " + ")))

# glm() fits a Generalised Linear Model.
# family = binomial() tells glm to use the logit link (logistic regression).
# The model estimates β coefficients that maximise the log-likelihood.
lr_model <- glm(formula_lr, data = train_df, family = binomial())

# Print coefficient summary: estimates, standard errors, z-statistics, p-values.
cat("\nLogistic Regression Coefficients:\n")
print(summary(lr_model)$coefficients)

# Predict on the TEST set. type = "response" returns probabilities (not log-odds).
# Each value represents the model's estimated P(HAI=1 | features).
lr_pred_test <- predict(lr_model, newdata = test_df, type = "response")

# ── 3.3  Platt Scaling — Logistic Regression Calibration ─────────────────────
#
# Even well-specified logistic models can be miscalibrated when the test
# population differs from training. Platt scaling is a simple fix:
# fit a NEW logistic regression with just one predictor — the raw score itself.
# This "squishes" overconfident probabilities toward the true rate.

# Build a calibration data frame using TRAINING predictions (not test — that
# would be "cheating" / overfitting the calibration).
lr_pred_train <- predict(lr_model, newdata = train_df, type = "response")
cal_df        <- data.frame(raw_score = lr_pred_train, outcome = train_df$hai)

# Fit the Platt calibration model.
cal_model <- glm(outcome ~ raw_score, data = cal_df, family = binomial())

# Apply the calibration model to the test set raw scores.
lr_pred_calibrated <- predict(
  cal_model,
  newdata = data.frame(raw_score = lr_pred_test),
  type    = "response"
)

# ── 3.4  MODEL B: Random Forest ───────────────────────────────────────────────

# randomForest() builds an ensemble of ntree decision trees via bootstrap
# aggregation (bagging). Each tree is trained on a bootstrap sample of patients
# and a random subset of mtry features at each split.
#
# as.factor() converts hai (0/1 integer) to a factor — required for
# classification mode (vs regression mode). In classification mode, the forest
# produces class probability estimates rather than a continuous prediction.

rf_model <- randomForest(
  x         = train_df[, features],    # feature matrix (no outcome column)
  y         = as.factor(train_df$hai), # outcome as factor (0/1 → class labels)
  ntree     = 500,                     # grow 500 trees; more trees = more stable
  mtry      = 3,                       # try 3 random features at each split
                                       # (rule of thumb: √p for classification)
  importance = TRUE                    # compute variable importance measures
)

# Predict probabilities on the test set.
# type = "prob" returns a matrix: column "0" = P(no HAI), column "1" = P(HAI).
# We extract column "1" for the HAI probability.
rf_pred_test <- predict(rf_model, newdata = test_df[, features], type = "prob")[, "1"]

# ── 3.5  ENSEMBLE SCORE ───────────────────────────────────────────────────────

# Simple average ensemble: reduces variance by combining diverse models.
# This is a "wisdom of crowds" approach — when LR and RF disagree, their
# average is usually closer to the truth than either alone.
ensemble_score <- (lr_pred_calibrated + rf_pred_test) / 2

# Attach predictions back to the test data frame for easy downstream analysis.
test_df$score_lr  <- lr_pred_calibrated   # calibrated logistic regression score
test_df$score_rf  <- rf_pred_test         # random forest score
test_df$score_ens <- ensemble_score       # ensemble score (primary score for CUSUM)

# ── 3.6  Variable importance from Random Forest ───────────────────────────────

# importance() extracts two importance measures:
# MeanDecreaseAccuracy: how much accuracy drops when this feature is permuted
# MeanDecreaseGini: total reduction in node impurity from splits on this feature
importance_df <- data.frame(
  feature            = rownames(importance(rf_model)),
  MeanDecreaseGini   = importance(rf_model)[, "MeanDecreaseGini"],
  MeanDecreaseAccuracy = importance(rf_model)[, "MeanDecreaseAccuracy"]
) %>% arrange(desc(MeanDecreaseGini))  # sort by Gini importance

cat("\nRandom Forest Variable Importance:\n")
print(importance_df)


# =============================================================================
# SECTION 4 — MODEL CALIBRATION & EVALUATION
# =============================================================================
#
# A model is DISCRIMINATING if it ranks high-risk patients above low-risk ones.
# AUC (area under the ROC curve) measures this: 0.5 = random, 1.0 = perfect.
#
# A model is CALIBRATED if its predicted probabilities match observed rates.
# A patient predicted at 0.20 risk should have HAI ~20% of the time.
# We assess calibration with Hosmer-Lemeshow (HL) test and calibration plots.
# =============================================================================

banner("SECTION 4: Model Calibration & Evaluation")

# ── 4.1  AUC via ROC curve ────────────────────────────────────────────────────

# roc() computes the ROC curve: sensitivity vs (1-specificity) across all
# possible decision thresholds.
# quiet = TRUE suppresses verbose output.
roc_lr  <- roc(test_df$hai, test_df$score_lr,  quiet = TRUE)
roc_rf  <- roc(test_df$hai, test_df$score_rf,  quiet = TRUE)
roc_ens <- roc(test_df$hai, test_df$score_ens, quiet = TRUE)

# auc() extracts the single AUC number from the ROC object.
auc_lr  <- as.numeric(auc(roc_lr))
auc_rf  <- as.numeric(auc(roc_rf))
auc_ens <- as.numeric(auc(roc_ens))

cat(sprintf("Logistic Regression AUC       : %.4f\n", auc_lr))
cat(sprintf("Random Forest AUC             : %.4f\n", auc_rf))
cat(sprintf("Ensemble AUC                  : %.4f\n", auc_ens))

# ── 4.2  Brier Score (proper scoring rule) ───────────────────────────────────

# Brier Score = mean squared error between predicted probability and outcome.
# Lower is better. Perfect calibration gives BS ≈ base rate × (1 - base rate).
brier_lr  <- mean((test_df$score_lr  - test_df$hai)^2)
brier_rf  <- mean((test_df$score_rf  - test_df$hai)^2)
brier_ens <- mean((test_df$score_ens - test_df$hai)^2)

cat(sprintf("Logistic Regression Brier     : %.4f\n", brier_lr))
cat(sprintf("Random Forest Brier           : %.4f\n", brier_rf))
cat(sprintf("Ensemble Brier                : %.4f\n", brier_ens))

# ── 4.3  Hosmer-Lemeshow calibration test ────────────────────────────────────

# The HL test divides patients into g = 10 deciles of predicted risk
# and compares observed vs expected infection counts via a chi-square test.
# H₀: model is well-calibrated.
# A non-significant p-value (p > 0.05) indicates good calibration.
# NOTE: HL test is sensitive to sample size; large n → often significant even
#       for clinically well-calibrated models. Interpret with this in mind.

hl_lr  <- hoslem.test(test_df$hai, test_df$score_lr,  g = 10)
hl_ens <- hoslem.test(test_df$hai, test_df$score_ens, g = 10)

cat(sprintf("\nHosmer-Lemeshow LR  p-value   : %.4f (%s)\n",
            hl_lr$p.value,
            ifelse(hl_lr$p.value > 0.05, "Good calibration", "Possible miscalibration")))
cat(sprintf("Hosmer-Lemeshow Ens p-value   : %.4f (%s)\n",
            hl_ens$p.value,
            ifelse(hl_ens$p.value > 0.05, "Good calibration", "Possible miscalibration")))

# ── 4.4  Calibration decile data for plotting ─────────────────────────────────

# Divide patients into 10 equal groups (deciles) by their predicted score.
# For each decile, compare the mean predicted score to the observed HAI rate.
# On a perfectly calibrated model, these should lie on the diagonal (y = x).

n_bins <- 10
test_df$decile <- ntile(test_df$score_ens, n_bins)
# ntile() assigns rank-based bin labels (1 = lowest risk decile, 10 = highest).

calibration_data <- test_df %>%
  group_by(decile) %>%
  summarise(
    mean_predicted = mean(score_ens),       # average predicted probability in decile
    mean_observed  = mean(hai),             # actual HAI rate in decile
    n              = n(),
    .groups        = "drop"
  )

cat("\nCalibration Table (Ensemble Score — 10 Deciles):\n")
print(calibration_data)


# =============================================================================
# SECTION 5 — CUSUM SURVEILLANCE
# =============================================================================
#
# THE CUSUM ALGORITHM:
#   Designed by Page (1954), CUSUM detects a sustained shift in a data stream.
#   Unlike Shewhart charts (which trigger on a single extreme point), CUSUM
#   accumulates small persistent deviations — making it sensitive to gradual drifts.
#
#   STANDARD CUSUM EQUATION (upward shift detection):
#     S₀ = 0
#     Sₙ = max(0,  Sₙ₋₁ + (xₙ - μ₀) - k)
#
#   where:
#     xₙ   = the n-th observation (here: residual = observed - predicted)
#     μ₀   = expected value under null (0, because residuals should average 0)
#     k    = reference value (allowable slack): k = δ/2, where δ is the
#            minimum shift you want to detect reliably
#     Sₙ   = cumulative sum statistic
#     h    = decision threshold: signal when Sₙ > h
#
#   INTUITION:
#     Each term (xₙ - μ₀ - k) is negative when outcomes match expectation,
#     which drags Sₙ back toward zero. But when outcomes persistently exceed
#     expectation (outbreak), the positive terms win and Sₙ climbs until alarm.
#
#   IN THIS CONTEXT:
#     xₙ = hai_n - mlScore_n   (residual: was patient infected MORE than expected?)
#     k  = 0.5 (we want to detect a rate that is at least 0.5 units above expected)
#     h  = 4.0 (trigger after sufficient accumulated evidence)
#
#   RESET POLICY:
#     After an alert, Sₙ resets to 0. This allows detection of subsequent
#     outbreaks if the first one resolves and a new one begins.
# =============================================================================

banner("SECTION 5: CUSUM Surveillance")

# ── 5.1  Core CUSUM function ──────────────────────────────────────────────────

#' Run CUSUM sequential monitoring on a stream of ML-scored patients.
#'
#' @param ml_scores  Numeric vector: model's predicted HAI probability per patient
#' @param outcomes   Integer vector: observed HAI status (0 or 1) per patient
#' @param k          Reference value (allowable slack). Typical: 0.5.
#'                   Smaller k = more sensitive but more false alarms.
#' @param h          Decision threshold. Smaller h = faster alerts, more false alarms.
#'                   Larger h = fewer alarms but slower detection.
#' @param reset      Logical: should CUSUM reset to 0 after each alert? Default TRUE.
#'
#' @return A data frame with one row per patient containing:
#'         residual, S_pos (upward stat), S_neg (downward stat), alert flag.

run_cusum <- function(ml_scores, outcomes, k = 0.5, h = 4.0, reset = TRUE) {

  n      <- length(ml_scores)   # number of patients in this stream

  # Pre-allocate result vectors (faster than growing in a loop).
  S_pos    <- numeric(n)   # Upward CUSUM: detects MORE infections than expected
  S_neg    <- numeric(n)   # Downward CUSUM: detects FEWER infections (improvement)
  residual <- numeric(n)   # Observed - Predicted for each patient
  alert    <- logical(n)   # TRUE if an alert fires at this patient

  # Running statistics: start at 0 (nothing has accumulated yet).
  s_up <- 0
  s_dn <- 0

  for (i in seq_len(n)) {
    # Residual: positive when patient was infected more than predicted,
    # negative when not infected despite high predicted risk.
    residual[i] <- outcomes[i] - ml_scores[i]

    # UPWARD CUSUM UPDATE:
    # Add (residual - k) to the running sum, floor at 0.
    # Floor at 0 means: don't let the statistic go negative (we restart counting).
    # This is the "max(0, ...)" in the CUSUM formula.
    s_up <- max(0, s_up + residual[i] - k)

    # DOWNWARD CUSUM UPDATE:
    # Mirror of upward: detects when HAI rate is LOWER than expected.
    # Useful for monitoring quality improvement after an intervention.
    s_dn <- max(0, s_dn - residual[i] - k)

    # Store the current statistic values.
    S_pos[i] <- s_up
    S_neg[i] <- s_dn

    # ALERT CHECK: if S_pos exceeds threshold h, fire an alert.
    if (s_up > h) {
      alert[i] <- TRUE
      # RESET: bring statistic back to 0 after alert.
      # Justification: an alert triggers a clinical review/intervention;
      # the surveillance "clock" restarts fresh after that investigation.
      if (reset) s_up <- 0
    }
  }

  # Return a structured data frame for downstream analysis and plotting.
  data.frame(
    index    = seq_len(n),      # patient sequence number within this ward
    score    = ml_scores,       # ML predicted probability
    outcome  = outcomes,        # actual HAI status
    residual = residual,        # observed - predicted
    S_pos    = round(S_pos, 4), # upward CUSUM statistic
    S_neg    = round(S_neg, 4), # downward CUSUM statistic
    alert    = alert            # alert fired at this patient?
  )
}

# ── 5.2  Apply CUSUM to each ward separately ──────────────────────────────────

# CUSUM parameters (will be explored in sensitivity analysis later).
K_VALUE <- 0.5   # reference value k
H_VALUE <- 4.0   # decision threshold h

# We monitor the TEST cohort only (patients 1501–3000).
# The ML model was trained on 1–1500 and is now deployed prospectively.

ward_cusum_list <- list()  # store results per ward

for (w in paste0("Ward-", 1:N_WARDS)) {

  # Extract test patients belonging to this ward.
  ward_test <- test_df[test_df$ward == w, ]

  # Skip if fewer than 5 patients (insufficient data for monitoring).
  if (nrow(ward_test) < 5) next

  # Run CUSUM on this ward's patient stream.
  cu <- run_cusum(
    ml_scores = ward_test$score_ens,   # ensemble ML score
    outcomes  = ward_test$hai,         # actual outcomes
    k         = K_VALUE,
    h         = H_VALUE,
    reset     = TRUE
  )

  # Attach ward metadata to the CUSUM result.
  cu$ward        <- w
  cu$patient_id  <- ward_test$patient_id   # global patient IDs for traceability
  cu$global_idx  <- ward_test$patient_id   # same here (patient_id = sequential index)
  cu$outbreak    <- ward_test$outbreak_flag

  ward_cusum_list[[w]] <- cu
}

# Combine all wards into one long data frame.
ward_cusum_df <- bind_rows(ward_cusum_list)

# ── 5.3  Detection performance summary ───────────────────────────────────────

# For Ward-3: find the first alert and compare to when the outbreak started.
w3_cusum <- ward_cusum_df[ward_cusum_df$ward == OUTBREAK_WARD, ]

# Find index of first outbreak patient within Ward-3's monitoring sequence.
w3_outbreak_local <- which(w3_cusum$outbreak)[1]

# Find index of first CUSUM alert within Ward-3's monitoring sequence.
w3_first_alert <- which(w3_cusum$alert)[1]

# Detection delay = how many Ward-3 patients arrived BETWEEN outbreak start
# and first alarm. Smaller delay = better system sensitivity.
detection_delay <- if (!is.na(w3_first_alert) && !is.na(w3_outbreak_local)) {
  max(0, w3_first_alert - w3_outbreak_local)
} else {
  NA  # outbreak not detected
}

# Count false alarms in non-outbreak wards.
# A false alarm = alert fired on a ward with no injected outbreak.
false_alarms <- ward_cusum_df %>%
  filter(ward != OUTBREAK_WARD) %>%
  summarise(total = sum(alert)) %>%
  pull(total)

cat(sprintf("Ward-3 outbreak start (local) : Patient #%s\n",
            ifelse(is.na(w3_outbreak_local), "N/A", w3_outbreak_local)))
cat(sprintf("Ward-3 first CUSUM alert      : Patient #%s\n",
            ifelse(is.na(w3_first_alert), "MISSED", w3_first_alert)))
cat(sprintf("Detection delay               : %s patients\n",
            ifelse(is.na(detection_delay), "MISSED", detection_delay)))
cat(sprintf("False alarms (non-outbreak)   : %d\n", false_alarms))
cat(sprintf("Average run length (ARL) ≈    : %.0f patients between false alarms\n",
            nrow(ward_cusum_df[ward_cusum_df$ward != OUTBREAK_WARD, ]) /
              max(1, false_alarms)))


# =============================================================================
# SECTION 6 — VISUALISATION
# =============================================================================
#
# Six plots are produced, each targeting a different research question:
#   Plot 1: How are patients distributed across risk deciles?
#   Plot 2: How well do models discriminate (ROC curves)?
#   Plot 3: Are predicted probabilities calibrated?
#   Plot 4: What does the CUSUM trajectory look like per ward?
#   Plot 5: What does the residual stream look like for Ward-3?
#   Plot 6: What is the random forest's view of feature importance?
# =============================================================================

banner("SECTION 6: Generating Plots")

# Common theme for all plots: clean white background, readable fonts.
theme_academic <- theme_bw(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 13, hjust = 0),
    plot.subtitle = element_text(size = 10, colour = "grey40", hjust = 0),
    plot.caption  = element_text(size = 9,  colour = "grey50", hjust = 1),
    strip.background = element_rect(fill = "#2C3E50"),
    strip.text    = element_text(colour = "white", face = "bold", size = 11),
    legend.position  = "bottom",
    legend.title     = element_text(size = 10),
    panel.grid.minor = element_blank()
  )

# ── PLOT 1: Predicted risk distribution by HAI status ────────────────────────

plot1 <- ggplot(
    test_df,
    aes(x = score_ens, fill = factor(hai))
  ) +
  # geom_histogram with alpha = 0.65 so both groups are visible when overlapping.
  # bins = 40 gives fine resolution; position = "identity" overlaps rather than stacks.
  geom_histogram(bins = 40, alpha = 0.65, position = "identity") +
  # scale_x_continuous with percent_format() converts 0.10 → "10%" on axis.
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  # Custom colours: blue = no HAI, red = HAI (standard epidemiology convention).
  scale_fill_manual(
    values = c("0" = "#3498DB", "1" = "#E74C3C"),
    labels = c("No HAI", "HAI Occurred")
  ) +
  labs(
    title    = "Figure 1: Distribution of ML Ensemble Risk Scores",
    subtitle = "Higher scores indicate greater predicted infection probability; HAI patients should skew right",
    x        = "Predicted HAI Probability (Ensemble Score)",
    y        = "Number of Patients",
    fill     = "Outcome",
    caption  = "Training cohort: n=1500; Monitoring cohort: n=1500"
  ) +
  theme_academic

ggsave(file.path(output_dir, "Fig1_Risk_Score_Distribution.png"),
       plot1, width = 9, height = 5, dpi = 180)
cat("Saved: Fig1_Risk_Score_Distribution.png\n")

# ── PLOT 2: ROC Curves ────────────────────────────────────────────────────────

# Convert ROC objects to data frames for ggplot2.
# Each ROC object stores vectors of sensitivity and specificity.
roc_to_df <- function(roc_obj, model_label) {
  data.frame(
    FPR   = 1 - roc_obj$specificities,   # False Positive Rate = 1 - Specificity
    TPR   = roc_obj$sensitivities,        # True Positive Rate  = Sensitivity
    model = model_label
  )
}

roc_df <- bind_rows(
  roc_to_df(roc_lr,  sprintf("Logistic Reg (AUC=%.3f)", auc_lr)),
  roc_to_df(roc_rf,  sprintf("Random Forest (AUC=%.3f)", auc_rf)),
  roc_to_df(roc_ens, sprintf("Ensemble (AUC=%.3f)", auc_ens))
)

plot2 <- ggplot(roc_df, aes(x = FPR, y = TPR, colour = model)) +
  # ROC lines for each model.
  geom_line(linewidth = 1.1) +
  # Reference diagonal: the ROC curve of a random (uninformative) classifier.
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", colour = "grey50", linewidth = 0.8) +
  annotate("text", x = 0.75, y = 0.60, label = "Random classifier\n(AUC=0.50)",
           colour = "grey50", size = 3.2) +
  scale_colour_brewer(palette = "Dark2") +
  coord_equal() +    # ensure square aspect ratio (standard for ROC plots)
  labs(
    title    = "Figure 2: ROC Curves — ML Risk Models",
    subtitle = "AUC measures discrimination: how well the model ranks infected above uninfected patients",
    x        = "False Positive Rate (1 – Specificity)",
    y        = "True Positive Rate (Sensitivity)",
    colour   = "Model",
    caption  = "Evaluated on prospective monitoring cohort (n=1500)"
  ) +
  theme_academic

ggsave(file.path(output_dir, "Fig2_ROC_Curves.png"),
       plot2, width = 7, height = 6, dpi = 180)
cat("Saved: Fig2_ROC_Curves.png\n")

# ── PLOT 3: Calibration plot ──────────────────────────────────────────────────

# A well-calibrated model's points lie on the diagonal line y = x.
# Points above the diagonal: model UNDERESTIMATES risk (observed > predicted).
# Points below the diagonal: model OVERESTIMATES risk (observed < predicted).

plot3 <- ggplot(calibration_data, aes(x = mean_predicted, y = mean_observed)) +
  # Reference diagonal (perfect calibration).
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", colour = "grey50", linewidth = 0.8) +
  # Actual calibration points, sized by number of patients in each decile.
  geom_point(aes(size = n), colour = "#2980B9", alpha = 0.85) +
  # Connect the dots to show trend.
  geom_line(colour = "#2980B9", linewidth = 0.8) +
  # Size legend: larger circles = more patients in that decile.
  scale_size_continuous(range = c(3, 10), name = "Patients in decile") +
  # Show both axes on same [0, max] scale for fair comparison.
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title    = "Figure 3: Calibration Plot — Ensemble Model (10 Deciles)",
    subtitle = "Perfect calibration: all points on the dashed diagonal",
    x        = "Mean Predicted Probability",
    y        = "Observed HAI Rate",
    caption  = sprintf("Hosmer-Lemeshow p = %.3f", hl_ens$p.value)
  ) +
  theme_academic

ggsave(file.path(output_dir, "Fig3_Calibration_Plot.png"),
       plot3, width = 7, height = 6, dpi = 180)
cat("Saved: Fig3_Calibration_Plot.png\n")

# ── PLOT 4: Ward-level CUSUM trajectories ────────────────────────────────────

plot4 <- ggplot(
    ward_cusum_df,
    aes(x = index, y = S_pos, colour = ward, group = ward)
  ) +
  geom_line(linewidth = 0.9) +
  # Horizontal dashed line at threshold h: alerts fire above this line.
  geom_hline(yintercept = H_VALUE,
             linetype = "dashed", colour = "black", linewidth = 0.8) +
  # Label the threshold line.
  annotate("text", x = Inf, y = H_VALUE + 0.1,
           label = paste0("h = ", H_VALUE), hjust = 1.1, size = 3.2) +
  # Red triangles at each alert event.
  geom_point(
    data   = filter(ward_cusum_df, alert),
    aes(x = index, y = S_pos),
    colour = "red", shape = 17, size = 3
  ) +
  # Use separate panels for each ward (facets). scales = "free_x" allows
  # different x ranges because wards have different numbers of patients.
  facet_wrap(~ward, ncol = 1, scales = "free_x") +
  scale_colour_brewer(palette = "Set1") +
  labs(
    title    = "Figure 4: CUSUM Statistics by Ward (Prospective Monitoring Period)",
    subtitle = paste0("Red ▲ = alert triggered (S+ > h=", H_VALUE,
                      "). Ward-3 should show a sustained climb during the simulated outbreak."),
    x        = "Patient Sequence Index (within ward)",
    y        = "CUSUM Statistic  S+",
    colour   = "Ward",
    caption  = paste0("CUSUM parameters: k = ", K_VALUE, ", h = ", H_VALUE,
                      " | Reset policy: reset after each alert")
  ) +
  theme_academic

ggsave(file.path(output_dir, "Fig4_Ward_CUSUM_Panels.png"),
       plot4, width = 10, height = 14, dpi = 180)
cat("Saved: Fig4_Ward_CUSUM_Panels.png\n")

# ── PLOT 5: Ward-3 residual stream ────────────────────────────────────────────

# This plot zooms into Ward-3 to show the raw signal driving the CUSUM.
# Red bars = patient infected more than expected (positive residual, bad).
# Blue bars = patient not infected despite predicted risk (negative residual, good).
# The dark overlay shows the scaled CUSUM trajectory rising into the outbreak period.

w3_plot_data <- ward_cusum_df %>% filter(ward == OUTBREAK_WARD)

# We overlay the CUSUM statistic on the residual bar chart.
# Because they're on different scales, we scale S_pos to fit within residual range.
scale_factor <- max(abs(w3_plot_data$residual)) /
                max(w3_plot_data$S_pos + 0.001)  # avoid division by zero

plot5 <- ggplot(w3_plot_data, aes(x = index)) +
  # Residual bars: red = positive (HAI exceeded prediction), blue = negative.
  geom_col(aes(y = residual, fill = residual > 0), alpha = 0.75, width = 0.9) +
  # Scaled CUSUM line overlaid on residuals.
  geom_line(aes(y = S_pos * scale_factor), colour = "#2C3E50",
            linewidth = 1.2, linetype = "solid") +
  # Vertical line where the outbreak began.
  geom_vline(xintercept = w3_outbreak_local,
             colour = "#E74C3C", linetype = "dotted", linewidth = 1.0) +
  annotate("text", x = w3_outbreak_local + 1, y = 0.9,
           label = "← Outbreak start", hjust = 0, colour = "#E74C3C", size = 3.5) +
  # Horizontal zero line for reference.
  geom_hline(yintercept = 0, colour = "grey50", linewidth = 0.4) +
  scale_fill_manual(
    values = c("TRUE" = "#E74C3C", "FALSE" = "#3498DB"),
    labels = c("Below prediction (negative residual)",
               "Above prediction (positive residual)"),
    name   = "Residual direction"
  ) +
  labs(
    title    = paste0("Figure 5: ", OUTBREAK_WARD, " — Residual Stream & CUSUM Trajectory"),
    subtitle = "Red bars = observed infection exceeded ML prediction. Dark line = scaled CUSUM S+.",
    x        = paste0("Patient Index within ", OUTBREAK_WARD, " (Monitoring Period)"),
    y        = "Residual (Observed – Predicted)",
    caption  = "Dark overlay = CUSUM S+ (scaled to residual axis for display)"
  ) +
  theme_academic

ggsave(file.path(output_dir, "Fig5_Ward3_Residuals_CUSUM.png"),
       plot5, width = 11, height = 5, dpi = 180)
cat("Saved: Fig5_Ward3_Residuals_CUSUM.png\n")

# ── PLOT 6: Variable importance ───────────────────────────────────────────────

plot6 <- ggplot(importance_df,
                aes(x = reorder(feature, MeanDecreaseGini),
                    y = MeanDecreaseGini)) +
  # Horizontal bar chart: easier to read feature names.
  geom_col(fill = "#8E44AD", alpha = 0.85, width = 0.7) +
  # Flip axes so features are on y-axis and importance on x-axis.
  coord_flip() +
  labs(
    title    = "Figure 6: Random Forest Variable Importance",
    subtitle = "Mean Decrease in Gini Impurity: how much each feature contributes to pure splits",
    x        = "Clinical Risk Factor",
    y        = "Mean Decrease in Gini Impurity",
    caption  = "Larger values = feature more important to the model's predictions"
  ) +
  theme_academic

ggsave(file.path(output_dir, "Fig6_Variable_Importance.png"),
       plot6, width = 8, height = 5, dpi = 180)
cat("Saved: Fig6_Variable_Importance.png\n")


# =============================================================================
# SECTION 7 — SENSITIVITY ANALYSIS
# =============================================================================
#
# RESEARCH QUESTION: How do CUSUM parameters k and h affect performance?
#
# We systematically vary:
#   k ∈ {0.2, 0.3, 0.5, 0.7, 0.9}   — allowable slack (sensitivity dial)
#   h ∈ {2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0}  — decision threshold
#
# For each combination, on Ward-3 we compute:
#   1. Number of alerts (ideally exactly 1 true alarm)
#   2. Detection delay (smaller = better)
#   3. False alarm rate on non-outbreak wards (lower = better)
#
# This is the classic sensitivity-specificity tradeoff:
#   Low h / low k → faster detection BUT more false alarms
#   High h / high k → fewer false alarms BUT slower detection
# =============================================================================

banner("SECTION 7: Sensitivity Analysis")

# Define grids for the two parameters.
k_grid <- c(0.2, 0.3, 0.5, 0.7, 0.9)
h_grid <- c(2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)

# expand.grid() creates every combination of k and h values.
# Result is a data frame with nrow = length(k_grid) × length(h_grid).
param_grid <- expand.grid(k = k_grid, h = h_grid)

# For each parameter combination, run CUSUM and collect metrics.
sensitivity_results <- lapply(seq_len(nrow(param_grid)), function(i) {

  k_val <- param_grid$k[i]
  h_val <- param_grid$h[i]

  # --- Ward-3: true outbreak ward ---
  w3_test  <- test_df[test_df$ward == OUTBREAK_WARD, ]
  cu_w3    <- run_cusum(w3_test$score_ens, w3_test$hai, k = k_val, h = h_val)
  n_alerts <- sum(cu_w3$alert)

  # Local index of the first Ward-3 patient in the outbreak period.
  outbreak_local <- which(w3_test$outbreak_flag)[1]
  first_alert    <- which(cu_w3$alert)[1]

  delay <- if (!is.na(first_alert) && !is.na(outbreak_local)) {
    max(0, first_alert - outbreak_local)
  } else NA_real_

  # --- Non-outbreak wards: count false alarms ---
  fa <- 0
  for (w in setdiff(paste0("Ward-", 1:N_WARDS), OUTBREAK_WARD)) {
    w_test <- test_df[test_df$ward == w, ]
    if (nrow(w_test) < 5) next
    cu_w <- run_cusum(w_test$score_ens, w_test$hai, k = k_val, h = h_val)
    fa   <- fa + sum(cu_w$alert)
  }

  data.frame(k = k_val, h = h_val,
             n_alerts_ward3   = n_alerts,
             detection_delay  = delay,
             false_alarms     = fa,
             detected         = !is.na(first_alert))

}) %>% bind_rows()

cat("\nSensitivity Analysis Results (Ward-3):\n")
print(sensitivity_results, digits = 3)

# Save to CSV for further analysis.
write.csv(sensitivity_results,
          file.path(output_dir, "Sensitivity_Analysis.csv"),
          row.names = FALSE)

# ── Sensitivity Plot: Detection Delay heatmap ─────────────────────────────────

# Replace NA (missed outbreak) with a large penalty value for visualisation.
sensitivity_results$delay_plot <- ifelse(
  is.na(sensitivity_results$detection_delay),
  max(sensitivity_results$detection_delay, na.rm = TRUE) + 15,
  sensitivity_results$detection_delay
)

plot_sa1 <- ggplot(sensitivity_results,
                   aes(x = factor(h), y = factor(k), fill = delay_plot)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  # Print the actual delay (or "MISS") inside each cell.
  geom_text(aes(label = ifelse(detected,
                               as.character(round(detection_delay)),
                               "MISS")),
            size = 4, fontface = "bold",
            colour = ifelse(sensitivity_results$detected, "white", "red")) +
  # Gradient: blue = fast detection (good), yellow = slow detection (bad).
  scale_fill_gradient(low = "#1A5276", high = "#F4D03F",
                      name = "Detection delay\n(patients)") +
  labs(
    title    = "Figure 7a: Detection Delay by CUSUM Parameters (Ward-3)",
    subtitle = "Cells show patients between outbreak start and first alert. 'MISS' = outbreak not detected.",
    x        = "Decision Threshold h",
    y        = "Reference Value k",
    caption  = paste0("Outbreak OR = ", OUTBREAK_OR, " | Monitoring period n = ", N_TOTAL - N_TRAIN)
  ) +
  theme_academic +
  theme(legend.position = "right")

ggsave(file.path(output_dir, "Fig7a_Sensitivity_Delay.png"),
       plot_sa1, width = 9, height = 5, dpi = 180)
cat("Saved: Fig7a_Sensitivity_Delay.png\n")

# ── Sensitivity Plot: False alarm heatmap ─────────────────────────────────────

plot_sa2 <- ggplot(sensitivity_results,
                   aes(x = factor(h), y = factor(k), fill = false_alarms)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = false_alarms), size = 4, fontface = "bold", colour = "white") +
  # Gradient: dark blue = few false alarms (good), orange = many (bad).
  scale_fill_gradient(low = "#1A5276", high = "#E67E22",
                      name = "False alarms\n(other wards)") +
  labs(
    title    = "Figure 7b: False Alarm Count by CUSUM Parameters (Non-Outbreak Wards)",
    subtitle = "Cells show total alerts across Ward-1, 2, 4, 5 (no true outbreak present).",
    x        = "Decision Threshold h",
    y        = "Reference Value k"
  ) +
  theme_academic +
  theme(legend.position = "right")

ggsave(file.path(output_dir, "Fig7b_Sensitivity_FalseAlarms.png"),
       plot_sa2, width = 9, height = 5, dpi = 180)
cat("Saved: Fig7b_Sensitivity_FalseAlarms.png\n")


# =============================================================================
# SECTION 8 — FINAL PERFORMANCE REPORT
# =============================================================================

banner("SECTION 8: Final Summary Report")

cat("╔══════════════════════════════════════════════════════════╗\n")
cat("║        HAI SURVEILLANCE SIMULATION — FINAL REPORT       ║\n")
cat("╠══════════════════════════════════════════════════════════╣\n")
cat("║  STUDY DESIGN                                            ║\n")
cat(sprintf("║  Total patients simulated     : %-25d║\n", N_TOTAL))
cat(sprintf("║  Training cohort (1–%d)      : %-25d║\n", N_TRAIN, N_TRAIN))
cat(sprintf("║  Prospective monitoring       : %-25d║\n", N_TOTAL - N_TRAIN))
cat(sprintf("║  Number of wards monitored    : %-25d║\n", N_WARDS))
cat(sprintf("║  Outbreak ward                : %-25s║\n", OUTBREAK_WARD))
cat(sprintf("║  Outbreak odds ratio          : %-25.1f║\n", OUTBREAK_OR))
cat("╠══════════════════════════════════════════════════════════╣\n")
cat("║  ML MODEL PERFORMANCE                                    ║\n")
cat(sprintf("║  Logistic Regression AUC      : %-25.4f║\n", auc_lr))
cat(sprintf("║  Random Forest AUC            : %-25.4f║\n", auc_rf))
cat(sprintf("║  Ensemble AUC                 : %-25.4f║\n", auc_ens))
cat(sprintf("║  Ensemble Brier Score         : %-25.4f║\n", brier_ens))
cat(sprintf("║  H-L calibration p-value      : %-25.4f║\n", hl_ens$p.value))
cat("╠══════════════════════════════════════════════════════════╣\n")
cat("║  CUSUM SURVEILLANCE                                      ║\n")
cat(sprintf("║  Parameters: k=%.1f, h=%.1f                              ║\n",
            K_VALUE, H_VALUE))
cat(sprintf("║  Outbreak detected            : %-25s║\n",
            ifelse(!is.na(w3_first_alert), "YES", "NO")))
cat(sprintf("║  Detection delay              : %-25s║\n",
            ifelse(is.na(detection_delay), "N/A (missed)", paste(detection_delay, "patients"))))
cat(sprintf("║  False alarms (other wards)   : %-25d║\n", false_alarms))
cat("╠══════════════════════════════════════════════════════════╣\n")
cat("║  OUTPUT FILES                                            ║\n")
for (f in list.files(output_dir)) {
  cat(sprintf("║  → %-53s║\n", f))
}
cat("╚══════════════════════════════════════════════════════════╝\n")

cat("\n\nKEY RESEARCH INSIGHTS FROM THIS SIMULATION:\n")
cat("─────────────────────────────────────────────────────────\n")
cat("1. ML Risk Adjustment is critical. Without it, CUSUM would confound\n")
cat("   high-acuity wards (ICU) with outbreak signals, producing chronic\n")
cat("   false alarms. The residual (observed - predicted) removes this bias.\n\n")
cat("2. CUSUM accumulates evidence. A single high-risk patient or single\n")
cat("   infection does not trigger an alert — the signal must be sustained.\n")
cat("   This dramatically reduces alert fatigue vs threshold-based alarms.\n\n")
cat("3. Parameter trade-off is unavoidable. Small h detects outbreaks faster\n")
cat("   but generates false alarms. Choosing h/k requires balancing:\n")
cat("     • Clinical harm of missed/delayed detection\n")
cat("     • Staff burden of investigating false alarms\n")
cat("     • Average run length (ARL) under null hypothesis\n\n")
cat("4. Calibration matters. An overconfident model gives residuals that are\n")
cat("   systematically non-zero even under no outbreak — biasing CUSUM upward\n")
cat("   and inflating false alarm rates. Always calibrate before CUSUM.\n\n")
cat("5. Ensemble models provide stability. Neither LR nor RF is uniformly\n")
cat("   better; averaging them reduces variance in the risk score stream,\n")
cat("   producing a smoother CUSUM signal with fewer spurious spikes.\n")

cat("\nSimulation complete. All output files saved to: ./", output_dir, "/\n\n", sep="")
