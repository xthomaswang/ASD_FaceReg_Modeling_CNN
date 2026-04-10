set.seed(1)

################## EIB Accuracy Analysis — CORnet
################## Conditions: Inhibitated (α=0.5), Balanced (α=1.0), Excitated (α=2.0)

# ============================
# Packages
# ============================
if (!require(jsonlite))   install.packages("jsonlite",   repos = "https://cloud.r-project.org")
if (!require(Rmisc))      install.packages("Rmisc",      repos = "https://cloud.r-project.org")
if (!require(reshape2))   install.packages("reshape2",   repos = "https://cloud.r-project.org")
if (!require(rstatix))    install.packages("rstatix",    repos = "https://cloud.r-project.org")
if (!require(stringr))    install.packages("stringr",    repos = "https://cloud.r-project.org")
if (!require(ggplot2))    install.packages("ggplot2",    repos = "https://cloud.r-project.org")
if (!require(ggbeeswarm)) install.packages("ggbeeswarm", repos = "https://cloud.r-project.org")
if (!require(ggpubr))     install.packages("ggpubr",     repos = "https://cloud.r-project.org")

library(jsonlite)
library(Rmisc)
library(reshape2)
library(rstatix)
library(stringr)
library(ggplot2)
library(ggbeeswarm)
library(ggpubr)

# ============================
# CONFIG
# ============================
EXP_DIR <- "/Users/tuomasier/Library/CloudStorage/GoogleDrive-xthomaswang@gmail.com/My Drive/ASD_FaceReg_Modeling_CNN/results/EIB/cornet/100_20_100_100_2026-02-07_03-11-07"
OUTPUT_DIR <- "/Users/tuomasier/Library/CloudStorage/GoogleDrive-xthomaswang@gmail.com/My Drive/ASD_FaceReg_Modeling_CNN/results/EIB/cornet/100_20_100_100_2026-02-07_03-11-07/output"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

N_RUNS <- 20
EPOCHS <- 100

# ---------------------------------------------------------------
# Use the ACTUAL alpha values as slope: 0.5, 1, 2
# ---------------------------------------------------------------
COND_INFO <- data.frame(
  name   = c("Inhibitated",  "Balanced",  "Excitated"),
  folder = c("inhibitated",  "balanced",  "excitated"),
  slope  = c(0.5,             1.0,         2.0),
  stringsAsFactors = FALSE
)

SLOPE_LEVELS <- c("0.5", "1", "2")
SLOPE_LABELS <- c("Inhibitated (α=0.5)", "Balanced (α=1.0)", "Excitated (α=2.0)")

ACC_KEY <- "val_acc"
BOXPLOT_EPOCHS <- c(10, 50, 100)

# ============================
# 0. Diagnostic
# ============================
cat("=== DIAGNOSTIC ===\n")
sample_path <- file.path(EXP_DIR, "run_0", COND_INFO$folder[1], "history_run0.json")
if (!file.exists(sample_path)) {
  stop(paste("File not found:", sample_path, "\nCheck EXP_DIR."))
}
sample_json <- fromJSON(sample_path)
cat("Keys in history JSON:\n")
for (k in names(sample_json)) {
  v <- sample_json[[k]]
  if (is.list(v) || (is.vector(v) && length(v) > 1)) {
    cat(sprintf("  '%s': length=%d, first=%.4f, last=%.4f\n",
                k, length(v), as.numeric(v[[1]]), as.numeric(v[[length(v)]])))
  } else {
    cat(sprintf("  '%s': %s\n", k, as.character(v)))
  }
}
cat(sprintf("\nUsing '%s' for per-epoch accuracy\n\n", ACC_KEY))

# ============================
# 1. Read all history files
# ============================
cat("Reading history files...\n")

records <- list()
final_records <- list()
rec_idx <- 0
fin_idx <- 0

for (run_idx in 0:(N_RUNS - 1)) {
  for (c_idx in 1:nrow(COND_INFO)) {
    json_path <- file.path(EXP_DIR,
                           paste0("run_", run_idx),
                           COND_INFO$folder[c_idx],
                           paste0("history_run", run_idx, ".json"))
    history <- fromJSON(json_path)
    
    t_acc <- as.numeric(unlist(history[["train_acc"]]))[1:EPOCHS]
    v_acc <- as.numeric(unlist(history[["val_acc"]]))[1:EPOCHS]
    f_acc <- as.numeric(history[["final_test_acc"]])
    
    sub_id <- paste0("Sub", run_idx + 1)
    
    for (ep in 1:EPOCHS) {
      rec_idx <- rec_idx + 1
      records[[rec_idx]] <- data.frame(
        Sub       = sub_id,
        slope     = COND_INFO$slope[c_idx],
        Epoch     = ep,
        Train_Acc = t_acc[ep],
        Val_Acc   = v_acc[ep],
        stringsAsFactors = FALSE
      )
    }
    
    fin_idx <- fin_idx + 1
    final_records[[fin_idx]] <- data.frame(
      Sub            = sub_id,
      slope          = COND_INFO$slope[c_idx],
      Final_Test_Acc = f_acc,
      stringsAsFactors = FALSE
    )
  }
}

EIB_long  <- do.call(rbind.data.frame, records)
EIB_final <- do.call(rbind.data.frame, final_records)

EIB_long$Sub   <- as.factor(EIB_long$Sub)
EIB_long$slope <- factor(EIB_long$slope, levels = SLOPE_LEVELS)

EIB_final$Sub   <- as.factor(EIB_final$Sub)
EIB_final$slope <- factor(EIB_final$slope, levels = SLOPE_LEVELS)

cat(sprintf("Loaded: %d rows (%d runs × %d conditions × %d epochs)\n",
            nrow(EIB_long), N_RUNS, nrow(COND_INFO), EPOCHS))
cat(sprintf("Final test: %d rows\n\n", nrow(EIB_final)))

# ============================
# 2. Summaries
# ============================
train_sum <- summarySE(EIB_long, measurevar = "Train_Acc",
                       groupvars = c("slope", "Epoch"),
                       na.rm = TRUE, conf.interval = 0.95)
val_sum   <- summarySE(EIB_long, measurevar = "Val_Acc",
                       groupvars = c("slope", "Epoch"),
                       na.rm = TRUE, conf.interval = 0.95)

write.csv(train_sum, file = file.path(OUTPUT_DIR, "EIB_train_acc_summary.csv"), row.names = FALSE)
write.csv(val_sum,   file = file.path(OUTPUT_DIR, "EIB_val_acc_summary.csv"),   row.names = FALSE)

# ============================
# 3. Plots
# ============================

## A: Training accuracy
plotA <- ggplot(train_sum, aes(x = Epoch, y = Train_Acc, color = slope)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = Train_Acc - ci, ymax = Train_Acc + ci,
                  group = slope, fill = slope), alpha = 0.3) +
  geom_point(data = EIB_long,
             aes(x = Epoch, y = Train_Acc, colour = slope),
             position = position_jitter(width = 0.3),
             size = 0.3, alpha = 0.3) +
  scale_color_manual(values = c("#7FB2D3", "#B1DC64", "#FFB363"),
                     labels = SLOPE_LABELS) +
  scale_fill_manual(values = c("#7FB2D3", "#B1DC64", "#FFB363"),
                    labels = SLOPE_LABELS) +
  labs(x = "Epoch", y = "Training Accuracy",
       color = "Condition", fill = "Condition") +
  theme_classic()

## B: Validation accuracy
plotB <- ggplot(val_sum, aes(x = Epoch, y = Val_Acc, color = slope)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = Val_Acc - ci, ymax = Val_Acc + ci,
                  group = slope, fill = slope), alpha = 0.3) +
  geom_point(data = EIB_long,
             aes(x = Epoch, y = Val_Acc, colour = slope),
             position = position_jitter(width = 0.3),
             size = 0.3, alpha = 0.3) +
  scale_color_manual(values = c("#7FB2D3", "#B1DC64", "#FFB363"),
                     labels = SLOPE_LABELS) +
  scale_fill_manual(values = c("#7FB2D3", "#B1DC64", "#FFB363"),
                    labels = SLOPE_LABELS) +
  labs(x = "Epoch", y = "Validation Accuracy",
       color = "Condition", fill = "Condition") +
  theme_classic()

## C: Val acc boxplot at selected epochs
plotC <- ggplot(EIB_long[EIB_long$Epoch %in% BOXPLOT_EPOCHS, ],
                aes(x = as.factor(Epoch), y = Val_Acc, fill = slope)) +
  geom_boxplot(width = 0.3, outlier.shape = NA,
               position = position_dodge2(preserve = "single"),
               size = 0.8, alpha = 0.5) +
  geom_quasirandom(aes(colour = slope), groupOnX = TRUE,
                   width = 0.1, dodge.width = 0.3) +
  scale_color_manual(values = c("#7FB2D3", "#B1DC64", "#FFB363"),
                     labels = SLOPE_LABELS) +
  scale_fill_manual(values = c("#7FB2D3", "#B1DC64", "#FFB363"),
                    labels = SLOPE_LABELS) +
  scale_x_discrete(name = "Epoch") +
  labs(y = "Validation Accuracy", color = "Condition", fill = "Condition") +
  theme_classic()

## D: Final test accuracy
plotD <- ggplot(EIB_final, aes(x = slope, y = Final_Test_Acc, fill = slope)) +
  geom_boxplot(width = 0.3, outlier.shape = NA,
               size = 0.8, alpha = 0.5) +
  geom_quasirandom(aes(colour = slope), width = 0.1) +
  scale_color_manual(values = c("#7FB2D3", "#B1DC64", "#FFB363"),
                     labels = SLOPE_LABELS) +
  scale_fill_manual(values = c("#7FB2D3", "#B1DC64", "#FFB363"),
                    labels = SLOPE_LABELS) +
  scale_x_discrete(labels = SLOPE_LABELS) +
  labs(x = "Condition", y = "Final Test Accuracy",
       color = "Condition", fill = "Condition") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 15, hjust = 1))

# ============================
# 4. ANOVA
# ============================
sink(file = file.path(OUTPUT_DIR, "EIB_acc_tests.txt"))

cat("============================================================\n")
cat("TRAINING ACCURACY (20 runs per slope×epoch)\n")
cat("============================================================\n\n")

anova_epochs <- c(1, seq(11, 91, by = 10), 100)
train_anova <- EIB_long[EIB_long$Epoch %in% anova_epochs, ]
train_anova$Epoch_f <- as.factor(train_anova$Epoch)

cat("=== Omnibus ANOVA (Epoch factor, selected epochs) ===\n")
print(anova_test(Train_Acc ~ Epoch_f * slope,
                 data = train_anova,
                 between = slope, within = Epoch_f,
                 effect.size = "pes", type = 3))

cat("\n============================================================\n")
cat("VALIDATION ACCURACY (20 runs per slope×epoch)\n")
cat("============================================================\n\n")

val_anova <- EIB_long[EIB_long$Epoch %in% anova_epochs, ]
val_anova$Epoch_f <- as.factor(val_anova$Epoch)

cat("=== Omnibus ANOVA (Epoch factor, selected epochs) ===\n")
print(anova_test(Val_Acc ~ Epoch_f * slope,
                 data = val_anova,
                 between = slope, within = Epoch_f,
                 effect.size = "pes", type = 3))

cat("\n=== Per-epoch Val Acc ANOVAs + Tukey ===\n")
for (ep in BOXPLOT_EPOCHS) {
  cat(paste0("\n--- Epoch ", ep, " ---\n"))
  sub_data <- EIB_long[EIB_long$Epoch == ep, ]
  sub_data$slope <- droplevels(sub_data$slope)
  if (nlevels(sub_data$slope) >= 2) {
    print(anova_test(Val_Acc ~ slope, data = sub_data,
                     effect.size = "pes", type = 3))
    print(aov(Val_Acc ~ slope, data = sub_data) %>% tukey_hsd())
  }
}

cat("\n============================================================\n")
cat("FINAL TEST ACCURACY (single evaluation per run)\n")
cat("============================================================\n\n")

print(anova_test(Final_Test_Acc ~ slope, data = EIB_final,
                 effect.size = "pes", type = 3))
print(aov(Final_Test_Acc ~ slope, data = EIB_final) %>% tukey_hsd())

sink()
cat("Statistical tests saved.\n")

# ============================
# 5. Save figures
# ============================
combined <- ggarrange(plotA, plotB, plotC, plotD,
                      labels = c("A", "B", "C", "D"),
                      ncol = 2, nrow = 2,
                      common.legend = TRUE, legend = "bottom")

ggsave(file.path(OUTPUT_DIR, "EIB_acc_plots.pdf"),
       combined, width = 14, height = 10)
ggsave(file.path(OUTPUT_DIR, "EIB_acc_plots.png"),
       combined, width = 14, height = 10, dpi = 300)

cat("\nDone! Outputs in:", OUTPUT_DIR, "\n")