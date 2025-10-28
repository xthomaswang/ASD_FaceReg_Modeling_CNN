# ===================================================================
#           Final Analysis Script (positive_slope outer group)
# ===================================================================

# --- Step 1: Clean Environment & Load Libraries ---
rm(list = ls())

library(plyr)
library(dplyr)
library(stringr)
library(reshape2)
library(Rmisc)
library(rstatix)
library(ggplot2)
library(ggbeeswarm)
library(ggpubr)

# --- Step 2: Setup Paths & Directories ---------------------------------
set.seed(1)

if (!grepl("1_orginal_data_analysis$", getwd()))
  stop("Please setwd() to '1_orginal_data_analysis' first.")

base_dir       <- getwd()
targeted_folder <- "csv_1000_negslope"

# 内层比较因子索引: 1=negative_slope, 2=noise, 3=threshold
compare_index <- 1    

data_dir     <- file.path(base_dir, "EIBResult", targeted_folder, "data")
analysis_dir <- file.path(base_dir, "EIBResult", targeted_folder, "analysis")
image_dir    <- file.path(analysis_dir, "image")

dir.create(analysis_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(image_dir,    showWarnings = FALSE, recursive = TRUE)

# --- Step 3: Load Data --------------------------------------------------
cat("--> Loading data from:", data_dir, "\n")
file_list <- list.files(data_dir, pattern = "trainedACC_1000_\\d+\\.csv$", full.names = TRUE)
if (length(file_list) == 0) stop("No CSV files found in data_dir.")
EIB_acc <- do.call(rbind, lapply(file_list, read.csv, header = TRUE, na.strings = "NA"))
cat("-->", length(file_list), "data files loaded successfully.\n")

# --- Step 4: Parse slope & build factors -------------------------------
cat("--> Parsing slope string ...\n")

# 抽取所有数字，长度不足 4 补 0
num_list <- str_extract_all(EIB_acc$slope, "-?[0-9.]+")
num_mat  <- t(sapply(num_list, function(v) {
  length(v) <- 4
  v[is.na(v)] <- "0"
  v
}))
colnames(num_mat) <- c("positive_slope", "negative_slope", "noise", "threshold")
EIB_acc <- cbind(EIB_acc, num_mat)

# outer group: positive_slope
EIB_acc$pos_group <- as.factor(num_mat[, "positive_slope"])

# inner compare factor
comp_names <- c("negative_slope", "noise", "threshold")
if (compare_index > length(comp_names)) stop("compare_index must be 1–3")
compare_var_name <- comp_names[compare_index]
EIB_acc$compare  <- as.factor(num_mat[, compare_var_name])

cat("--> Outer group: positive_slope; inner compare factor:", compare_var_name, "\n")

# subject ID
EIB_acc$Sub <- as.factor(seq_len(nrow(EIB_acc)))
EIB_acc     <- EIB_acc[, c("Sub", setdiff(names(EIB_acc), "Sub"))]

# --- Step 5: Loop over each positive_slope group -----------------------
sink(file = file.path(analysis_dir, "EIB_all_statistical_tests.txt"))

group_levels <- levels(EIB_acc$pos_group)

for (lvl in group_levels) {
  cat("\n========== positive_slope =", lvl, " ==========\n")
  
  # 1) 过滤本组
  sub <- EIB_acc %>% filter(pos_group == lvl)
  sub$compare <- droplevels(sub$compare)
  
  # 2) epoch 列：每 10 取 1
  epoch_cols     <- grep("epoch_", names(sub), value = TRUE)
  epoch_cols_sel <- epoch_cols[seq(1, length(epoch_cols), by = 10)]
  dat_wide       <- sub[, c("Sub", "compare", epoch_cols_sel)]
  
  # 3) 长格式
  dat_long <- melt(dat_wide, id.vars = c("Sub", "compare"),
                   variable.name = "Epoch", value.name = "Acc")
  dat_long$Epoch <- as.numeric(str_remove_all(dat_long$Epoch, "epoch_"))
  dat_long       <- dat_long[!is.na(dat_long$Acc), ]
  
  # 4) 汇总
  sum_df <- summarySE(dat_long, "Acc", c("compare", "Epoch"),
                      na.rm = TRUE, conf.interval = 0.95)
  
  # ---------- 绘图 ----------
  title_suffix <- paste("(positive_slope =", lvl, ")")
  
  plot1 <- ggplot(sum_df, aes(Epoch, Acc, colour = compare, fill = compare)) +
    geom_line(linewidth = 1) +
    geom_ribbon(aes(ymin = Acc - ci, ymax = Acc + ci, group = compare),
                alpha = 0.3) +
    geom_point(data = dat_long, position = position_jitter(width = 2),
               size = 0.8, alpha = 0.5) +
    scale_color_manual(values = c("#B1DC64", "#FFB363", "#DA6004", "#7FB2D3", "#AE5EFF", "#6E6766")) +
    scale_fill_manual(values  = c("#B1DC64", "#FFB363", "#DA6004", "#7FB2D3", "#AE5EFF", "#6E6766")) +
    theme_classic() +
    labs(title = paste("Accuracy over Epochs", title_suffix),
         color = "Compare", fill = "Compare")
  
  # boxplot at 100/300/500
  box_long <- dat_long[dat_long$Epoch %in% c(100, 300, 500), ]
  plot2 <- ggplot(box_long,
                  aes(as.factor(Epoch), Acc, colour = compare, fill = compare)) +
    geom_boxplot(width = 0.3, outlier.shape = NA,
                 position = position_dodge2(preserve = "single"),
                 linewidth = 0.8, alpha = 0.5) +
    geom_quasirandom(groupOnX = TRUE, width = 0.1, dodge.width = 0.3) +
    scale_color_manual(values = c("#B1DC64", "#FFB363", "#DA6004", "#7FB2D3", "#AE5EFF", "#6E6766")) +
    scale_fill_manual(values  = c("#B1DC64", "#FFB363", "#DA6004", "#7FB2D3", "#AE5EFF", "#6E6766")) +
    theme_classic() +
    labs(title = paste("Accuracy at Specific Epochs", title_suffix),
         color = "Compare", fill = "Compare")
  
  # ---------- 统计检验 ----------
  cat("--- STATISTICS for pos =", lvl, " ---\n")
  
  if (nlevels(dat_long$compare) >= 2) {
    dat_long$Epoch <- as.factor(dat_long$Epoch)
    print(anova_test(dat_long,
                     Acc ~ Epoch * compare,
                     between = compare, within = Epoch,
                     effect.size = "pes", type = 3))
  } else {
    cat("  (skip overall RM-ANOVA: only one compare level)\n")
  }
  
  for (ep in c(100, 300, 500)) {
    cat("\n--- ANOVA @ Epoch", ep, "---\n")
    slice <- subset(dat_long, Epoch == ep)
    if (nlevels(slice$compare) < 2) {
      cat("  (skip: only one compare level)\n")
      next
    }
    ares <- anova_test(Acc ~ compare, data = slice,
                       effect.size = "pes", type = 3)
    print(ares)
    if (ares$p < 0.05)
      print(aov(Acc ~ compare, data = slice) %>% tukey_hsd())
  }
  
  # ---------- 保存图 ----------
  combo_name  <- paste0("EIB_plots_positive_", lvl, ".png")
  single_name <- paste0("EIB_plots_positive_", lvl, "_acc.png")
  
  ggsave(file.path(image_dir, combo_name),
         ggarrange(plot1, plot2, labels = c("A", "B"),
                   ncol = 2, common.legend = TRUE, legend = "bottom"),
         width = 16, height = 7, dpi = 300, bg = "white")
  ggsave(file.path(image_dir, single_name),
         plot1, width = 8, height = 5, dpi = 300, bg = "white")
  cat("--> Plots saved:", combo_name, "and", single_name, "\n")
}

sink()  # close stats log

# --- Step 6: Finalize ---------------------------------------------------
cat("\n***** ALL ANALYSES COMPLETE! *****\n")
cat("Stats saved to", file.path(analysis_dir, "EIB_all_statistical_tests.txt"), "\n")
cat("Plots saved to", image_dir, "\n")
