set.seed(1)

################## EIB NRS Barplot — CORnet
##################
################## Reads nrs_within_between.csv (pre-extracted by extract_nrs.py)
################## NO .npy loading in R. Instant, crash-free.
##################
################## Produces:
##################   A: Train RSA barplot (within vs between)
##################   B: Test RSA barplot (within vs between)

# ============================
# Packages
# ============================
if (!require(Rmisc))      install.packages("Rmisc",      repos = "https://cloud.r-project.org")
if (!require(ggplot2))    install.packages("ggplot2",    repos = "https://cloud.r-project.org")
if (!require(ggbeeswarm)) install.packages("ggbeeswarm", repos = "https://cloud.r-project.org")
if (!require(rstatix))    install.packages("rstatix",    repos = "https://cloud.r-project.org")
if (!require(ggpubr))     install.packages("ggpubr",     repos = "https://cloud.r-project.org")

library(Rmisc)
library(ggplot2)
library(ggbeeswarm)
library(rstatix)
library(ggpubr)

# ============================
# CONFIG
# ============================
OUTPUT_DIR <- "/Users/tuomasier/Library/CloudStorage/GoogleDrive-xthomaswang@gmail.com/My Drive/ASD_FaceReg_Modeling_CNN/results/EIB/cornet/100_20_100_100_2026-02-07_03-11-07/output"
EPOCHS <- 100

SLOPE_LEVELS <- c("0.5", "1", "2")
SLOPE_LABELS <- c("Inhibitated (α=0.5)", "Balanced (α=1.0)", "Excitated (α=2.0)")

# ============================
# 1. Read the pre-extracted CSV
# ============================
csv_path <- file.path(OUTPUT_DIR, "nrs_within_between.csv")
if (!file.exists(csv_path)) {
  stop(paste("CSV not found:", csv_path,
             "\nRun extract_nrs.py first (in Colab or terminal)."))
}

df <- read.csv(csv_path, stringsAsFactors = FALSE)
df$Sub   <- as.factor(df$Sub)
df$slope <- factor(df$slope, levels = SLOPE_LEVELS)
df$Type  <- as.factor(df$Type)
df$split <- as.factor(df$split)

cat(sprintf("Loaded %d rows from CSV\n", nrow(df)))
cat(sprintf("Splits: %s\n", paste(levels(df$split), collapse = ", ")))
cat(sprintf("Slopes: %s\n\n", paste(levels(df$slope), collapse = ", ")))

# ============================
# 2. ANOVA — separate for train and test
# ============================
sink(file = file.path(OUTPUT_DIR, paste0("EIB_repres_", EPOCHS, "_ANOVA.txt")))

for (sp in c("train", "test")) {
  cat(paste0("============================================================\n"))
  cat(paste0(toupper(sp), " RSA\n"))
  cat(paste0("============================================================\n\n"))
  
  df_sp <- df[df$split == sp, ]
  
  cat("=== Mixed ANOVA: slope × Type ===\n")
  print(anova_test(corr ~ slope * Type + Error(Sub / Type),
                   data = df_sp, effect.size = "pes", type = 3))
  
  cat("\n=== Post-hoc: Within by slope ===\n")
  print(aov(corr ~ slope, data = df_sp[df_sp$Type == "Within", ]) %>% tukey_hsd())
  
  cat("\n=== Post-hoc: Between by slope ===\n")
  print(aov(corr ~ slope, data = df_sp[df_sp$Type == "Between", ]) %>% tukey_hsd())
  
  cat("\n\n")
}

sink()
cat("ANOVA saved.\n")

# Save CSVs
for (sp in c("train", "test")) {
  df_sp <- df[df$split == sp, ]
  grand <- summarySE(df_sp, measurevar = "corr",
                     groupvars = c("slope", "Type"), na.rm = TRUE)
  write.csv(df_sp, file = file.path(OUTPUT_DIR, paste0("EIB_repres_", EPOCHS, "_", sp, "_individual.csv")), row.names = FALSE)
  write.csv(grand, file = file.path(OUTPUT_DIR, paste0("EIB_repres_", EPOCHS, "_", sp, "_grandmean.csv")), row.names = FALSE)
}

# ============================
# 3. Barplots — same style as Figure 3
# ============================
make_barplot <- function(df_sp, title_str) {
  p <- ggplot(df_sp, aes(x = slope, y = corr, fill = Type)) +
    geom_boxplot(width = 0.3, outlier.shape = NA,
                 position = position_dodge2(preserve = "single"),
                 size = 0.8, alpha = 0.5) +
    geom_quasirandom(aes(colour = Type), groupOnX = TRUE,
                     width = 0.1, dodge.width = 0.3) +
    scale_color_manual(values = c("#6CC6D8", "#EE7564")) +
    scale_fill_manual(values = c("#B5E5EF", "#F8C9C4")) +
    scale_y_continuous(name = "NRS (r)") +
    scale_x_discrete(labels = SLOPE_LABELS) +
    ggtitle(title_str) +
    theme_classic() +
    theme(plot.title = element_text(color = "gray20", size = 8,
                                    face = "bold", hjust = 0.5),
          plot.margin = margin(t = 1, r = 1, b = 1, l = 1, "cm"),
          legend.title = element_text(size = 10),
          legend.text = element_text(size = 8))
  return(p)
}

plotA <- make_barplot(df[df$split == "train", ],
                      paste0("Train RSA (Epoch = ", EPOCHS, ")"))
plotB <- make_barplot(df[df$split == "test", ],
                      paste0("Test RSA (Epoch = ", EPOCHS, ")"))

## Combined figure
combined <- ggarrange(plotA, plotB,
                      nrow = 1, ncol = 2,
                      labels = c("A", "B"),
                      common.legend = TRUE, legend = "right")

ggsave(file.path(OUTPUT_DIR, paste0("EIB_repres_", EPOCHS, "_barplots.pdf")),
       combined, width = 12, height = 5)
ggsave(file.path(OUTPUT_DIR, paste0("EIB_repres_", EPOCHS, "_barplots.png")),
       combined, width = 12, height = 5, dpi = 300)

## Also save individual plots
ggsave(file.path(OUTPUT_DIR, paste0("EIB_repres_", EPOCHS, "_train_barplot.pdf")),
       plotA, width = 6, height = 5)
ggsave(file.path(OUTPUT_DIR, paste0("EIB_repres_", EPOCHS, "_test_barplot.pdf")),
       plotB, width = 6, height = 5)

cat("\nDone! Outputs in:", OUTPUT_DIR, "\n")