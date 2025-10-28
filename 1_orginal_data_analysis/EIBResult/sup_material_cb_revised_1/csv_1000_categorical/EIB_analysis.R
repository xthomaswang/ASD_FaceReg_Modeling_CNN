set.seed(1)

##################EIB part
#read in the data
for (i in 0:19) {
  if (i == 0) {
    EIB_acc<-read.csv(paste0("trainedACC_1000_",i,".csv"),header = T, na.strings = "NA")
  } else {
    EIB_acc<-rbind.data.frame(EIB_acc,read.csv(paste0("trainedACC_1000_",i,".csv"),header = T, na.strings = "NA"))
  }
}

##Extract positive slope from tuple (first value)
library(stringr)
# Extract the first number from the slope tuple like "(0.005, 0, 0)" -> "0.005"
EIB_acc$slope_extracted <- str_extract(EIB_acc$slope, "\\d+\\.?\\d*")
EIB_acc$slope_extracted <- as.numeric(EIB_acc$slope_extracted)
EIB_acc$slope <- as.factor(EIB_acc$slope_extracted)
EIB_acc$slope <- factor(EIB_acc$slope, levels = c("0.005","0.05","0.5"))

##create sub ID
EIB_acc$Sub<-paste0("Sub",rep(1:60))
EIB_acc<-EIB_acc[,c(ncol(EIB_acc),1:(ncol(EIB_acc)-1))]
EIB_acc$Sub<-as.factor(EIB_acc$Sub)

# Remove the slope_extracted column as it's no longer needed
EIB_acc$slope_extracted <- NULL

require(Rmisc)
require(reshape2)
require(rstatix)
require(stringr)

##select every 10 epochs
EIB_acc_sel<-EIB_acc[,c(1,2,which((1:ncol(EIB_acc)-2)%%10==1),ncol(EIB_acc))]

EIB_acc_long<-melt(EIB_acc_sel,id.vars = c("Sub","slope"),variable.name = "Epoch",value.name = "Acc")
EIB_acc_long$Epoch<-as.numeric(str_remove_all(EIB_acc_long$Epoch,pattern = "epoch_"))
#EIB_acc_long<-EIB_acc_long[EIB_acc_long$Epoch<=500,]
EIB_acc_sum<-summarySE(EIB_acc_long,measurevar = "Acc",groupvars = c("slope","Epoch"),na.rm = T,conf.interval = 0.95)

# Create analysis directory if it doesn't exist
if (!dir.exists("../analysis")) {
  dir.create("../analysis")
}

write.csv(EIB_acc_sum,file = "../analysis/EIB_acc_mean.csv",
          row.names = F)

require(ggplot2)
require(ggbeeswarm)

plot1<-ggplot(EIB_acc_sum,aes(x = Epoch,y=Acc,color = slope)) +
  geom_line(size = 1) +
  #geom_point(size = 3,aes(shape = slope)) +
  #geom_errorbar(aes(ymin=Acc-se,ymax=Acc+se),size=1) +
  geom_ribbon(aes(ymin=Acc-ci,ymax=Acc+ci,group=slope,fill=slope),alpha=0.3) +
  geom_point(data=EIB_acc_long,aes(x = Epoch, y = Acc, colour = slope), position = position_jitter(width = 2),size = 0.8, alpha = 0.5) +
  scale_color_manual(values = c("#7FB2D3","#B1DC64","#FFB363")) +
  scale_fill_manual(values = c("#7FB2D3","#B1DC64","#FFB363")) +
  theme_classic()

plot2<-ggplot(EIB_acc_long[(EIB_acc_long$Epoch %in% c(100,300,500)),],aes(x = as.factor(Epoch), y = Acc,fill = slope)) +
  geom_boxplot(width=.3, outlier.shape = NA,
               position = position_dodge2(preserve = "single"),size = 0.8,alpha = 0.5) +
  #geom_violin(alpha = 0.5,position = position_dodge(0.3)) +
  geom_quasirandom(aes(colour = slope), groupOnX = TRUE,
                   width=.1, dodge.width = 0.3) +
  scale_color_manual(values = c("#7FB2D3","#B1DC64","#FFB363")) +
  scale_fill_manual(values = c("#7FB2D3","#B1DC64","#FFB363")) +
  scale_x_discrete(name = "Epoch") +
  theme_classic()

# Save plots
ggsave("../analysis/EIB_acc_line_plot.png", plot1, width = 10, height = 6, dpi = 300)
ggsave("../analysis/EIB_acc_box_plot.png", plot2, width = 10, height = 6, dpi = 300)

##ANOVA test
sink(file="../analysis/EIB_acc_tests.txt")
anova_test(EIB_acc_long,Acc~Epoch*slope, between = slope, within = Epoch,effect.size = "pes",type = 3)
EIB_acc_long$Epoch<-as.factor(EIB_acc_long$Epoch)
anova_test(EIB_acc_long,Acc~Epoch*slope, between = slope, within = Epoch,effect.size = "pes",type = 3)

t.test(Acc~slope, data=EIB_acc_long[(EIB_acc_long$Epoch==100 & 
                                       EIB_acc_long$slope!=0.5),],var.equal = T)
anova_test(Acc~slope, data=EIB_acc_long[EIB_acc_long$Epoch==100,],effect.size = "pes",type = 3)
aov(Acc~slope, data=EIB_acc_long[EIB_acc_long$Epoch==100,]) %>% tukey_hsd()
anova_test(Acc~slope, data=EIB_acc_long[EIB_acc_long$Epoch==300,],effect.size = "pes",type = 3)
aov(Acc~slope, data=EIB_acc_long[EIB_acc_long$Epoch==300,]) %>% tukey_hsd()
anova_test(Acc~slope, data=EIB_acc_long[EIB_acc_long$Epoch==500,],effect.size = "pes",type = 3)
aov(Acc~slope, data=EIB_acc_long[EIB_acc_long$Epoch==500,]) %>% tukey_hsd()
#aov(Acc~slope, data=EIB_acc_long[EIB_acc_long$Epoch==750,]) %>% tukey_hsd()
sink()

cat("EIB analysis completed successfully!\n")
cat("Results saved in ../analysis/\n")
cat("- EIB_acc_mean.csv: Summary statistics\n")
cat("- EIB_acc_line_plot.png: Line plot with confidence intervals\n")
cat("- EIB_acc_box_plot.png: Box plots for epochs 100, 300, 500\n")
cat("- EIB_acc_tests.txt: Statistical test results\n") 