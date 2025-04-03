set.seed(1)

##################EIB part
#read in the data
setwd("./EIBResult/csv_1000/trainedACC_CSV/")
for (i in 0:19) {
  if (i == 0) {
    EIB_acc<-read.csv(paste0("trainedACC_1000_",i,".csv"),header = T, na.strings = "NA")
  } else {
    EIB_acc<-rbind.data.frame(EIB_acc,read.csv(paste0("trainedACC_1000_",i,".csv"),header = T, na.strings = "NA"))
  }
}

EIB_acc$slope<-as.factor(EIB_acc$slope)
EIB_acc$slope<-factor(EIB_acc$slope,levels = c("0.005","0.05","0.5"))

##create sub ID
EIB_acc$Sub<-paste0("Sub",rep(1:60))
EIB_acc<-EIB_acc[,c(ncol(EIB_acc),1:(ncol(EIB_acc)-1))]
EIB_acc$Sub<-as.factor(EIB_acc$Sub)

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
write.csv(EIB_acc_sum,file = "/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/EIB_acc_mean.csv",
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


##ANOVA test
sink(file="/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/EIB_acc_tests.txt")
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
##############################################
##################IN part
#read in the data
setwd("/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/GSNResult/csv1000/trainedACC_CSV/")

for (i in 0:19) {
  if (i == 0) {
    GSN_acc<-read.csv(paste0("trainedACC_1000_",i,".csv"),header = T, na.strings = "NA")
  } else {
    GSN_acc<-rbind.data.frame(GSN_acc,read.csv(paste0("trainedACC_1000_",i,".csv"),header = T, na.strings = "NA"))
  }
}

##change colname
colnames(GSN_acc)[1]<-"noise"
GSN_acc$noise<-as.factor(GSN_acc$noise)
GSN_acc$noise<-factor(GSN_acc$noise,levels = c("0","0.5","5"))

##create sub ID
GSN_acc$Sub<-paste0("Sub",rep(0:59))
GSN_acc<-GSN_acc[,c(ncol(GSN_acc),1:(ncol(GSN_acc)-1))]
GSN_acc$Sub<-as.factor(GSN_acc$Sub)

##select every 10 epochs
GSN_acc_sel<-GSN_acc[,c(1,2,which((1:ncol(GSN_acc)-2)%%10==1),ncol(GSN_acc))]

GSN_acc_long<-melt(GSN_acc_sel,id.vars = c("Sub","noise"),variable.name = "Epoch",value.name = "Acc")
GSN_acc_long$Epoch<-as.numeric(str_remove_all(GSN_acc_long$Epoch,pattern = "epoch_"))
#GSN_acc_long<-GSN_acc_long[GSN_acc_long$Epoch<=500,]
GSN_acc_sum<-summarySE(GSN_acc_long,measurevar = "Acc",groupvars = c("noise","Epoch"),na.rm = T,conf.interval = 0.95)
write.csv(GSN_acc_sum,file = "/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/GSN_acc_mean.csv",
          row.names = F)


plot3<-ggplot(GSN_acc_sum,aes(x = Epoch,y=Acc,color = noise)) +
  geom_line(size = 1) +
  #geom_point(size = 3,aes(shape = noise)) +
  #geom_errorbar(aes(ymin=Acc-se,ymax=Acc+se),size=1) +
  geom_ribbon(aes(ymin=Acc-ci,ymax=Acc+ci,group=noise,fill=noise),alpha=0.3) +
  geom_point(data=GSN_acc_long,aes(x = Epoch, y = Acc, colour = noise), position = position_jitter(width = 2),size = 0.8, alpha = 0.5) +
  scale_color_manual(values = c("#B1DC64","#FFB363","#DA6004")) +
  scale_fill_manual(values = c("#B1DC64","#FFB363","#DA6004")) +
  theme_classic()

plot4<-ggplot(GSN_acc_long[(EIB_acc_long$Epoch %in% c(100,300,500)),],aes(x = as.factor(Epoch), y = Acc,fill = noise)) +
  geom_boxplot(width=.3, outlier.shape = NA,
               position = position_dodge2(preserve = "single"),size = 0.8,alpha = 0.5) +
  #geom_violin(alpha = 0.5,position = position_dodge(0.3)) +
  geom_quasirandom(aes(colour = noise), groupOnX = TRUE,
                   width=.1, dodge.width = 0.3) +
  scale_color_manual(values = c("#B1DC64","#FFB363","#DA6004")) +
  scale_fill_manual(values = c("#B1DC64","#FFB363","#DA6004")) +
  scale_x_discrete(name = "Epoch") +
  theme_classic()


##ANOVA test
sink(file="/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/GSN_acc_tests.txt")
anova_test(GSN_acc_long,Acc~Epoch*noise, between = noise, within = Epoch,effect.size = "pes",type = 3)
GSN_acc_long$Epoch<-as.factor(GSN_acc_long$Epoch)
anova_test(GSN_acc_long,Acc~Epoch*noise, between = noise, within = Epoch,effect.size = "pes",type = 3)

t.test(Acc~noise, data=GSN_acc_long[(GSN_acc_long$Epoch==100 &
                                       GSN_acc_long$noise!=5),],var.equal = T)
anova_test(Acc~noise, data=GSN_acc_long[GSN_acc_long$Epoch==100,],effect.size = "pes",type = 3)
aov(Acc~noise, data=GSN_acc_long[GSN_acc_long$Epoch==100,]) %>% tukey_hsd()
anova_test(Acc~noise, data=GSN_acc_long[GSN_acc_long$Epoch==300,],effect.size = "pes",type = 3)
aov(Acc~noise, data=GSN_acc_long[GSN_acc_long$Epoch==300,]) %>% tukey_hsd()
anova_test(Acc~noise, data=GSN_acc_long[GSN_acc_long$Epoch==500,],effect.size = "pes",type = 3)
aov(Acc~noise, data=GSN_acc_long[GSN_acc_long$Epoch==500,]) %>% tukey_hsd()
#aov(Acc~noise, data=GSN_acc_long[GSN_acc_long$Epoch==750,]) %>% tukey_hsd()
sink()
##Combining figures
#install.packages("ggpubr")
library(ggpubr)

ggarrange(plot1,plot2,plot3,plot4, labels = c("A","B","C","D"),
          ncol = 2, nrow = 2)
