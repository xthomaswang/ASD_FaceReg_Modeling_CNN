set.seed(1)

##EIB part
mydata<-read.csv("EIB_inactive_neurons_results.csv",header = T, na.strings = "NA")
colnames(mydata)[1]<-"Sub_ID"
mydata$Sub_ID<-as.factor(paste0("Sub_",mydata$Sub_ID+1,"_",mydata$Slope))
mydata$Slope<-as.factor(mydata$Slope)

require(rstatix)
sink(file = "/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/EIB_InactiveUnit_ANOVA.txt")
anova_test(data=mydata, Inactive_Percentage~Slope*Layer,wid = "Sub_ID", 
           within = "Layer",between = "Slope",effect.size = "pes",type = 3)
##post hoc
aov(Inactive_Percentage~Slope, data = mydata[mydata$Layer=="conv_block1_relu",]) %>% tukey_hsd()
aov(Inactive_Percentage~Slope, data = mydata[mydata$Layer=="conv_block2_relu",]) %>% tukey_hsd()
aov(Inactive_Percentage~Slope, data = mydata[mydata$Layer=="conv_block3_relu",]) %>% tukey_hsd()
sink()
require(Rmisc)
summarySE(data=mydata,measurevar = "Inactive_Percentage",groupvars = c("Layer","Slope"))

EIB_inactive_sum<-summarySE(data=mydata,measurevar = "Inactive_Percentage",groupvars = c("Layer","Slope"))
write.csv(EIB_inactive_sum,
          file = "/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/EIB_InativeUnit_grandmean.csv",row.names = F)
require(ggplot2)
require(ggbeeswarm)

plot1<-ggplot(mydata,aes(x = Layer, y = Inactive_Percentage,fill = Slope)) +
  geom_boxplot(width=.3, outlier.shape = NA,
               position = position_dodge2(preserve = "single"),size = 0.8,alpha = 0.5) +
  geom_quasirandom(aes(colour = Slope), groupOnX = TRUE,
                   width=.1, dodge.width = 0.3) +
  scale_color_manual(values = c("#7FB2D3","#B1DC64","#FFB363")) +
  scale_fill_manual(values = c("#7FB2D3","#B1DC64","#FFB363")) +
  scale_y_continuous(name = "Inactive units percentage (%)") +
  scale_x_discrete(labels=c("conv_block1_relu" = "Conv1", "conv_block2_relu" = "Conv2",
                            "conv_block3_relu" = "Conv3")) +
  theme_classic()

##GSN part
mydata<-read.csv("GSN_inactive_neurons_results.csv",header = T, na.strings = "NA")
colnames(mydata)[1]<-"Sub_ID"
mydata$Sub_ID<-as.factor(paste0("Sub_",mydata$Sub_ID+1,"_",mydata$STD))
mydata$STD<-as.factor(mydata$STD)

require(rstatix)
sink(file = "/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/GSN_InactiveUnit_ANOVA.txt")
anova_test(data=mydata, Inactive_Percentage~STD*Layer,wid = "Sub_ID", 
           within = "Layer",between = "STD",effect.size = "pes",type = 3)
##post hoc
aov(Inactive_Percentage~STD, data = mydata[mydata$Layer=="conv_block1_relu",]) %>% tukey_hsd()
aov(Inactive_Percentage~STD, data = mydata[mydata$Layer=="conv_block2_relu",]) %>% tukey_hsd()
aov(Inactive_Percentage~STD, data = mydata[mydata$Layer=="conv_block3_relu",]) %>% tukey_hsd()
sink()
require(Rmisc)
summarySE(data=mydata,measurevar = "Inactive_Percentage",groupvars = c("Layer","STD"))

GSN_inactive_sum<-summarySE(data=mydata,measurevar = "Inactive_Percentage",groupvars = c("Layer","STD"))
write.csv(GSN_inactive_sum,
          file = "/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/GSN_InativeUnit_grandmean.csv",row.names = F)

require(ggplot2)
require(ggbeeswarm)

plot2<-ggplot(mydata,aes(x = Layer, y = Inactive_Percentage,fill = STD)) +
  geom_boxplot(width=.3, outlier.shape = NA,
               position = position_dodge2(preserve = "single"),size = 0.8,alpha = 0.5) +
  geom_quasirandom(aes(colour = STD), groupOnX = TRUE,
                   width=.1, dodge.width = 0.3) +
  scale_color_manual(values = c("#B1DC64","#FFB363","#DA6004")) +
  scale_fill_manual(values = c("#B1DC64","#FFB363","#DA6004")) +
  scale_y_continuous(name = "Inactive units percentage (%)") +
  scale_x_discrete(labels=c("conv_block1_relu" = "Conv1", "conv_block2_relu" = "Conv2",
                            "conv_block3_relu" = "Conv3")) +
  theme_classic()

library(ggpubr)

ggarrange(plot1,plot2,labels = c("C","D"),
          ncol = 2, nrow = 1)

