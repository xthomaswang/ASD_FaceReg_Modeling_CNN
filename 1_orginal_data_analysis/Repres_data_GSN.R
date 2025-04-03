set.seed(1)

##################GSN part
###500 epoch
#read in the data & ###calculate the within-person vs. between-person correlation
setwd("./GSNResult/500/cor_output_CSV/")
require(magic)
##construct diagnal matrix
dia_matrix<-matrix(T, 4, 4)
dia_matrix_block<-adiag(dia_matrix,dia_matrix,dia_matrix,dia_matrix,dia_matrix,dia_matrix,dia_matrix,dia_matrix,dia_matrix,dia_matrix)
for (j in c(0,0.5,5)) {
  for (i in 0:19) {
    cor_temp<-read.csv(paste0("cor_output_500_",j,"_",i,".csv"),header = T, na.strings = "NA")[,-1]
    ###remove the first image of all P identity (which is used for testing)
    cor_temp<-cor_temp[-which(1:nrow(cor_temp)%%5==1),-which(1:ncol(cor_temp)%%5==1)]
    if (i == 0) {
      GSN_mean_repres<-cor_temp
      assign(paste0("GSN_repres_500_within_noise_",j), cor_temp[(lower.tri(cor_temp) & dia_matrix_block)])
      assign(paste0("GSN_repres_500_between_noise_",j), cor_temp[(lower.tri(cor_temp) & !dia_matrix_block)])
    } else {
      GSN_mean_repres<-GSN_mean_repres+cor_temp
      assign(paste0("GSN_repres_500_within_noise_",j), rbind.data.frame(get(paste0("GSN_repres_500_within_noise_",j)),
                                                                        cor_temp[(lower.tri(cor_temp) & dia_matrix_block)]))
      assign(paste0("GSN_repres_500_between_noise_",j), rbind.data.frame(get(paste0("GSN_repres_500_between_noise_",j)),
                                                                         cor_temp[(lower.tri(cor_temp) & !dia_matrix_block)]))
    }
  }
  assign(paste0("GSN_mean_repres_noise_",j),GSN_mean_repres/20)
}
colnames(GSN_repres_500_within_noise_0)<-paste0("wcor",1:dim(GSN_repres_500_within_noise_0)[2])
colnames(GSN_repres_500_within_noise_0.5)<-paste0("wcor",1:dim(GSN_repres_500_within_noise_0.5)[2])
colnames(GSN_repres_500_within_noise_5)<-paste0("wcor",1:dim(GSN_repres_500_within_noise_5)[2])

colnames(GSN_repres_500_between_noise_0)<-paste0("bcor",1:dim(GSN_repres_500_between_noise_0)[2])
colnames(GSN_repres_500_between_noise_0.5)<-paste0("bcor",1:dim(GSN_repres_500_between_noise_0.5)[2])
colnames(GSN_repres_500_between_noise_5)<-paste0("bcor",1:dim(GSN_repres_500_between_noise_5)[2])

##make heat plots

colnames(GSN_mean_repres_noise_0)<-paste0("P",rep(1:10,each=4),":",rep(1:4,10))
rownames(GSN_mean_repres_noise_0)<-paste0("P",rep(1:10,each=4),":",rep(1:4,10))

colnames(GSN_mean_repres_noise_0.5)<-paste0("P",rep(1:10,each=4),":",rep(1:4,10))
rownames(GSN_mean_repres_noise_0.5)<-paste0("P",rep(1:10,each=4),":",rep(1:4,10))

colnames(GSN_mean_repres_noise_5)<-paste0("P",rep(1:10,each=4),":",rep(1:4,10))
rownames(GSN_mean_repres_noise_5)<-paste0("P",rep(1:10,each=4),":",rep(1:4,10))

require(ggcorrplot)
corplot0_500<-ggcorrplot(as.matrix(GSN_mean_repres_noise_0),method = "square",
                            outline.color = "white",tl.cex = 3, tl.col = "gray35",
                            title = "SD(Noise) = 0",
                            legend.title = "",
                            ggtheme = theme(
                              plot.title = element_text(color="gray20", size=8, face="bold",
                                                        hjust = 0.5,vjust=-1),
                              plot.margin = margin(0,0,0,0, 'cm'),
                              axis.ticks = element_blank(), 
                              legend.position = "bottom",
                              legend.justification = "center",
                              legend.text = element_text(color = "gray20", size = 6),
                              legend.key.height= unit(5, 'pt'),
                              legend.key.width= unit(20, 'pt')
                            ),
                            col=c("#3CC4E2","#B5E0E6","#EB5D25")) +
  geom_rect(xmin=0.5,xmax=4.5,ymin=0.5,ymax=4.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=4.5,xmax=8.5,ymin=4.5,ymax=8.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=8.5,xmax=12.5,ymin=8.5,ymax=12.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=12.5,xmax=16.5,ymin=12.5,ymax=16.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=16.5,xmax=20.5,ymin=16.5,ymax=20.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=20.5,xmax=24.5,ymin=20.5,ymax=24.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=24.5,xmax=28.5,ymin=24.5,ymax=28.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=28.5,xmax=32.5,ymin=28.5,ymax=32.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=32.5,xmax=36.5,ymin=32.5,ymax=36.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=36.5,xmax=40.5,ymin=36.5,ymax=40.5,fill=NA,color="#B455A5")

corplot05_500<-ggcorrplot(as.matrix(GSN_mean_repres_noise_0.5),method = "square",
                           outline.color = "white",tl.cex = 3, tl.col = "gray35",
                           title = "SD(noise) = 0.5",
                           legend.title = "",
                           ggtheme = theme(
                             plot.title = element_text(color="gray20", size=8, face="bold",
                                                       hjust = 0.5,vjust=-1),
                             plot.margin = margin(0,0,0,0, 'cm'),
                             axis.ticks = element_blank(), 
                             legend.position = "bottom",
                             legend.justification = "center",
                             legend.text = element_text(color = "gray20", size = 6),
                             legend.key.height= unit(5, 'pt'),
                             legend.key.width= unit(20, 'pt')
                           ),
                           col=c("#3CC4E2","#B5E0E6","#EB5D25")) +
  geom_rect(xmin=0.5,xmax=4.5,ymin=0.5,ymax=4.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=4.5,xmax=8.5,ymin=4.5,ymax=8.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=8.5,xmax=12.5,ymin=8.5,ymax=12.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=12.5,xmax=16.5,ymin=12.5,ymax=16.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=16.5,xmax=20.5,ymin=16.5,ymax=20.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=20.5,xmax=24.5,ymin=20.5,ymax=24.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=24.5,xmax=28.5,ymin=24.5,ymax=28.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=28.5,xmax=32.5,ymin=28.5,ymax=32.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=32.5,xmax=36.5,ymin=32.5,ymax=36.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=36.5,xmax=40.5,ymin=36.5,ymax=40.5,fill=NA,color="#B455A5")

corplot5_500<-ggcorrplot(as.matrix(GSN_mean_repres_noise_5),method = "square",
                          outline.color = "white",tl.cex = 3, tl.col = "gray35",
                          title = "SD(noise) = 5",
                          legend.title = "",
                          ggtheme = theme(
                            plot.title = element_text(color="gray20", size=8, face="bold",
                                                      hjust = 0.5,vjust=-1),
                            plot.margin = margin(0,0,0,0, 'cm'),
                            axis.ticks = element_blank(), 
                            legend.position = "bottom",
                            legend.justification = "center",
                            legend.text = element_text(color = "gray20", size = 6),
                            legend.key.height= unit(5, 'pt'),
                            legend.key.width= unit(20, 'pt')
                          ),
                          col=c("#3CC4E2","#B5E0E6","#EB5D25")) +
  geom_rect(xmin=0.5,xmax=4.5,ymin=0.5,ymax=4.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=4.5,xmax=8.5,ymin=4.5,ymax=8.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=8.5,xmax=12.5,ymin=8.5,ymax=12.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=12.5,xmax=16.5,ymin=12.5,ymax=16.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=16.5,xmax=20.5,ymin=16.5,ymax=20.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=20.5,xmax=24.5,ymin=20.5,ymax=24.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=24.5,xmax=28.5,ymin=24.5,ymax=28.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=28.5,xmax=32.5,ymin=28.5,ymax=32.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=32.5,xmax=36.5,ymin=32.5,ymax=36.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=36.5,xmax=40.5,ymin=36.5,ymax=40.5,fill=NA,color="#B455A5")

#combine data
GSN_repres_500_within_noise_0$noise<-0
GSN_repres_500_within_noise_0.5$noise<-0.5
GSN_repres_500_within_noise_5$noise<-5

GSN_repres_500_between_noise_0$noise<-0
GSN_repres_500_between_noise_0.5$noise<-0.5
GSN_repres_500_between_noise_5$noise<-5

GSN_repres_500_within<-rbind.data.frame(GSN_repres_500_within_noise_0,GSN_repres_500_within_noise_0.5,GSN_repres_500_within_noise_5)
GSN_repres_500_between<-rbind.data.frame(GSN_repres_500_between_noise_0,GSN_repres_500_between_noise_0.5,GSN_repres_500_between_noise_5)

GSN_repres_500_within$Sub<-paste0("Sub",1:nrow(GSN_repres_500_within))
GSN_repres_500_between$Sub<-paste0("Sub",1:nrow(GSN_repres_500_between))
GSN_repres_500_within$Type<-"Within"
GSN_repres_500_between$Type<-"Between"

require(Rmisc)
require(reshape2)
require(ggplot2)
require(ggbeeswarm)

GSN_repres_500_within_long<-melt(GSN_repres_500_within,id.vars = c("Sub","noise","Type"),variable.name = "pair",value.name = "corr")
GSN_repres_500_betw_long<-melt(GSN_repres_500_between,id.vars = c("Sub","noise","Type"),variable.name = "pair",value.name = "corr")

GSN_repres_500_long<-rbind.data.frame(GSN_repres_500_within_long,GSN_repres_500_betw_long)
GSN_repres_500_long$Sub<-as.factor(GSN_repres_500_long$Sub)
GSN_repres_500_long$noise<-as.factor(GSN_repres_500_long$noise)
GSN_repres_500_long$noise<-factor(GSN_repres_500_long$noise,levels = c("0","0.5","5"))

##ANOVA
require(rstatix)
sink(file = "/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/GSN_repres_500_ANOVA.txt")
GSN_repres_500_sum<-summarySE(GSN_repres_500_long,measurevar = "corr",groupvars = c("Sub","noise","Type"),na.rm = T)
anova_test(corr~noise*Type+Error(Sub/Type),data=GSN_repres_500_sum,effect.size = "pes",type = 3)
##post hoc
aov(corr~noise, data = GSN_repres_500_sum[GSN_repres_500_sum$Type=="Within",]) %>% tukey_hsd()
aov(corr~noise, data = GSN_repres_500_sum[GSN_repres_500_sum$Type=="Between",]) %>% tukey_hsd()
sink()
##plot
GSN_repres_500_grand_sum<-summarySE(GSN_repres_500_sum[,c(1,2,3,5)],measurevar = "corr",groupvars = c("noise","Type"),na.rm = T)

write.csv(GSN_repres_500_sum,
          file = "/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/GSN_repres_500_individual.csv",row.names = F)
write.csv(GSN_repres_500_grand_sum,
          file = "/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/GSN_repres_500_grandmean.csv",row.names = F)


barplot_500<-ggplot(GSN_repres_500_sum,aes(x = noise, y = corr,fill = Type)) +
  geom_boxplot(width=.3, outlier.shape = NA,
               position = position_dodge2(preserve = "single"),size = 0.8,alpha = 0.5) +
  #geom_violin(alpha = 0.5,position = position_dodge(0.3)) +
  geom_quasirandom(aes(colour = Type), groupOnX = TRUE,
                   width=.1, dodge.width = 0.3) +
  scale_color_manual(values = c("#6CC6D8","#EE7564")) +
  scale_fill_manual(values = c("#B5E5EF","#F8C9C4")) +
  scale_y_continuous(name = "NRS (r)",limits = c(-0.2,1)) +
  ggtitle("Epoch = 500") +
  theme_classic()
barplot_500<-barplot_500 + 
  theme(plot.title = element_text(color="gray20", size=8, face="bold",hjust = 0.5),
        plot.margin = margin(0,0,0,0, 'cm'),
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8))

###750 epoch
#read in the data & ###calculate the within-person vs. between-person correlation
setwd("/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/GSNResult/750/cor_output_CSV/")
require(magic)
##construct diagnal matrix
dia_matrix<-matrix(T, 4, 4)
dia_matrix_block<-adiag(dia_matrix,dia_matrix,dia_matrix,dia_matrix,dia_matrix,dia_matrix,dia_matrix,dia_matrix,dia_matrix,dia_matrix)
for (j in c(5,0.5,0)) {
  for (i in 0:19) {
    cor_temp<-read.csv(paste0("cor_output_750_",j,"_",i,".csv"),header = T, na.strings = "NA")[,-1]
    ###remove the first image of all P identity (which is used for testing)
    cor_temp<-cor_temp[-which(1:nrow(cor_temp)%%5==1),-which(1:ncol(cor_temp)%%5==1)]
    if (i == 0) {
      GSN_mean_repres<-cor_temp
      assign(paste0("GSN_repres_750_within_noise_",j), cor_temp[(lower.tri(cor_temp) & dia_matrix_block)])
      assign(paste0("GSN_repres_750_between_noise_",j), cor_temp[(lower.tri(cor_temp) & !dia_matrix_block)])
    } else {
      GSN_mean_repres<-GSN_mean_repres+cor_temp
      assign(paste0("GSN_repres_750_within_noise_",j), rbind.data.frame(get(paste0("GSN_repres_750_within_noise_",j)),
                                                                         cor_temp[(lower.tri(cor_temp) & dia_matrix_block)]))
      assign(paste0("GSN_repres_750_between_noise_",j), rbind.data.frame(get(paste0("GSN_repres_750_between_noise_",j)),
                                                                          cor_temp[(lower.tri(cor_temp) & !dia_matrix_block)]))
    }
  }
  assign(paste0("GSN_mean_repres_noise_",j),GSN_mean_repres/20)
}
colnames(GSN_repres_750_within_noise_0)<-paste0("wcor",1:dim(GSN_repres_750_within_noise_0)[2])
colnames(GSN_repres_750_within_noise_0.5)<-paste0("wcor",1:dim(GSN_repres_750_within_noise_0.5)[2])
colnames(GSN_repres_750_within_noise_5)<-paste0("wcor",1:dim(GSN_repres_750_within_noise_5)[2])

colnames(GSN_repres_750_between_noise_0)<-paste0("bcor",1:dim(GSN_repres_750_between_noise_0)[2])
colnames(GSN_repres_750_between_noise_0.5)<-paste0("bcor",1:dim(GSN_repres_750_between_noise_0.5)[2])
colnames(GSN_repres_750_between_noise_5)<-paste0("bcor",1:dim(GSN_repres_750_between_noise_5)[2])

##make heat plots

colnames(GSN_mean_repres_noise_0)<-paste0("P",rep(1:10,each=4),":",rep(1:4,10))
rownames(GSN_mean_repres_noise_0)<-paste0("P",rep(1:10,each=4),":",rep(1:4,10))

colnames(GSN_mean_repres_noise_0.5)<-paste0("P",rep(1:10,each=4),":",rep(1:4,10))
rownames(GSN_mean_repres_noise_0.5)<-paste0("P",rep(1:10,each=4),":",rep(1:4,10))

colnames(GSN_mean_repres_noise_5)<-paste0("P",rep(1:10,each=4),":",rep(1:4,10))
rownames(GSN_mean_repres_noise_5)<-paste0("P",rep(1:10,each=4),":",rep(1:4,10))

require(ggcorrplot)
corplot0_750<-ggcorrplot(as.matrix(GSN_mean_repres_noise_0),method = "square",
                         outline.color = "white",tl.cex = 3, tl.col = "gray35",
                         title = "SD(Noise) = 0",
                         legend.title = "",
                         ggtheme = theme(
                           plot.title = element_text(color="gray20", size=8, face="bold",
                                                     hjust = 0.5,vjust=-1),
                           plot.margin = margin(0,0,0,0, 'cm'),
                           axis.ticks = element_blank(), 
                           legend.position = "bottom",
                           legend.justification = "center",
                           legend.text = element_text(color = "gray20", size = 6),
                           legend.key.height= unit(5, 'pt'),
                           legend.key.width= unit(20, 'pt')
                         ),
                         col=c("#3CC4E2","#B5E0E6","#EB5D25")) +
  geom_rect(xmin=0.5,xmax=4.5,ymin=0.5,ymax=4.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=4.5,xmax=8.5,ymin=4.5,ymax=8.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=8.5,xmax=12.5,ymin=8.5,ymax=12.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=12.5,xmax=16.5,ymin=12.5,ymax=16.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=16.5,xmax=20.5,ymin=16.5,ymax=20.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=20.5,xmax=24.5,ymin=20.5,ymax=24.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=24.5,xmax=28.5,ymin=24.5,ymax=28.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=28.5,xmax=32.5,ymin=28.5,ymax=32.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=32.5,xmax=36.5,ymin=32.5,ymax=36.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=36.5,xmax=40.5,ymin=36.5,ymax=40.5,fill=NA,color="#B455A5")

corplot05_750<-ggcorrplot(as.matrix(GSN_mean_repres_noise_0.5),method = "square",
                          outline.color = "white",tl.cex = 3, tl.col = "gray35",
                          title = "SD(noise) = 0.5",
                          legend.title = "",
                          ggtheme = theme(
                            plot.title = element_text(color="gray20", size=8, face="bold",
                                                      hjust = 0.5,vjust=-1),
                            plot.margin = margin(0,0,0,0, 'cm'),
                            axis.ticks = element_blank(), 
                            legend.position = "bottom",
                            legend.justification = "center",
                            legend.text = element_text(color = "gray20", size = 6),
                            legend.key.height= unit(5, 'pt'),
                            legend.key.width= unit(20, 'pt')
                          ),
                          col=c("#3CC4E2","#B5E0E6","#EB5D25")) +
  geom_rect(xmin=0.5,xmax=4.5,ymin=0.5,ymax=4.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=4.5,xmax=8.5,ymin=4.5,ymax=8.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=8.5,xmax=12.5,ymin=8.5,ymax=12.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=12.5,xmax=16.5,ymin=12.5,ymax=16.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=16.5,xmax=20.5,ymin=16.5,ymax=20.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=20.5,xmax=24.5,ymin=20.5,ymax=24.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=24.5,xmax=28.5,ymin=24.5,ymax=28.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=28.5,xmax=32.5,ymin=28.5,ymax=32.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=32.5,xmax=36.5,ymin=32.5,ymax=36.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=36.5,xmax=40.5,ymin=36.5,ymax=40.5,fill=NA,color="#B455A5")

corplot5_750<-ggcorrplot(as.matrix(GSN_mean_repres_noise_5),method = "square",
                         outline.color = "white",tl.cex = 3, tl.col = "gray35",
                         title = "SD(noise) = 5",
                         legend.title = "",
                         ggtheme = theme(
                           plot.title = element_text(color="gray20", size=8, face="bold",
                                                     hjust = 0.5,vjust=-1),
                           plot.margin = margin(0,0,0,0, 'cm'),
                           axis.ticks = element_blank(), 
                           legend.position = "bottom",
                           legend.justification = "center",
                           legend.text = element_text(color = "gray20", size = 6),
                           legend.key.height= unit(5, 'pt'),
                           legend.key.width= unit(20, 'pt')
                         ),
                         col=c("#3CC4E2","#B5E0E6","#EB5D25")) +
  geom_rect(xmin=0.5,xmax=4.5,ymin=0.5,ymax=4.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=4.5,xmax=8.5,ymin=4.5,ymax=8.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=8.5,xmax=12.5,ymin=8.5,ymax=12.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=12.5,xmax=16.5,ymin=12.5,ymax=16.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=16.5,xmax=20.5,ymin=16.5,ymax=20.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=20.5,xmax=24.5,ymin=20.5,ymax=24.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=24.5,xmax=28.5,ymin=24.5,ymax=28.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=28.5,xmax=32.5,ymin=28.5,ymax=32.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=32.5,xmax=36.5,ymin=32.5,ymax=36.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=36.5,xmax=40.5,ymin=36.5,ymax=40.5,fill=NA,color="#B455A5")

#combine data
GSN_repres_750_within_noise_0$noise<-0
GSN_repres_750_within_noise_0.5$noise<-0.5
GSN_repres_750_within_noise_5$noise<-5

GSN_repres_750_between_noise_0$noise<-0
GSN_repres_750_between_noise_0.5$noise<-0.5
GSN_repres_750_between_noise_5$noise<-5

GSN_repres_750_within<-rbind.data.frame(GSN_repres_750_within_noise_0,GSN_repres_750_within_noise_0.5,GSN_repres_750_within_noise_5)
GSN_repres_750_between<-rbind.data.frame(GSN_repres_750_between_noise_0,GSN_repres_750_between_noise_0.5,GSN_repres_750_between_noise_5)

GSN_repres_750_within$Sub<-paste0("Sub",1:nrow(GSN_repres_750_within))
GSN_repres_750_between$Sub<-paste0("Sub",1:nrow(GSN_repres_750_between))
GSN_repres_750_within$Type<-"Within"
GSN_repres_750_between$Type<-"Between"

require(Rmisc)
require(reshape2)
require(ggplot2)

GSN_repres_750_within_long<-melt(GSN_repres_750_within,id.vars = c("Sub","noise","Type"),variable.name = "pair",value.name = "corr")
GSN_repres_750_betw_long<-melt(GSN_repres_750_between,id.vars = c("Sub","noise","Type"),variable.name = "pair",value.name = "corr")

GSN_repres_750_long<-rbind.data.frame(GSN_repres_750_within_long,GSN_repres_750_betw_long)
GSN_repres_750_long$Sub<-as.factor(GSN_repres_750_long$Sub)
GSN_repres_750_long$noise<-as.factor(GSN_repres_750_long$noise)
GSN_repres_750_long$noise<-factor(GSN_repres_750_long$noise,levels = c("0","0.5","5"))

##ANOVA
require(rstatix)
GSN_repres_750_sum<-summarySE(GSN_repres_750_long,measurevar = "corr",groupvars = c("Sub","noise","Type"),na.rm = T)

sink(file = "/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/GSN_repres_750_ANOVA.txt")
anova_test(corr~noise*Type+Error(Sub/Type),data=GSN_repres_750_sum,effect.size = "pes",type = 3)
##post hoc
aov(corr~noise, data = GSN_repres_750_sum[GSN_repres_750_sum$Type=="Within",]) %>% tukey_hsd()
aov(corr~noise, data = GSN_repres_750_sum[GSN_repres_750_sum$Type=="Between",]) %>% tukey_hsd()
sink()
##plot
GSN_repres_750_grand_sum<-summarySE(GSN_repres_750_sum[,c(1,2,3,5)],measurevar = "corr",groupvars = c("noise","Type"),na.rm = T)

write.csv(GSN_repres_750_sum,
          file = "/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/GSN_repres_750_individual.csv",row.names = F)
write.csv(GSN_repres_750_grand_sum,
          file = "/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/GSN_repres_750_grandmean.csv",row.names = F)


barplot_750<-ggplot(GSN_repres_750_sum,aes(x = noise, y = corr,fill = Type)) +
  geom_boxplot(width=.3, outlier.shape = NA,
               position = position_dodge2(preserve = "single"),size = 0.8,alpha = 0.5) +
  #geom_violin(alpha = 0.5,position = position_dodge(0.3)) +
  geom_quasirandom(aes(colour = Type), groupOnX = TRUE,
                   width=.1, dodge.width = 0.3) +
  scale_color_manual(values = c("#6CC6D8","#EE7564")) +
  scale_fill_manual(values = c("#B5E5EF","#F8C9C4")) +
  scale_y_continuous(name = "NRS (r)",limits = c(-0.2,1)) +
  ggtitle("Epoch = 750") +
  theme_classic()
barplot_750<-barplot_750 + 
  theme(plot.title = element_text(color="gray20", size=8, face="bold",hjust = 0.5),
        plot.margin = margin(0,0,0,0, 'cm'),
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8))

  

###1000 epoch
#read in the data & ###calculate the within-person vs. between-person correlation
setwd("/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/GSNResult/1000/cor_output_CSV/")
require(magic)
##construct diagnal matrix
dia_matrix<-matrix(T, 4, 4)
dia_matrix_block<-adiag(dia_matrix,dia_matrix,dia_matrix,dia_matrix,dia_matrix,dia_matrix,dia_matrix,dia_matrix,dia_matrix,dia_matrix)
for (j in c(5,0.5,0)) {
  for (i in 0:19) {
    cor_temp<-read.csv(paste0("cor_output_1000_",j,"_",i,".csv"),header = T, na.strings = "NA")[,-1]
    ###remove the first image of all P identity (which is used for testing)
    cor_temp<-cor_temp[-which(1:nrow(cor_temp)%%5==1),-which(1:ncol(cor_temp)%%5==1)]
    if (i == 0) {
      GSN_mean_repres<-cor_temp
      assign(paste0("GSN_repres_1000_within_noise_",j), cor_temp[(lower.tri(cor_temp) & dia_matrix_block)])
      assign(paste0("GSN_repres_1000_between_noise_",j), cor_temp[(lower.tri(cor_temp) & !dia_matrix_block)])
    } else {
      GSN_mean_repres<-GSN_mean_repres+cor_temp
      assign(paste0("GSN_repres_1000_within_noise_",j), rbind.data.frame(get(paste0("GSN_repres_1000_within_noise_",j)),
                                                                        cor_temp[(lower.tri(cor_temp) & dia_matrix_block)]))
      assign(paste0("GSN_repres_1000_between_noise_",j), rbind.data.frame(get(paste0("GSN_repres_1000_between_noise_",j)),
                                                                         cor_temp[(lower.tri(cor_temp) & !dia_matrix_block)]))
    }
  }
  assign(paste0("GSN_mean_repres_noise_",j),GSN_mean_repres/20)
}
colnames(GSN_repres_1000_within_noise_0)<-paste0("wcor",1:dim(GSN_repres_1000_within_noise_0)[2])
colnames(GSN_repres_1000_within_noise_0.5)<-paste0("wcor",1:dim(GSN_repres_1000_within_noise_0.5)[2])
colnames(GSN_repres_1000_within_noise_5)<-paste0("wcor",1:dim(GSN_repres_1000_within_noise_5)[2])

colnames(GSN_repres_1000_between_noise_0)<-paste0("bcor",1:dim(GSN_repres_1000_between_noise_0)[2])
colnames(GSN_repres_1000_between_noise_0.5)<-paste0("bcor",1:dim(GSN_repres_1000_between_noise_0.5)[2])
colnames(GSN_repres_1000_between_noise_5)<-paste0("bcor",1:dim(GSN_repres_1000_between_noise_5)[2])

##make heat plots

colnames(GSN_mean_repres_noise_0)<-paste0("P",rep(1:10,each=4),":",rep(1:4,10))
rownames(GSN_mean_repres_noise_0)<-paste0("P",rep(1:10,each=4),":",rep(1:4,10))

colnames(GSN_mean_repres_noise_0.5)<-paste0("P",rep(1:10,each=4),":",rep(1:4,10))
rownames(GSN_mean_repres_noise_0.5)<-paste0("P",rep(1:10,each=4),":",rep(1:4,10))

colnames(GSN_mean_repres_noise_5)<-paste0("P",rep(1:10,each=4),":",rep(1:4,10))
rownames(GSN_mean_repres_noise_5)<-paste0("P",rep(1:10,each=4),":",rep(1:4,10))

require(ggcorrplot)
corplot0_1000<-ggcorrplot(as.matrix(GSN_mean_repres_noise_0),method = "square",
                         outline.color = "white",tl.cex = 3, tl.col = "gray35",
                         title = "SD(Noise) = 0",
                         legend.title = "",
                         ggtheme = theme(
                           plot.title = element_text(color="gray20", size=8, face="bold",
                                                     hjust = 0.5,vjust=-1),
                           plot.margin = margin(0,0,0,0, 'cm'),
                           axis.ticks = element_blank(), 
                           legend.position = "bottom",
                           legend.justification = "center",
                           legend.text = element_text(color = "gray20", size = 6),
                           legend.key.height= unit(5, 'pt'),
                           legend.key.width= unit(20, 'pt')
                         ),
                         col=c("#3CC4E2","#B5E0E6","#EB5D25")) +
  geom_rect(xmin=0.5,xmax=4.5,ymin=0.5,ymax=4.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=4.5,xmax=8.5,ymin=4.5,ymax=8.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=8.5,xmax=12.5,ymin=8.5,ymax=12.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=12.5,xmax=16.5,ymin=12.5,ymax=16.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=16.5,xmax=20.5,ymin=16.5,ymax=20.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=20.5,xmax=24.5,ymin=20.5,ymax=24.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=24.5,xmax=28.5,ymin=24.5,ymax=28.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=28.5,xmax=32.5,ymin=28.5,ymax=32.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=32.5,xmax=36.5,ymin=32.5,ymax=36.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=36.5,xmax=40.5,ymin=36.5,ymax=40.5,fill=NA,color="#B455A5")

corplot05_1000<-ggcorrplot(as.matrix(GSN_mean_repres_noise_0.5),method = "square",
                          outline.color = "white",tl.cex = 3, tl.col = "gray35",
                          title = "SD(noise) = 0.5",
                          legend.title = "",
                          ggtheme = theme(
                            plot.title = element_text(color="gray20", size=8, face="bold",
                                                      hjust = 0.5,vjust=-1),
                            plot.margin = margin(0,0,0,0, 'cm'),
                            axis.ticks = element_blank(), 
                            legend.position = "bottom",
                            legend.justification = "center",
                            legend.text = element_text(color = "gray20", size = 6),
                            legend.key.height= unit(5, 'pt'),
                            legend.key.width= unit(20, 'pt')
                          ),
                          col=c("#3CC4E2","#B5E0E6","#EB5D25")) +
  geom_rect(xmin=0.5,xmax=4.5,ymin=0.5,ymax=4.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=4.5,xmax=8.5,ymin=4.5,ymax=8.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=8.5,xmax=12.5,ymin=8.5,ymax=12.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=12.5,xmax=16.5,ymin=12.5,ymax=16.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=16.5,xmax=20.5,ymin=16.5,ymax=20.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=20.5,xmax=24.5,ymin=20.5,ymax=24.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=24.5,xmax=28.5,ymin=24.5,ymax=28.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=28.5,xmax=32.5,ymin=28.5,ymax=32.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=32.5,xmax=36.5,ymin=32.5,ymax=36.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=36.5,xmax=40.5,ymin=36.5,ymax=40.5,fill=NA,color="#B455A5")

corplot5_1000<-ggcorrplot(as.matrix(GSN_mean_repres_noise_5),method = "square",
                         outline.color = "white",tl.cex = 3, tl.col = "gray35",
                         title = "SD(noise) = 5",
                         legend.title = "",
                         ggtheme = theme(
                           plot.title = element_text(color="gray20", size=8, face="bold",
                                                     hjust = 0.5,vjust=-1),
                           plot.margin = margin(0,0,0,0, 'cm'),
                           axis.ticks = element_blank(), 
                           legend.position = "bottom",
                           legend.justification = "center",
                           legend.text = element_text(color = "gray20", size = 6),
                           legend.key.height= unit(5, 'pt'),
                           legend.key.width= unit(20, 'pt')
                         ),
                         col=c("#3CC4E2","#B5E0E6","#EB5D25")) +
  geom_rect(xmin=0.5,xmax=4.5,ymin=0.5,ymax=4.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=4.5,xmax=8.5,ymin=4.5,ymax=8.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=8.5,xmax=12.5,ymin=8.5,ymax=12.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=12.5,xmax=16.5,ymin=12.5,ymax=16.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=16.5,xmax=20.5,ymin=16.5,ymax=20.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=20.5,xmax=24.5,ymin=20.5,ymax=24.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=24.5,xmax=28.5,ymin=24.5,ymax=28.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=28.5,xmax=32.5,ymin=28.5,ymax=32.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=32.5,xmax=36.5,ymin=32.5,ymax=36.5,fill=NA,color="#B455A5") +
  geom_rect(xmin=36.5,xmax=40.5,ymin=36.5,ymax=40.5,fill=NA,color="#B455A5")

#combine data
GSN_repres_1000_within_noise_0$noise<-0
GSN_repres_1000_within_noise_0.5$noise<-0.5
GSN_repres_1000_within_noise_5$noise<-5

GSN_repres_1000_between_noise_0$noise<-0
GSN_repres_1000_between_noise_0.5$noise<-0.5
GSN_repres_1000_between_noise_5$noise<-5

GSN_repres_1000_within<-rbind.data.frame(GSN_repres_1000_within_noise_0,GSN_repres_1000_within_noise_0.5,GSN_repres_1000_within_noise_5)
GSN_repres_1000_between<-rbind.data.frame(GSN_repres_1000_between_noise_0,GSN_repres_1000_between_noise_0.5,GSN_repres_1000_between_noise_5)

GSN_repres_1000_within$Sub<-paste0("Sub",1:nrow(GSN_repres_1000_within))
GSN_repres_1000_between$Sub<-paste0("Sub",1:nrow(GSN_repres_1000_between))
GSN_repres_1000_within$Type<-"Within"
GSN_repres_1000_between$Type<-"Between"

require(Rmisc)
require(reshape2)
require(ggplot2)

GSN_repres_1000_within_long<-melt(GSN_repres_1000_within,id.vars = c("Sub","noise","Type"),variable.name = "pair",value.name = "corr")
GSN_repres_1000_betw_long<-melt(GSN_repres_1000_between,id.vars = c("Sub","noise","Type"),variable.name = "pair",value.name = "corr")

GSN_repres_1000_long<-rbind.data.frame(GSN_repres_1000_within_long,GSN_repres_1000_betw_long)
GSN_repres_1000_long$Sub<-as.factor(GSN_repres_1000_long$Sub)
GSN_repres_1000_long$noise<-as.factor(GSN_repres_1000_long$noise)
GSN_repres_1000_long$noise<-factor(GSN_repres_1000_long$noise,levels = c("0","0.5","5"))

##ANOVA
require(rstatix)
GSN_repres_1000_sum<-summarySE(GSN_repres_1000_long,measurevar = "corr",groupvars = c("Sub","noise","Type"),na.rm = T)
sink(file = "/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/GSN_repres_1000_ANOVA.txt")
anova_test(corr~noise*Type+Error(Sub/Type),data=GSN_repres_1000_sum,effect.size = "pes",type = 3)
##post hoc
aov(corr~noise, data = GSN_repres_1000_sum[GSN_repres_1000_sum$Type=="Within",]) %>% tukey_hsd()
aov(corr~noise, data = GSN_repres_1000_sum[GSN_repres_1000_sum$Type=="Between",]) %>% tukey_hsd()
sink()
##plot
GSN_repres_1000_grand_sum<-summarySE(GSN_repres_1000_sum[,c(1,2,3,5)],measurevar = "corr",groupvars = c("noise","Type"),na.rm = T)

write.csv(GSN_repres_1000_sum,
          file = "/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/GSN_repres_1000_individual.csv",row.names = F)
write.csv(GSN_repres_1000_grand_sum,
          file = "/Users/lchen4/Documents/Research/LCCN related/ASD Face Memory_Modeling/Data analysis/GSN_repres_1000_grandmean.csv",row.names = F)


barplot_1000<-ggplot(GSN_repres_1000_sum,aes(x = noise, y = corr,fill = Type)) +
  geom_boxplot(width=.3, outlier.shape = NA,
               position = position_dodge2(preserve = "single"),size = 0.8,alpha = 0.5) +
  #geom_violin(alpha = 0.5,position = position_dodge(0.3)) +
  geom_quasirandom(aes(colour = Type), groupOnX = TRUE,
                   width=.1, dodge.width = 0.3) +
  scale_color_manual(values = c("#6CC6D8","#EE7564")) +
  scale_fill_manual(values = c("#B5E5EF","#F8C9C4")) +
  scale_y_continuous(name = "NRS (r)",limits=c(-0.2,1)) +
  ggtitle("Epoch = 1,000") +
  theme_classic()
barplot_1000<-barplot_1000 + 
  theme(plot.title = element_text(color="gray20", size=8, face="bold",hjust = 0.5),
        plot.margin = margin(0,0,0,0, 'cm'),
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8))

require(ggpubr)
ggarrange(corplot0_500,corplot05_500,corplot5_500,
          corplot0_750,corplot05_750,corplot5_750,
          corplot0_1000,corplot05_1000,corplot5_1000,
          nrow = 3,ncol=3,labels = c("A","B","C",
                                     "D","E","F",
                                     "G","H","I"),
          common.legend = T,legend = "bottom")

ggarrange(barplot_500,barplot_750,barplot_1000,nrow = 1, ncol = 3,
          labels=c("A","B","C"),common.legend = T,legend = "right")
