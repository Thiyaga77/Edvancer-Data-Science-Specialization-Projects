library(randomForest)
library(ggplot2)
library(dplyr)
library(tree)
library(cvTools)

library(dplyr)

CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    name=gsub("\\/","_",name)
    name=gsub(">","GT_",name)
    name=gsub("=","EQ_",name)
    name=gsub(",","",name)
    
    data[,name]=as.numeric(data[,var]==cat)
  }
  
  data[,var]=NULL
  return(data)
}


## Classification tree

## Data prep same as earlier module

rg_train=read.csv("bank-full_train.csv",stringsAsFactors = F)

rg_test= read.csv("bank-full_test.csv",stringsAsFactors = F)

rg_test$y=NA

rg_train$data='train'
rg_test$data='test'

rg=rbind(rg_train,rg_test)

###glimpse(rg)

lapply(rg,function(x) length(unique(x)))

names(rg)[sapply(rg,function(x) is.character(x))]

# we'll exclude column named data as it simply represent which dataset the observation is from

cat_cols=c("job","marital","education","default","housing","loan",
           "contact","month","poutcome")

for(cat in cat_cols){
  rg=CreateDummies(rg,cat,50)
}


for(col in names(rg)){
  
  if(sum(is.na(rg[,col]))>0 & !(col %in% c("data","y"))){
    
    rg[is.na(rg[,col]),col]=mean(rg[rg$data=='train',col],na.rm=T)
  }
  
}

rg$y=as.numeric(rg$y=="yes")

## For classification tree we'll need to convert response to factor type

rg$y=as.factor(rg$y)

## Rest of the data prep steps

for(col in names(rg)){
  
  if(sum(is.na(rg[,col]))>0 & !(col %in% c("data","y"))){
    
    rg[is.na(rg[,col]),col]=mean(rg[rg$data=='train',col],na.rm=T)
  }
  
}

rg_train=rg %>% filter(data=='train') %>% select(-data)
rg_test=rg %>% filter(data=='test') %>% select (-data,-y)

set.seed(2)
s=sample(1:nrow(rg_train),0.8*nrow(rg_train))
rg_train1=rg_train[s,]
rg_train2=rg_train[-s,]

## building tree 

rg.tree=tree(y~.-ID-month_may
             -month_jul-month_aug-job_blue_collar-job_management
             -month_jun-job_technician-month_nov
             -job_admin.-month_apr
             -month_feb-job_services,data=rg_train1)

## Performance on validation set

val.score=predict(rg.tree,newdata = rg_train2,type='vector')[,1]
pROC::roc(rg_train2$y,val.score)$auc

## build model on entire data

rg.tree.final=tree(y~.-ID-month_may
                   -month_jul-month_aug-job_blue_collar-job_management
                   -month_jun-job_technician-month_nov
                   -job_admin.-month_apr
                   -month_feb-job_services,data=rg_train)

## Probability score prediciton on test/production data

test.score=predict(rg.tree.final,newdata=rg_test,type='vector')[,1]
write.csv(test.score,"mysubmission_DTC_Project5.csv",row.names = F)

## Function for selecting random subset of params
## Need to include in RFC
subset_paras=function(full_list_para,n=10){
  
  all_comb=expand.grid(full_list_para)
  
  s=sample(1:nrow(all_comb),n)
  
  subset_para=all_comb[s,]
  
  return(subset_para)
}

glimpse(rg_train)

param=list(mtry=c(5,10,15,20,25,35),
           ntree=c(50,100,200,500,700),
           maxnodes=c(5,10,15,20,30,50,100),
           nodesize=c(1,2,5,10)
)

mycost_auc=function(y,yhat){
  roccurve=pROC::roc(y,yhat)
  score=pROC::auc(roccurve)
  return(score)
}

num_trials=15
my_params=subset_paras(param,num_trials)
my_params

myauc=0

## Cvtuning
for(i in 1:num_trials){
  
  params=my_params[i,]
  
  k=cvTuning(randomForest,y~., 
             data =rg_train,
             tuning =params,
             ##***K=10 Changed to K=5 or K=2             
             folds = cvFolds(nrow(rg_train), K=5, type ="random"),
             cost =mycost_auc, seed =2,
             predictArgs = list(type="prob")
  )
  score.this=k$cv[,2]
  
  if(score.this>myauc){
    #print(params)
    # uncomment the line above to keep track of progress
    myauc=score.this
    #print(myauc)
    # uncomment the line above to keep track of progress
    best_params=params
  }
  
  #print('DONE')
  # uncomment the line above to keep track of progress
}

ci.rf.final=randomForest(y~.,
                         mtry=best_params$mtry,
                         ntree=best_params$ntree,
                         maxnodes=best_params$maxnodes,
                         nodesize=best_params$nodesize,
                         data=rg_train
)

## test production data

test.score=predict(ci.rf.final,newdata = rg_test,type='prob')[,1]
write.csv(test.score,'mysubmissionRFC05.csv',row.names = F)     

ci.rf.final

## Probability score prediciton on test/production data

###test.score=predict(rg.tree.final,newdata=rg_test,type='vector')[,1]
###write.csv(test.score,"mysubmission.csv",row.names = F)

## For hardclass prediciton we'll need to find a cutoff on score

train.score=predict(ci.rf.final,newdata=rg_train,type='prob')[,1]
real=rg_train$y

cutoffs=seq(0.001,0.999,0.001)

cutoff_data=data.frame(cutoff=99,Sn=99,Sp=99,KS=99,F5=99,F.1=99,M=99)

for(cutoff in cutoffs){
  
  predicted=as.numeric(train.score>cutoff)
  
  TP=sum(real==1 & predicted==1)
  TN=sum(real==0 & predicted==0)
  FP=sum(real==0 & predicted==1)
  FN=sum(real==1 & predicted==0)
  
  P=TP+FN
  N=TN+FP
  
  Sn=TP/P
  Sp=TN/N
  precision=TP/(TP+FP)
  recall=Sn
  
  KS=abs((TP/P)-(FP/N))
  F5=(26*precision*recall)/((25*precision)+recall)
  F.1=(1.01*precision*recall)/((.01*precision)+recall)
  
  M=(4*FP+FN)/(5*(P+N))
  
  cutoff_data=rbind(cutoff_data,c(cutoff,Sn,Sp,KS,F5,F.1,M))
}

cutoff_data=cutoff_data[-1,]

my_cutoff=cutoff_data$cutoff[which.max(cutoff_data$KS)]

my_cutoff

## Once we know the cutoff we can use it to convert test score to 
## hard classes

test.predicted=as.numeric(test.score>my_cutoff)
write.csv(test.predicted,"mysubmission_RFC_Project5.csv",row.names = F)
View(cutoff_data)






