## imports
library(dplyr)
library(gbm)

##DataPrep Steps 

bd=read.csv("housing_train.csv",stringsAsFactors = F)

bd_test= read.csv("housing_test.csv",stringsAsFactors = F)

bd_test$Price=NA

bd$data='train'
bd_test$data='test'

colnames(bd)
colnames(bd_test)

##rearrange the column
bd_test=bd_test[,c(1,2,3,4,16,5:15,17)]

colnames(bd)
colnames(bd_test)

bd_all=rbind(bd,bd_test)

library(dplyr)
glimpse(bd_all)

CreateDummies=function(data,var,freq_cutoff=0){
  
  ### function body
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
    
    data[,name]=as.numeric(data[,var]==cat)
  }
  
  data[,var]=NULL
  return(data)
}

glimpse(bd_all)


bd_all=CreateDummies(bd_all,"Suburb",500)
bd_all=CreateDummies(bd_all,"Address",500)
bd_all=CreateDummies(bd_all,"Type",500)
bd_all=CreateDummies(bd_all,"Method",500)
bd_all=CreateDummies(bd_all,"SellerG",500)
bd_all=CreateDummies(bd_all,"CouncilArea",500)

glimpse(bd_all)

## NA values

lapply(bd_all,function(x) sum(is.na(x)))

for(col in names(bd_all)){
  
  if(sum(is.na(bd_all[,col]))>0 & !(col %in% c("data","Price"))){
    
    bd_all[is.na(bd_all[,col]),col]=mean(bd_all[,col],na.rm=T)
  }
  
}

## separate train and test

bd=bd_all %>% filter(data=='train') %>% select(-data)
bd_test=bd_all %>% filter(data=='test') %>% select(-data,-Price)
#################3
set.seed(2)
s=sample(1:nrow(bd),0.7*nrow(bd))
bd_train1=bd[s,]
bd_test1=bd[-s,]

# GBM 
gbm.fit=gbm(Price~.,
            data=bd_train1,
            distribution = "gaussian",
            n.trees = 800,interaction.depth = 5)

test.predicted=predict.gbm(gbm.fit,newdata=bd_test1,n.trees=800)

(test.predicted-bd_test1$Price)**2 %>% mean() %>% sqrt()
# > 212467/312939.1
#[1] 0.6789404

test.predicted=predict.gbm(gbm.fit,newdata=bd_test,n.trees=800)
bd_test$PredictedPrice=test.predicted
write.csv(bd_test,"PredictedPrice_Project1_gbm_June6.csv",row.names = F)
write.csv(bd_test$PredictedPrice,"PredictedPriceInc_Project1_gbm_June6.csv",row.names = F)