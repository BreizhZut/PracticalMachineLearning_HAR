# Human Activity Recognition: Predicting Quality
Dylan Tweed  
July 31, 2016  

# Abstract

For a total of 6 subjects, data from accelerometers on the belt, forearm, arm, and dumbell were collected while performing barbell lifts correctly and incorrectly in 5 different ways. Using random forest machine learning algorithm, we created a model to infer from the data, how the exercise was performed. With an out of sample error rate smaller than 0.3 %,
we can provide diagnostics helping future users to optimize their training exercise and avoid potential injury. 

# Data set

The data for this project was made publicly available in [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har) [^1]. Further detail can be found in [http://groupware.les.inf.puc-rio.br](http://groupware.les.inf.puc-rio.br/har#ixzz4FytImzrr). 

## HAR Training and HAR Testing data sets


```r
HARtrain <- read.csv("pml-training.csv")
HARtest <- read.csv("pml-testing.csv")
```



Source files:

* HAR training data set: [pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) contains:
    * 19622 measurements with 160 observations. 
    * 67 columns contain some missing value
    * 0 columns contain only missing value
    * The last column being the `class` a factor variable A to E, referring to the quality of the exercise
* HAR testing data set: [pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv), this data set constitute a blind test for assessing the performance of the classification model. It contains:
    * 20 measurements with 160 observations.
    * 100 columns contain some missing values
    * 100 columns contain only missing values
    * The last column being the `problem_id`, this data set constitute a blind test for the model we create from the HAR training data set. 
* For both data sets the first 7 entries correspond to the subject names the date and experimental setup. We shall neglect these variables since our model should not depend or rely on those.

## Preprocessing: Column selection

As the HAR testing data set contains the least number of observable, we shall use this set to select the relevant variables.


```r
selectcol <- apply(HARtest,2,function(tcol){!sum(is.na(tcol)) > 0})
## Ignore the first 7 variables
selectcol[1:7] <- F
## Ignore the problem id variable
selectcol["problem_id"] <- F
selectvar <- names(HARtest)[selectcol]
```

After post-processing the following 52 variables remain:


```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"
```

## Preprocessing: training and testing data sets

We create training and testing data set from the HAR train data set. 


```r
library(caret)
selectvar <- names(HARtest)[selectcol]
set.seed(3855)
inTrain = createDataPartition(HARtrain$class, p = 0.7,list=F)
# Save classication as seperate factor varaible 
classtrain <- HARtrain[inTrain,"classe"]
classtest  <- HARtrain[-inTrain,"classe"]
# Save variable used for both training and testing sets
training <- HARtrain[inTrain,selectvar]
testing <- HARtrain[-inTrain,selectvar]
```

* The training sample contains 13737 measurements
* The testing sample contains 5885 measurements

[^1]: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.]

# Machine learning: random forest

Preliminary testing, on a small subset of the training sample show that: 

* decision tree (method `rpart`), with and without Principal Component Analysis (PCA) show bad performances, that may be improved with appropriate parameter tuning
* boosting (method `gbm`), shows better performance that the decision trees, both with and without PCA, but at a cost of computational time
* Random forests (method `rf`), shows the best performances, in term of in sample accuracy, within manageable computational time depending on the parameters. Using PCA didn't seem to show any significant effect on the performances.

Even though, we could have explore more models and fine tune some parameters for improved performances. We chose to limit our modeling to random forest. We explore different values of parameter `mtry` limited to `sqrt(ncol(training)` in order to have an extra margin of tuning to the model.
Cross validation is preformed through 10-fold re-sampling.


```r
# Setting control parameters 10-fold resampling cross-validation
trCtrl <- trainControl(method="cv", number=10, search="random",trim=T)
# Setting mtry
mtry <- floor(sqrt(ncol(training)))
tGrid <- expand.grid(.mtry=c(1:mtry))
# Setting seed
set.seed(9696)
# Creating the model timing is save in rftime for 
system.time(
    modrf <- train(
        classtrain~.,data=training,method="rf",
        trControl=trCtrl,tuneGrid=tGrid)
)
```

```
##     user   system  elapsed 
## 2082.543   17.286 2102.173
```

## Parameter search

<div class="figure">
<img src="HAR_quality_files/figure-html/unnamed-chunk-6-1.png" alt="Parameter tunning of the random forest: !0-fold cross validation accuracy as a function of the number of selected predictors `mtry`"  />
<p class="caption">Parameter tunning of the random forest: !0-fold cross validation accuracy as a function of the number of selected predictors `mtry`</p>
</div>

We display the accuracy improvement for different values of the number of selected predictors `mtry`, the best model correspond to `mtry=7`. However, it corresponds to an improvement of 0.6 % compared to the least accurate model we tried at `mtry=1`.

## In sample error


```r
trainrf <- predict(modrf,training)
cMtrain <- confusionMatrix(classtrain,trainrf)
```

<table>
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> A </th>
   <th style="text-align:right;"> B </th>
   <th style="text-align:right;"> C </th>
   <th style="text-align:right;"> D </th>
   <th style="text-align:right;"> E </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> A </td>
   <td style="text-align:right;"> 3906 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> B </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 2658 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> C </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 2396 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> D </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 2252 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> E </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 2525 </td>
  </tr>
</tbody>
</table>

The confusion matrix, show perfect classification for the training sample. 

<table>
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> Sensitivity </th>
   <th style="text-align:right;"> Specificity </th>
   <th style="text-align:right;"> Pos Pred Value </th>
   <th style="text-align:right;"> Neg Pred Value </th>
   <th style="text-align:right;"> Prevalence </th>
   <th style="text-align:right;"> Detection Rate </th>
   <th style="text-align:right;"> Detection Prevalence </th>
   <th style="text-align:right;"> Balanced Accuracy </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Class: A </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0.2843 </td>
   <td style="text-align:right;"> 0.2843 </td>
   <td style="text-align:right;"> 0.2843 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Class: B </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0.1935 </td>
   <td style="text-align:right;"> 0.1935 </td>
   <td style="text-align:right;"> 0.1935 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Class: C </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0.1744 </td>
   <td style="text-align:right;"> 0.1744 </td>
   <td style="text-align:right;"> 0.1744 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Class: D </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0.1639 </td>
   <td style="text-align:right;"> 0.1639 </td>
   <td style="text-align:right;"> 0.1639 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Class: E </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0.1838 </td>
   <td style="text-align:right;"> 0.1838 </td>
   <td style="text-align:right;"> 0.1838 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
</tbody>
</table>

As expected from the confusion matrix, the model show perfect sensitivity and specificity within the training sample for all classes.

## Out of sample error


```r
testrf <- predict(modrf,testing)
cMtest <- confusionMatrix(classtest,testrf)
```

The confusion matrix, show very reliable classification on the testing sample, a few mis-classifications occur as could be expected for an out of sample test.

<table>
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> A </th>
   <th style="text-align:right;"> B </th>
   <th style="text-align:right;"> C </th>
   <th style="text-align:right;"> D </th>
   <th style="text-align:right;"> E </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> A </td>
   <td style="text-align:right;"> 1673 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> B </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 1136 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> C </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 1024 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> D </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 6 </td>
   <td style="text-align:right;"> 957 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> E </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 1080 </td>
  </tr>
</tbody>
</table>

The out of sample accuracy is extremely good, which make us confident on the validity of our model. 

<table>
 <thead>
  <tr>
   <th style="text-align:right;"> Accuracy </th>
   <th style="text-align:right;"> Kappa </th>
   <th style="text-align:right;"> AccuracyLower </th>
   <th style="text-align:right;"> AccuracyUpper </th>
   <th style="text-align:right;"> AccuracyNull </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:right;"> 0.9975 </td>
   <td style="text-align:right;"> 0.9968 </td>
   <td style="text-align:right;"> 0.9958 </td>
   <td style="text-align:right;"> 0.9986 </td>
   <td style="text-align:right;"> 0.2846 </td>
  </tr>
</tbody>
</table>

This is further confirmed in more detail with the different accuracy estimates displayed per class in the following table.

<table>
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> Sensitivity </th>
   <th style="text-align:right;"> Specificity </th>
   <th style="text-align:right;"> Pos Pred Value </th>
   <th style="text-align:right;"> Neg Pred Value </th>
   <th style="text-align:right;"> Prevalence </th>
   <th style="text-align:right;"> Detection Rate </th>
   <th style="text-align:right;"> Detection Prevalence </th>
   <th style="text-align:right;"> Balanced Accuracy </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Class: A </td>
   <td style="text-align:right;"> 0.9988 </td>
   <td style="text-align:right;"> 0.9998 </td>
   <td style="text-align:right;"> 0.9994 </td>
   <td style="text-align:right;"> 0.9995 </td>
   <td style="text-align:right;"> 0.2846 </td>
   <td style="text-align:right;"> 0.2843 </td>
   <td style="text-align:right;"> 0.2845 </td>
   <td style="text-align:right;"> 0.9993 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Class: B </td>
   <td style="text-align:right;"> 0.9974 </td>
   <td style="text-align:right;"> 0.9994 </td>
   <td style="text-align:right;"> 0.9974 </td>
   <td style="text-align:right;"> 0.9994 </td>
   <td style="text-align:right;"> 0.1935 </td>
   <td style="text-align:right;"> 0.1930 </td>
   <td style="text-align:right;"> 0.1935 </td>
   <td style="text-align:right;"> 0.9984 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Class: C </td>
   <td style="text-align:right;"> 0.9932 </td>
   <td style="text-align:right;"> 0.9996 </td>
   <td style="text-align:right;"> 0.9981 </td>
   <td style="text-align:right;"> 0.9986 </td>
   <td style="text-align:right;"> 0.1752 </td>
   <td style="text-align:right;"> 0.1740 </td>
   <td style="text-align:right;"> 0.1743 </td>
   <td style="text-align:right;"> 0.9964 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Class: D </td>
   <td style="text-align:right;"> 0.9979 </td>
   <td style="text-align:right;"> 0.9986 </td>
   <td style="text-align:right;"> 0.9927 </td>
   <td style="text-align:right;"> 0.9996 </td>
   <td style="text-align:right;"> 0.1630 </td>
   <td style="text-align:right;"> 0.1626 </td>
   <td style="text-align:right;"> 0.1638 </td>
   <td style="text-align:right;"> 0.9982 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Class: E </td>
   <td style="text-align:right;"> 0.9991 </td>
   <td style="text-align:right;"> 0.9996 </td>
   <td style="text-align:right;"> 0.9982 </td>
   <td style="text-align:right;"> 0.9998 </td>
   <td style="text-align:right;"> 0.1837 </td>
   <td style="text-align:right;"> 0.1835 </td>
   <td style="text-align:right;"> 0.1839 </td>
   <td style="text-align:right;"> 0.9993 </td>
  </tr>
</tbody>
</table>

## Error summary

* In Sample Error 0
* 10 fold cross validation error for the best model 0.00597
* Out of Sample Error 0.00255

# Model predictions

We can now perform the model prediction on the testing sample `HARtest`
We perform the same columns selection as for the training data sample `HARtrain`. Arguably the column selection was performed of the `HARtest` test sample. 


```r
Quiztest <- HARtest[,selectcol]
Quizclass <- data.frame(
    problem_id = HARtest[,"problem_id"],
    classe_pred = predict(modrf,Quiztest)
) 
```

The result is saved in a new data frame that we displayed with the following command. For this report however we do not display the result.
Prior to this submission we confirmed that the predictions were validated with the corresponding quiz.


```r
library(knitr)
kable(t(Quizclass),format='html')
```
