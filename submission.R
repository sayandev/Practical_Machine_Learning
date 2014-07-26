library(corrplot)
library(caret)

# Load data, considering the strings 'NA', 'NULL' and blank spaces to be NA values
trainData <- read.csv('pml-training.csv', na.strings=c('', 'NA', 'NULL'))
testData <- read.csv('pml-testing.csv', na.strings=c('', 'NA', 'NULL'))

dim(trainData)

names(trainData)

head(trainData)

summary(trainData)

sapply(trainData[1, ], class)

duplicated(names(trainData))

unique(trainData$classe)

table(trainData$classe)

nsv <- nearZeroVar(trainData,saveMetrics=TRUE)
zeroVarPredictors <- nsv[nsv[,"zeroVar"] == TRUE,]

dropColumns <- names(trainData) %in% row.names(zeroVarPredictors)
trainData <- trainData[,!dropColumns]
testData <- testData[,!dropColumns]

# Sum NAs per column
blankValues <- apply(trainData, 2, function(x) { sum(is.na(x)) })

# Remove columns with more than 50% of NAs
threshold <- nrow(trainData) * 0.5
trainData <- trainData[, which(blankValues < threshold)]
testData <- testData[, which(blankValues < threshold)]

dropColumns <- grep("timestamp|user_name|new_window|num_window|X", names(trainData))
trainData <- trainData[, -dropColumns]
testData <- testData[, -dropColumns]



# Set the seed to make the model reproducible
set.seed(1445)
inTrain = createDataPartition(trainData$classe, p=0.7, list=FALSE)
# 70% of the original training data will be used to train the model
trainingSet <- trainData[inTrain,]
# The remaining 30% will be used to test the model
testingSet <- trainData[-inTrain,]
library(doParallel)
numCores <- detectCores()
registerDoParallel(cores = numCores - 1)
cvFolds <- 10
trControl <- trainControl(method="cv", number=cvFolds, verboseIter=T)
modelFit <- train(classe ~., data=trainingSet, method="rf", trControl=trControl, allowParallel=TRUE)
# Model summary
modelFit
# Final model
modelFit$finalModel
# In Sample Error
predictions <- predict(modelFit, newdata=trainingSet)
inSampleError <- sum(predictions != trainingSet$classe) * 100 / nrow(trainingSet)

# Test the model with a test set
predictions <- predict(modelFit, newdata=testingSet)
confusionMatrix(predictions,testingSet$classe)
outOfSampleError <- sum(predictions != testingSet$classe) * 100 / nrow(testingSet)
importance(modelFit$finalModel)

summary(modelFit)

predictions <- predict(modelFit, newdata=testData)
predictions
# Build answer files
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(predictions)
