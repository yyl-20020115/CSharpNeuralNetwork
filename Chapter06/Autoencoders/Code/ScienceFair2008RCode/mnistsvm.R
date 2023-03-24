library("kernlab")
unlabeledsvmtest <- function(acouts, numlabels, trainingdata, testingdata) {
  mnistsvm <- ksvm(x = data.matrix(trainingdata[1:numlabels,1:acouts]),y=trainingdata$Label[1:numlabels],type= "C-svc",kernel="rbfdot", kpar = "automatic",prob.model=TRUE)
  testanswers <- predict(mnistsvm, testingdata[,1:acouts])
  rightwrong <- testanswers == testingdata$Label
  return(sum(rightwrong))
}
whichacouts <- c(5,10,15,30,784)
whichlabeled <- c(50,100,250,500,1000,5000,20000,60000)
results <- 1:40
dim(results) <- c(5,8) 
for(i in 1:5){
  currentacout <- whichacouts[i]
  testingdata <- read.table(paste("/home/ben/R/Data/testing",currentacout,"outputs.data",sep = ""), header = TRUE, sep = ",")
  trainingdata <- read.table(paste("/home/ben/R/Data/training",currentacout,"outputs.data",sep = ""), header = TRUE, sep = ",")
  for(j in 3:3){
    currentlabeled <- whichlabeled[j]
    results[i,j] <- unlabeledsvmtest(acouts = currentacout, numlabels = currentlabeled, trainingdata = trainingdata, testingdata = testingdata)
  }
}
results
