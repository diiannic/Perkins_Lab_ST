# Libraries
install.packages("mlbench")
install.packages()
library(keras)
library(mlbench) 
library(dplyr)
library(magrittr)
library(neuralnet)

# Data
data("BostonHousing")
data <- BostonHousing
str(data)

#factor variables can't be used in nn replace with the below line of code
data %<>% mutate_if(is.factor, as.numeric)

# Neural Network Visualization
#medv is the dependent variable
# hidden layer first one has 10 neurons second one has 5 neurons
n <- neuralnet(medv ~ crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+b+lstat,
               data = data,#entire data
               hidden = c(10,5),
               linear.output = F,
               lifesign = 'full',
               rep=1)
plot(n,
     col.hidden = 'darkgreen',
     col.hidden.synapse = 'darkgreen',
     show.weights = F,
     information = F,
     fill = 'lightblue')

# Matrix
data <- as.matrix(data)
dimnames(data) <- NULL

# Partition
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(.7, .3))
training <- data[ind==1,1:13]
test <- data[ind==2, 1:13]
trainingtarget <- data[ind==1, 14]
testtarget <- data[ind==2, 14]

# Normalize
#mean and sd from training only
m <- colMeans(training)
#divide ssample into 2
s <- apply(training, 2, sd)
training <- scale(training, center = m, scale = s)
test <- scale(test, center = m, scale = s)

# Create Model
model <- keras_model_sequential()
model %>% 
         layer_dense(units = 5, activation = 'relu', input_shape = c(13)) %>%
         layer_dense(units = 1)
#above last is units = 1 as we want one output
# Compile

#mean square error
model %>% compile(loss = 'mse',
                  optimizer = 'rmsprop',
                  metrics = 'mae')

# Fit Model
mymodel <- model %>%
         fit(training,
             trainingtarget,
             epochs = 100,
             batch_size = 32,
             validation_split = 0.2)
# mean validation error is below
# Evaluate
# a lot of error in the prediction, we should see a straight line
# only 5 neurons used in the hidden layer
model %>% evaluate(test, testtarget)
pred <- model %>% predict(test)
mean((testtarget-pred)^2)
plot(testtarget, pred)

#improve this model, fine tuning it
model <- keras_model_sequential()
model %>% 
        layer_dense(units = 10, activation = 'relu', input_shape = c(13)) %>%
        layer_dense(units = 5, activation = 'relu') %>%
        layer_dense(units = 1)
#where the two lines overlap in the loss and vallossgraphs indicate overfitting is going on

#new fine tunning to the model
# 40% of neurons are not used
#plot looks more like a linear line
# no overfitting here, val loss is lower than the loss
library(keras)
model <- keras_model_sequential()
model %>% 
        layer_dense(units = 100, activation = 'relu', input_shape = c(13)) %>%
        layer_dropout(rate = 0.4)%>%#effective to avoid overfitting, during the training 40% of neurons are dropped 0, 
        layer_dense(units = 50, activation = 'relu')%>%
        layer_dropout(rate = 0.3)%>%
        layer_dense(units = 20, activation = 'relu')%>%
        layer_dropout(rate = 0.2)%>%
        layer_dense(units = 1)

#mean square error and fitting the model 

model %>% compile(loss = 'mse',
                  optimizer = optimizer_rmsprop(lr = 0.001),#learning rate how often weights in the network are updated, change to 0.01-0.05 to see which one works best,l
                  metrics = 'mae')

# Fit Model
mymodel <- model %>%
        fit(training,
            trainingtarget,
            epochs = 100,
            batch_size = 32,
            validation_split = 0.2)

#prediction, more linear, can go back and change the learning rate as well in the compilation of the model, loss is more so increase it again to 0.005, leads to more fluctutation, reduce to 0.001
model %>% evaluate(test, testtarget)
pred <- model %>% predict(test)
mean((testtarget-pred)^2)
plot(testtarget, pred)