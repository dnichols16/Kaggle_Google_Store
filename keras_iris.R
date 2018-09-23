data(iris)
library(keras)

str(iris)
#unclass(iris$species)

iris$Species <- as.numeric(iris$Species)
iris <- as.matrix(iris)
dim(iris)
str(iris)

head(iris)
#iris[ , 1:4] <- normalize(iris[, 1:4])

summary(iris)
set.seed(123)
ind <- sample(2,
              nrow(iris),
              replace=TRUE,
              prob=c(0.67, 0.33))
iris.training <- iris[ind==1, 1:4]
iris.test <- iris[ind==2, 1:4]

iris.trainingtarget <- iris[ind==1, 5]
iris.testtarget <- iris[ind==2, 5]

# One hot encode training target values
iris.trainLabels <- to_categorical(iris.trainingtarget)
iris.testLabels <- to_categorical(iris.testtarget)

print(iris.testLabels)
#**************************************
#To start constructing a model, you should first initialize
#a sequential model with the help of the keras_model_sequential()
#function. Then, you’re ready to start modeling.
#the values of the target variable: a flower is either of
#type versicolor, setosa or virginica and this is reflected
#with binary 1 and 0 values.
#A type of network that performs well on such a problem is
#a multi-layer perceptron. This type of neural network is often
#fully connected. That means that you’re looking to build a
#fairly simple stack of fully-connected layers to solve this
#problem. As for the activation functions that you will use,
#it’s best to use one of the most common ones here for the
#purpose of getting familiar with Keras and neural networks,
#which is the relu activation function. This rectifier activation
#function is used in a hidden layer, which is generally speaking
#a good practice.
#In addition, you also see that the softmax activation function
#is used in the output layer. You do this because you want to
#make sure that the output values are in the range of 0 and 1
#and may be used as predicted probabilities:

# Initialize a sequential model
model <- keras_model_sequential() 
# Add layers to the model
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 4, activation = 'softmax')#orig was 3
# Print a summary of a model
summary(model)

# Get model configuration
get_config(model)

# Get layer configuration
get_layer(model, index = 1)

# List the model's layers
model$layers

# List the input tensors
model$inputs

# List the output tensors
model$outputs

#Now that you have set up the architecture of your model, 
#it’s time to compile and fit the model to the data. To 
#compile your model, you configure the model with the adam 
#optimizer and the categorical_crossentropy loss function.
#Additionally, you also monitor the accuracy during the
#training by passing 'accuracy' to the metrics argument.
# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)
#The optimizer and the loss are two arguments that are required 
#if you want to compile the model.
#Some of the most popular optimization algorithms used are the
#Stochastic Gradient Descent (SGD), ADAM and RMSprop. Depending
#on whichever algorithm you choose, you’ll need to tune certain
#parameters, such as learning rate or momentum. The choice for
#a loss function depends on the task that you have at hand: for
#example, for a regression problem, you’ll usually use the Mean
#Squared Error (MSE).
#As you see in this example, you used categorical_crossentropy
#loss function for the multi-class classification problem of
#determining whether an iris is of type versicolor, virginica
#or setosa. However, note that if you would have had a 
#binary-class classification problem, you should have made use
#of the binary_crossentropy loss function.
#Next, you can also fit the model to your data; In this case,
#you train the model for 200 epochs or iterations over all the
#samples in iris.training and iris.trainLabels, in batches of
#5 samples.
# Fit the model 

# Fit the model 
 model %>% fit(
  iris.training, 
  iris.trainLabels, 
  epochs = 200, 
  batch_size = 5, 
  validation_split = 0.2
)
 #Also, it’s good to know that you can also visualize the
 #fitting if you assign the lines of code in the DataCamp
 #Light chunk above to a variable. You can then pass the 
 #variable to the plot() function, like you see in this 
 #particular code chunk!
 # Store the fitting history in `history` 
 history <- model %>% fit(
   iris.training, 
   iris.trainLabels, 
   epochs = 200,
   batch_size = 5, 
   validation_split = 0.2
 )
 
 # Plot the history
 plot(history)
 #One good thing to know is that the loss and acc indicate 
# the loss and accuracy of the model for the training data,
 #while the val_loss and val_acc are the same metrics, loss
 #and accuracy, for the test or validation data.
 #But, even as you know this, it’s not easy to interpret
 #these two graphs. Let’s try to break this up into pieces
 #that you might understand more easily! You’ll split up
 #these two plots and make two separate ones instead: you’ll
 #make one for the model loss and another one for the model
 #accuracy. Luckily, you can easily make use of the $ operator
 #to access the data and plot it step by step.
 
 # Plot the model loss of the training data
 plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")
 # Plot the model loss of the test data
 lines(history$metrics$val_loss, col="green")
 # Add legend
 legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))
 
 # Plot the accuracy of the training data 
 plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")
 # Plot the accuracy of the validation data
 lines(history$metrics$val_acc, col="green")
 # Add Legend
 legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

#Some things to keep in mind here are the following:
#If your training data accuracy keeps improving while 
#your validation data accuracy gets worse, you are probably 
#overfitting: your model starts to just memorize the data instead
#of learning from it.
#If the trend for accuracy on both datasets is still
#rising for the last few epochs, you can clearly see that
#the model has not yet over-learned the training dataset.

#Predict Labels of New Data
#Now that your model is created, compiled and has been fitted
#to the data, it’s time to actually use your model to predict
#the labels for your test set iris.test. As you might have
#expected, you can use the predict() function to do this.
#After, you can print out the confusion matrix to check out
#the predictions and the real labels of the iris.test data with
#the help of the table() function.
 
 # Predict the classes for the test data
 classes <- model %>% predict_classes(iris.test, batch_size = 128)
 
 # Confusion matrix
 table(iris.testtarget, classes)

 #Evaluating Your Model
 #Even though you already have a slight idea of how your
# model performed by looking at the predicted labels for 
 #iris.test, it’s still important that you take the time 
 #to evaluate your model. Use the evaluate() function to
 #do this: pass in the test data iris.test, the test labels
 #iris.testLabels and define the batch size. Store the result
 #in a variable score, like in the code example below:
 
 # Evaluate on test data and labels
 score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)
 
 # Print the score
 print(score)

 #Fine-tuning Your Model
 #Fine-tuning your model is probably something that you’ll
 #be doing a lot, especially in the beginning, because not
 #all classification and regression problems are as
 #straightforward as the one that you saw in the first part
 #of this tutorial. As you read above, there are already two
 #key decisions that you’ll probably want to adjust: how many
 #layers you’re going to use and how many “hidden units” you
 #will chose for each layer.
 
 #In the beginning, this will really be quite a journey.
 
 #Besides playing around with the number of epochs or the
 #batch size, there are other ways in which you can tweak
 #your model in the hopes that it will perform better: by
 #adding layers, by increasing the number of hidden units 
 #and by passing your own optimization parameters to the
 #compile() function. This section will go over these three
 #options.
 
 #Adding Layers
 #What would happen if you add another layer to your model?
 #What if it would look like this?
#****************************************************************
 # Initialize the sequential model
 model <- keras_model_sequential() 
 
 # Add layers to model
 model %>% 
   layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
   layer_dense(units = 5, activation = 'relu') %>% 
   layer_dense(units = 4, activation = 'softmax')
 
 # Compile the model
 model %>% compile(
   loss = 'categorical_crossentropy',
   optimizer = 'adam',
   metrics = 'accuracy'
 )
 
 # Fit the model to the data
 model %>% fit(
   iris.training, iris.trainLabels, 
   epochs = 200, batch_size = 5, 
   validation_split = 0.2
 )
 
 # Evaluate the model
 score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)
 
 # Print the score
 print(score) 
#********************************************************
 # Initialize a sequential model
 model <- keras_model_sequential() 
 
 # Add layers to the model
 model %>% 
   layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
   layer_dense(units = 5, activation = 'relu') %>% 
   layer_dense(units = 4, activation = 'softmax')
 
 # Compile the model
 model %>% compile(
   loss = 'categorical_crossentropy',
   optimizer = 'adam',
   metrics = 'accuracy'
 )
 
 # Save the training history in history
 history <- model %>% fit(
   iris.training, iris.trainLabels, 
   epochs = 200, batch_size = 5,
   validation_split = 0.2
 )
 
 # Plot the model loss
 plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")
 lines(history$metrics$val_loss, col="green")
 legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))
 
 # Plot the model accuracy
 plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")
 lines(history$metrics$val_acc, col="green")
 legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1)) 
 #*****************************************************
 #Hidden Units
 #Also try out the effect of adding more hidden units to your
 #model’s architecture and study the effect on the evaluation,
 #just like this:
 # Initialize a sequential model
 model <- keras_model_sequential() 
 
 # Add layers to the model
 model %>% 
   layer_dense(units = 28, activation = 'relu', input_shape = c(4)) %>% 
   layer_dense(units = 4, activation = 'softmax')
 
 # Compile the model
 model %>% compile(
   loss = 'categorical_crossentropy',
   optimizer = 'adam',
   metrics = 'accuracy'
 )
 
 # Fit the model to the data
 model %>% fit(
   iris.training, iris.trainLabels, 
   epochs = 200, batch_size = 5, 
   validation_split = 0.2
 )
 
 # Evaluate the model
 score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)
 
 # Print the score
 print(score)
 #Note that, in general, this is not always the best 
 #optimilization because, if you don’t have a ton of data,
 the overfitting can and will be worse. That’s why you should
 try use a small network with small datasets as this one.
 
 Why don’t you try visualizing the effect of the addition
 of the hidden nodes in your model? Try it out below:
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 















