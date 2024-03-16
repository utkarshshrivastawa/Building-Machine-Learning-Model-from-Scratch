# Building-Machine-Learning-Model-from-Scratch
## Chapter 1 : Building Linear Regression Model from Scratch
### Linear Regression Implementation:
1. A class Linear_Regression is defined with methods to fit the model to the training data (fit) and predict outcomes for new data (predict).
2. The fit method implements Gradient Descent to optimize the model's weights (w) and bias (b) based on the mean squared error cost function.
3. update_weights method calculates gradients and updates the weights and bias.
4. The model's predictions are made by computing the dot product of input features and weights, then adding the bias.

### Application to Salary Data:
1. The salary dataset, presumably containing information on employees' years of experience and their salary, is loaded.
2. Preliminary data exploration is conducted, including checking the shape of the dataset and for missing values.
3. The dataset is split into features (x) and target (y) where the target is the salary.
4. The data is further split into training and testing sets.
5. A Linear_Regression model instance is created with specified learning rate and number of iterations.
6. The model is trained on the training data.
7. The model's parameters (weight and bias) after training indicate the relationship between years of experience and salary.
8. Predictions are made on the test set and visualized against the actual salary values to assess the model's performance.

## Visualization:
1. A scatter plot visualizes the actual salaries (in red) against the years of experience.
2. The model's predictions are plotted (in blue) as a line graph against the years of experience, showing how the model approximates the relationship between experience and salary.

# linear_reg 
This version explicitly states that the code in question is specifically for creating a linear regression model.

# Linear Regression model by notepad++ model
In one version of the code, I build a linear regression model from scratch. In the other version, I achieve the same result by importing the linear regression model (linear_reg) instead of manually creating the class. 


# Chapter 2 : Implementation_Logistic_Regression_Model_from_Scratch
## Data Collection:
The dataset used is a diabetes dataset, likely the Pima Indians Diabetes Database. It includes features like pregnancies, glucose levels, blood pressure, skin thickness, insulin levels, BMI, diabetes pedigree function, age, and the outcome (whether the person has diabetes).

## Data Preprocessing:
1. Missing values are checked (none found in this dataset).
 2.Data is split into features (x) and targets (y), where the target is the Outcome column indicating diabetes presence.
3. The data is standardized using StandardScaler from sklearn, which scales the features to have a mean of 0 and a standard deviation of 1. This is important for models like Logistic Regression that are sensitive to the scale of the input features.
## Logistic Regression Implementation:
1. A class Logistic_Regression is defined with an __init__ method to set hyperparameters (learning rate and number of iterations), a fit method to train the model using gradient descent, and a predict method to classify new data points.
2. The fit method computes the sigmoid of the linear combination of inputs and weights, applies gradient descent to update the weights and biases to minimize the loss function, which is the log loss in logistic regression.
3. The predict method uses the trained weights to make predictions on new data. A threshold of 0.5 is used to classify outputs into 0 (not diabetic) or 1 (diabetic).
## Model Training and Evaluation:
1. The dataset is split into training and testing sets. The model is trained on the training set.
2. Model accuracy is evaluated on both training and testing sets using accuracy_score from sklearn.metrics. This gives an idea of how well the model is performing.
## Prediction System:
A single data point is taken as input, standardized using the same scaler used on the training data, and then fed into the trained model to predict whether the individual has diabetes.
## Conclusion:
The custom Logistic Regression model is capable of classifying individuals as diabetic or not based on their health metrics. The model achieves around 77% accuracy on both training and testing sets, indicating a decent performance. However, further improvements might be needed for a more robust prediction, possibly through hyperparameter tuning or using more complex models.
# Building Logistic Regression Model from Scratch
About Logistic Regression 
# Logictic_reg
This version explicitly states that the code in question is specifically for creating a logistic regression model for  library creation

# Chapter 3 : Building_SVM_machine_learning_from_scratch

## SVM Classifier Implementation:
1. A class SVM_classifier is defined with methods for initialization (__init__), model training (fit), weight updating (update_weights), and prediction (predict).
2. The fit method initializes weights and bias to zero and iterates over the dataset to update these parameters using a custom version of the gradient descent algorithm. The algorithm includes a regularization term (lambda_parameter) to control overfitting.
3. The update_weights method adjusts weights and bias based on whether each data point meets the margin requirements of SVM. It differentiates between correctly classified points (outside the margin) and misclassified or within-margin points, applying different updates accordingly.

## Data Preprocessing and Model Training:
1. The diabetes dataset is loaded, and basic exploratory data analysis is performed (checking for null values, dataset shape, descriptive statistics).
2. Features and targets are split, and feature data is standardized using StandardScaler to have a mean of 0 and a standard deviation of 1. This is crucial for SVM, which is sensitive to the scale of input features.
3. The dataset is divided into training and testing sets, and the SVM_classifier is instantiated with specific hyperparameters (learning rate, number of iterations, and lambda parameter for regularization).
4. The model is trained on the training set.

## Model Evaluation:
1. The trained SVM classifier is used to predict outcomes on both the training and testing sets.
2. Model accuracy is evaluated by comparing predictions to true labels using the accuracy_score function from sklearn.metrics.

## Making Predictions on New Data:
1. An example input is processed (standardized using the same scaler as the training data) and fed into the trained model to predict the diabetes status.
2. The prediction is output as either diabetic (1) or not diabetic (0), based on a decision threshold applied to the model's continuous output.

## Conclusion:
The custom SVM model demonstrates how to implement and train an SVM classifier using only NumPy for matrix operations and demonstrates basic preprocessing, training, and evaluation steps involved in machine learning tasks. This implementation provides a solid foundation for understanding SVM classifiers' workings and can be expanded or modified for more complex or different classification tasks.

# Support Vector Machine Model Notes 
About SVM

# svm_model
This version explicitly states that the code in question is specifically for creating a logistic regression model for  library creation

# Chapter 4 : Implementing_the_Lasso_Regression_Model 
## Lasso Regression Implementation:
1. A Lasso_Regression class is defined with methods for initialization (__init__), model fitting (fit), weight updating (update_weights), and prediction (predict).
2. The fit method initializes weights and bias to zeros, then iterates over the dataset to update these parameters using a modified version of the gradient descent algorithm that includes the Lasso regularization term (lambda_parameter). The regularization term penalizes the absolute size of the regression coefficients and helps in feature selection.
3. The update_weights method calculates the gradient for the weights considering the Lasso penalty, which is responsible for the feature selection capability of Lasso regression.

## Data Preprocessing and Model Training:
1. The salary dataset is loaded, and basic exploratory data analysis is performed (checking for null values, dataset shape).
2. The dataset is divided into features (x) and target (y), followed by a split into training and testing sets.
3. Both the custom Lasso_Regression model and the Lasso model from sklearn.linear_model are instantiated, trained on the training set, and used to predict salaries on the test set.

## Model Evaluation:
1. The performance of both models is evaluated using the R squared error and Mean Absolute Error (MAE). The R squared error measures how well the observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model. MAE measures the average magnitude of the errors in a set of predictions, without considering their direction.
2. The custom model and the sklearn model show very similar performance metrics, demonstrating the effectiveness of the custom Lasso Regression implementation.
# Conclusion:
The custom Lasso Regression model successfully demonstrates the concept of Lasso regularization in linear regression, showing how it can be implemented from scratch using numpy. The comparison with sklearn's Lasso model validates the custom model's performance. This implementation provides a solid foundation for understanding Lasso Regression's workings and can be expanded or modified for more complex tasks or different datasets.


# Contributions
Contributions to this project are welcome! Whether it's suggesting improvements, adding new features, or refining the predictive model, your input is valuable. Feel free to fork the repository, make changes, and submit a pull request with your enhancements.
 

