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
logistic Regression library creation

