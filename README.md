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
It is  the linear regression model creation code 

