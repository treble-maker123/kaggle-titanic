# Titanic: Machine Learning from Disaster

Features that have high correlation (|correlation| > 0.1)to survival are selected, below is the table showing correlation,

|             | Survived  | Title     | HasCabin     | HasRelative | Sex       | Embarked  | Fare      | Age       |
| ----------- | --------- | --------- | ------------ | ----------- | --------- | --------- | --------- | --------- |
| Survived    | 1.000000  | -0.193635 | 0.316912     | 0.203367    | -0.543351 | -0.163517 | 0.246223  | -0.146453 |
| Title       | -0.193635 | 1.000000  | -0.039745    | -0.069469   | 0.250075  | 0.071998  | -0.217061 | 0.206339  |
| HasCabin    | 0.316912  | -0.039745 | 1.000000     | 0.158029    | -0.140391 | -0.154457 | 0.255369  | -0.147788 | 
| HasRelative | 0.203367  | -0.069469 | 0.158029     | 1.000000    | -0.303646 | -0.065610 | 0.385286  | -0.225740 |
| Sex         | -0.543351 | 0.250075  | -0.140391    | -0.303646   | 1.000000  | 0.104057  | -0.213996 | 0.118330  |
| Embarked    | -0.163517 | 0.071998  | -0.154457    | -0.065610   | 0.104057  | 1.000000  | -0.275861 | 0.247199  |
| Fare        | 0.246223  | -0.217061 | 0.255369     | 0.385286    | -0.213996 | -0.275861 | 1.000000  | -0.886794 |
| Age         | -0.146453 | 0.206339  | -0.147788    | -0.225740   | 0.118330  | 0.247199  | -0.886794 | 1.000000  |

Kaggle foot-in-the-door competition, see https://www.kaggle.com/c/titanic for more information.

nn.py - Using a 3-layer neural network to tackle the problem. More layers led to diminishing returns, but more nodes per layer led to better performance, up to a certain point.

avg.py - Using multiple models and average the output. Logistic Regression and Stochastic Gradient Descent consistently perform worse. By excluding those models, the script generates a performance on-par with the output produced by the neural network.