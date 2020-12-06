# Machine Learning and Deep Learning Projects (Updating)

I am using this repository to record some self-taught projects conducted by myself. 

Finally, I have found *the hall of tortured souls*, where tormented sprits are everywhere.\
Fortunately, sooner or later I will be one of them, embarking a one-way trip to that place and becoming a new *tortured soul*.\
(https://spinpasta.fandom.com/wiki/Hall_of_Tortured_Souls and https://www.youtube.com/watch?v=WGqD-J_pRvs)

## Table of Content:
- [Breast Cancer Classification](https://github.com/Jacob-Ji-1453/Machine-Learning-and-Deep-Learning-Projects#1-breast-cancer-classification-binary-classification)
- [Fashion Class Classification](https://github.com/Jacob-Ji-1453/Machine-Learning-and-Deep-Learning-Projects#2-fashion-class-classification)
- [User Subscription Classification via App Behavior Analysis](https://github.com/Jacob-Ji-1453/Machine-Learning-and-Deep-Learning-Projects#2-user-subscription-classification-via-app-behavior-analysis)


## 1. Breast Cancer Classification (Binary Classification)

### 1. Introduction:
  - Dataset is extracted and well-processed from tumor images, generating attributes (e.g. cell radius, cell smoothness). [More details](https://github.com/Jacob-Ji-1453/Machine-Learning-and-Deep-Learning-Projects/blob/main/1.%20Breast%20Cancer%20Classification/wdbc.names).
  - The task is classifying the target tumor is **malignant or benign**, by inputing instances with **30 attributes**.
  - Dataset is available [here](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)) or be accessed from sklearn. 

### 2. Preprocessing:
  - The size of dataset is small (**569 instances in total**) and there is no missing value. 
  - Since the original data is in dictionary-like format, converting it into csv is needed.
  
### 3. Data Visualization:
  - The dataset is visualized by pairplot and correlation matrix.
  - Especially, according to the result of pairplot, it will be easily imagined that support vector machine (SVM) can be implemented, since the point is linearly separated.

### 4. Model Tranining:
  - Using train_test_split to make training and testing set, where their proportions are 0.76 and 0.33 respectively.
  - Changing test_size or random_state parameter will affect the accuracy of raw model dramatically.

### 5. Model Evaluation:
  - The dataset is evaluated by confusion matrix. Typically, type 2 error is valued.
  - For raw model, effect of type 2 error is not significant and its overall accuracy is 0.952.

### 6. Model Improvement:
  - Normalization and hyperparameter optimization is implemented in this step.
  - Normalization makes type 2 error rate sightly increased and reduces overall accuracy (0.883). Therefore, giving equal weight to each attribute is not decent.
  - Hyperparameter optimization is applied via grid search to tune C and Gamma. However, it makes type 2 error rate sightly increased and reduces overall accuracy (0.920). 
  - By checking Cs and Gammas of raw model and improved model, best C is found but best Gamma is not in the grid.

### 7. Conclusion:
  - The best model is raw model (overall accucacy 0.952). According to Ahmed et al. (2014), model that accuracy above 0.911 can be considered in diagnosis. 
 
### 8. References:
  - https://www.researchgate.net/publication/271907638_Breast_Cancer_Detection_with_Reduced_Feature_Set

## 2. Fashion Class Classification (Multi-Class Image Classification)

## 3. User Subscription Classification via App Behavior Analysis (Multi-Class Classification)
