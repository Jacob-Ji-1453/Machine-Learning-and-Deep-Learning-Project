# Machine Learning and Deep Learning Projects (Updating)

I am using this repository to record some self-taught projects conducted by myself. 

Finally, I have found *the hall of tortured souls*, where tormented sprits are everywhere.\
Fortunately, sooner or later I will be one of them, embarking a one-way trip to that place and becoming a new *tortured soul*.\
(https://spinpasta.fandom.com/wiki/Hall_of_Tortured_Souls and https://www.youtube.com/watch?v=WGqD-J_pRvs)

## Table of Content (Algorithms Specified):
- [Breast Cancer Classification](#1-breast-cancer-classification-binary-classification) (Support Vector Machine)
- [Fashion Class Classification](#2-fashion-class-classification-multi-class-image-classification) (Sequential Model)
- [User Subscription Classification via App Behavior Analysis](#3-user-subscription-classification-via-app-behavior-analysis-binary-classification) (Logistic Regression)
- [Customer Churn Rate Analysis and Minimization](#4-customer-churn-rate-analysis-and-minimization-binary-classification)
- [E-Signing Loan Classification via Finicial History Analysis](#5-e-signing-loan-classification-via-finicial-history-analysis-binary-classification)
- [Credit Card Fraud Detection](#6-credit-card-fraud-detection-binary-classification)


## 1. Breast Cancer Classification (Binary Classification)

### 1. Introduction:
  - It is crucial to detect cancer in early diagnosis. 
  - Dataset is extracted and well-processed from tumor images, generating attributes (e.g. cell radius, cell smoothness, etc.). [More details](https://github.com/Jacob-Ji-1453/Machine-Learning-and-Deep-Learning-Projects/blob/main/1.%20Breast%20Cancer%20Classification/wdbc.names).
  - **Task**: classify the target tumor **malignant or benign**, by inputing instances with **30 attributes**.
  - Dataset is available [here](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)) or be accessed from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) or [Kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data). 

### 2. Preprocessing:
  - The size of dataset is small (**569 instances in total**) and there is no missing value. 
  - Since the original data is in dictionary-like format, converting it into csv is needed.
  
### 3. Data Visualization:
  - The dataset is visualized by pairplot and correlation matrix.
  - Especially, according to the result of pairplot, it will be easily imagined that **support vector machine (SVM)** can be implemented, since the points in pairplots are linearly separated.

### 4. Model Building:
  - Using train_test_split to make training and testing set, where their proportions are 0.76 and 0.33 respectively.
  - Changing test_size or random_state parameter will affect the accuracy of raw model dramatically.

### 5. Model Evaluation:
  - The model is evaluated by confusion matrix. Typically, type 2 error is valued.
  - For raw model, effect of type 2 error is not significant and its overall accuracy is 0.952.

### 6. Model Improvement:
  - Normalization:
    - Normalization makes type 2 error rate sightly increased and reduces overall accuracy (0.883). Therefore, giving equal weight to each attribute is not decent.
  - Hyperparameter optimization
    - Hyperparameter optimization is applied via grid search to tune C and Gamma. However, it makes type 2 error rate sightly increased and reduces overall accuracy (0.920). 
    - By checking Cs and Gammas of raw model and improved model, best C is found but best Gamma is not in the grid.

### 7. Conclusion:
  - The best model is raw model (overall accucacy 0.952). According to Ahmed et al. (2014), model that accuracy above 0.911 can be considered in diagnosis. 
 
### 8. References:
  - https://www.researchgate.net/publication/271907638_Breast_Cancer_Detection_with_Reduced_Feature_Set

## 2. Fashion Class Classification (Multi-Class Image Classification)

### 1. Introduction:
  - Machine Learning & Deep Learning is able to help retailers to detect and forecast the fashion trend. (i.e. by classifying fashion objects via facebook/instagram pictures)
  - Handwritten digitized images are from Fashion-MNIST, where we can build a basic ML/DL model to recognize images.
  - Dataset has **784 features**, which each feature represents a pixel value (an integer ranged from 0 to 255) in a 28*28 grayscale image.
  - Training set and testing set are separated, with **60000 and 10000 instances** respectively and **10 different labels**.
  - **Task**: classify fashion images into **10 labels**. (i.e. 0 for t-shirt/top, 1 for trouser etc.)
  - Datasets are avaliable on [Kaggle](https://www.kaggle.com/zalando-research/fashionmnist). I did not upload datasets since they are too large (>100MB). If it is too large to download, data can be imported from [keras](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data).
  - **Note**: usually, "regular" machine learning models are **NOT** able to handle image recognition. Finding other models is urgently needed! 

### 2. Data Visualization:
  - An image matrix from the training set, with label below, in size of 15*15.
  - Recall labels:
    - 0 => T-shirt/top
    - 1 => Trouser
    - 2 => Pullover
    - 3 => Dress
    - 4 => Coat
    - 5 => Sandal
    - 6 => Shirt
    - 7 => Sneaker
    - 8 => Bag
    - 9 => Ankle boot
![Output](https://github.com/Jacob-Ji-1453/Machine-Learning-and-Deep-Learning-Projects/blob/main/2.%20Fashion%20Class%20Classification/output.png)

### 3. Model Building:
  - From other people's work on Kaggle, it is appropriate to build a **convolutional neural network**.
  - Before building model, normalization is applied to rearrange pixel values between 0 and 1, which makes sure they have the same distribution.
  - Training set is split to a smaller training set and a validation set, while testing set remains.
  - Build a convolutional neural network through a **sequential model**, where it consists of an input layer, a pooling layer, 2 hidden layer outputting the final result (an integer from 0 and 9).
  - Train the model with **50 epochs**, **512 batches**.
  - Training accuracy is 0.8853 and validation accuracy is 0.8684.
  
### 4. Model Evaluation:
  - Input testing set to the model, where the testing accuracy is 0.8755 (sightly declines).
  - Like evaluating "regular" machine learning model, confusion matirx is applied.
  - Confusion matirx shows that the model is worse-performed in classifying label "shirt" (accuracy 0.65).
  - The reason of the misclassification could be difference between "shirt", "pullover" and "coat" is tiny, which they are ambiguous to the model.

### 5. Conclusion:
  - By changing epoch, batch size, feature detector size and its dimension, pooling size could affect the final accuracy.
  - Final accuracy is acceptable, despite the misclassification on "shirt".
  - By importing new features (e.g. color) or increase image size may boost accuracy. 

### 6. References:
  - https://www.kdnuggets.com/2018/06/basic-keras-neural-network-sequential-model.html
  - https://towardsdatascience.com/intuitively-create-cnn-for-fashion-image-multi-class-classification-6e31421d5227
  - https://www.kaggle.com/faressayah/fashion-mnist-classification-using-cnns
  - https://setosa.io/ev/image-kernels/ (Image Kernel Explanation)
  - https://www.cs.ryerson.ca/~aharley/vis/conv/flat.html (MaxPooling/Flattening Explanation)
  - https://becominghuman.ai/image-data-pre-processing-for-neural-networks-498289068258 (Image Normalization Explanation)

## 3. User Subscription Classification via App Behavior Analysis (Binary Classification)

### 1. Introduction:
  - When a company releases an app, which it usually has 2 versions, free and premium. Therefore, the marketing aim is maximizing paid users from free users.
  - Datasets are from a free version app which it collects users' behaviors (e.g. verify phone, product review, etc.) and other pensonal information (e.g. age) in 24 hours. 
  - The number of instances are **50000 instances with 12 features**.
  - **Task**: classify a user would potentially enroll the paid membership or not.
  - Datasets are avaliable on [Kaggle](https://www.kaggle.com/abhishek2602/appdata10). Since the original dataset is private, more information can be found [here](https://indianaiproduction.com/directing-customers-to-subscription-through-financial-app-behavior-analysis-ml-project).

### 2. Exploratory Data Analysis (EDA), Feature Engineering and Preprocessing:
  - EDA:
    - Feature Distributions (via histograms): user who would like to enroll the membership is aged around 25, plays the minigame, uses premium and gives a like to the app. 
    - Correlation Matrices: 3 features are highly correlated with the label: age, minigame, number of screen activities.
  - Feature Engineering:
    - Create "difference" feature to record the time difference between enrolled date and user first open date.
    - Apply funnel analysis to categorize screen activities (see detailed analysis in notebook), counting the number of categorized activities in each category.
  - Preprocessing:
    - Apply standardization to avoid extremely-valued numerical features which may affect prediction dramatically.

### 3. Model Building:
  - Implement **logistic regression**.
  - Use original dataset and standardized dataset to fit the model, where their testing scores are 0.7724 and 0.7727.
  
### 4. Model Evaluation:
  - Confusion matrix: type 2 error is valued (fn is around 1000).
  - Cross Validation Score (CVS): there is a tiny difference between accuracy (0.7727) and CVS (0.7664), where the model's variance is low (0.00677).

### 5. Model Improvement:
  - The model will be improved via grid search.
  - Grid search shows that the final score is 0.7726.
  - Try another grid with small-ranged C may boost the score.

### 6. Conclusion:
  - Standardization boosts accuracy (from 0.7724 to 0.7727).
  - Final model has a decent accuracy (0.7727). 

### 7. References:
  - https://indianaiproduction.com/directing-customers-to-subscription-through-financial-app-behavior-analysis-ml-project
  - https://www.kaggle.com/babakgohardani/predicting-user-subscription-logistic-regression


## 4. Customer Churn Rate Analysis and Minimization (Binary Classification)

## 5. E-Signing Loan Classification via Finicial History Analysis (Binary Classification)

## 6. Credit Card Fraud Detection (Binary Classification)
