# Data-Analysis-and-Statistical-Inference-Python

## This repository contains statistical analysis for boston housing database & machine learning statistical analysis for loan dataset of 346 customers.

### 1. Statistical Analysis of Boston Housing Database

Dataset: [Boston Housing Dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)

This notebook uses scipy, matplotlib, seaborn and statsmodels.api for effective statistical and visualization analysis.

#### Generating Descriptive Statistics and Visualizations such as

1. Median value of owner-occupied homes:

![image](https://user-images.githubusercontent.com/86974424/172115011-35c16fd4-b4a6-4acb-956a-6f13e0a4bca7.png)

Quartile distribution of the owner occupied homes ranges from 17 to 25, q1 or 25% of range = 17 q2 or 75% of range = 25 and median = 21
##
2. Histogram for the Charles river variable:

![image](https://user-images.githubusercontent.com/86974424/172115103-0326b657-6720-44dc-9270-de191671c78b.png)

Through the histogram we can say that tract bounds river are majorly found
##
3. Boxplot for the MEDV variable vs the AGE variable. (Discretizing the age variable into three groups of 35 years and younger, between 35 and 70 years and 70 years and older):

![image](https://user-images.githubusercontent.com/86974424/172115192-6a3b02ac-d4bc-427c-ab39-015401dba334.png)
##
4. Scatter plot to show the relationship between Nitric oxide concentrations and the proportion of non-retail business acres per town:

![image](https://user-images.githubusercontent.com/86974424/172115376-72a2ce95-253e-4556-8c75-7b1b0338c4a1.png)

A linear relationship is seen between Nitric oxide concentrations and the proportion of non-retail business.
##
5. Histogram for the pupil to teacher ratio variable:

![image](https://user-images.githubusercontent.com/86974424/172115455-b17680b5-9433-4014-8dbf-6a252a560105.png)
##
#### Statistical Analysis

1. Is there a significant difference in median value of houses bounded by the Charles river or not?

Hypothesis: 
<br>H0:There is no significant difference in the mean values of median value of houses and Charles river variables (Null Hypothesis)
<br>H1:There is a significant difference in the mean values of median value of houses and Charles river variables

![image](https://user-images.githubusercontent.com/86974424/172116097-6e9a5bad-791a-49ec-b4f3-486241740bf7.png)

<b>Conclusion: Here we get to confirm that the P value is 0 which without doubt less than 0.05 thus we can reject our null hypothesis and opt the other hypothesis</b>
##
2. Is there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)? 

Hypothesis
<br>H0: All age groups have same mean values (Null Hypothesis)
<br>H1: Atleast one age group mean differs

Resut: F_statistic:36.40764999196599, P-value:1.7105011022702984e-15

<b>Conclusion: Here we get to know the P value is 1.7105011022702984e-15 which is less than 0.05 therefore we reject the null hypothesis and opt the other hypothesis</b>
##
3. What is the impact of an additional weighted distance to the five Boston employment centres on the median value of owner occupied homes?

Hypothesis
<br>H0: There is no impact of an additional weighted distance to the five Boston employment centres on the median value of owner occupied homes (Null Hypothesis)
<br>H1: There is an impact of an additional weighted distance to the five Boston employment centres on the median value of owner occupied homes

![image](https://user-images.githubusercontent.com/86974424/172116592-6005a43b-fb2d-4142-91aa-74251aba4ed4.png)

<b>Conclusion: Here we get to know the P value is 1.21e-08 which is less than 0.05 therefore we reject the null hypothesis and see that for every additional weighted distance, median value of owner occupied homes increases by 1.09 unit</b>
##

### 2. Machine learning statistical classification for loan dataset

Dataset: [Loan_Dataset](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv)
##
About dataset:
This dataset is about past loans. The Loan_train.csv data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:

![image](https://user-images.githubusercontent.com/86974424/172117387-a7fc9e92-ed69-4ab9-8101-a7bd2487025c.png)
##

This notebook uses scikit-learn, matplotlib, itertools, pandas and numpy for effective statistical and visualization analysis.

1. Visualization of principal amount and loan_status while considering both genders differently:

![image](https://user-images.githubusercontent.com/86974424/172126756-809a285f-12c3-46ab-b6b7-99194b82526e.png)
##
2. Visualization of age and loan_status while considering both genders differently:

![image](https://user-images.githubusercontent.com/86974424/172126917-60e5fa91-ea47-4794-b206-1f2084104608.png)
##
3. Visualization of dayofweek and loan_status while considering both genders differently:

![image](https://user-images.githubusercontent.com/86974424/172127141-bc8cd3ba-3dc2-43f4-be83-6c2b1a665add.png)
##

### Classification Methods:

1. K Nearest Neighbor(KNN)
2. Decision Tree
3. Support Vector Machine
4. Logistic Regression

- KNN: The best accuracy was 0.7857142857142857 with k= 7
- Decision Tree: DecisionTrees's Accuracy:  0.6142857142857143
- Logistic Regression Modelling: LogisticRegression(C=0.01, solver='liblinear')

## Model Evaluation using Test set

Loan Dataset for testing: [Test_Dataset](https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv)

# Result ~ Obtaining Jaccard and F1 score for different methods:

- `KNN`:
<br>Jaccard Score:  0.6041666666666666
<br>F1-Score:  0.658318980899626

- `Decision Tree`:
<br>Jaccard Score:  0.6590909090909091
<br>F1-Score:  0.7366818873668188

- `SVM`:
<br>Jaccard Score:  0.78
<br>F1-Score:  0.7583503077293734

- `Logistic Regression`:
<br>Jaccard Score:  0.7358490566037735
<br>F1-Score:  0.6604267310789049
<br>Log Loss:  0.5672153379912981

### Through this we can easily tell that Support Vector Machine or SVM is the best choice to proceed forward with statistic evaluation.
