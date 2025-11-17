# MachineLearning-FromScratch
| Term | Definition |
|------|------------|
| **AI** | **Artificial Intelligence** – The broad science of making machines "intelligent" or capable of performing tasks that require human intelligence. |
| **ML** | **Machine Learning** – A subset of AI that focuses on training algorithms to learn patterns from data and make predictions or decisions. |
| **DL** | **Deep Learning** – A subfield of ML using neural networks with multiple layers to model complex patterns in data. |
| **Supervised Learning** | ML tasks where the model is trained on labeled data (input-output pairs). |
| **Unsupervised Learning** | ML tasks where the model identifies patterns in data without labeled outputs. |
| **Reinforcement Learning** | A type of ML where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward. |


## Main Families of Methods for Creating ML Models:  
#### Below are the major categories you should know : 

**1. Linear Models**    
These models use a linear decision boundary.         
Examples:   
- Logistic Regression :Fits a linear boundary and outputs a probability Ex: Predict whether a customer will buy a product based on:age ,income etc it will outputs the proba that this customer will buy or not.    
- Linear SVM (Support vector machines): Finds a straight line that maximizes the margin between classes.Classify emails as spam or not spam based on text features.       
- LDA (Linear Discriminant Analysis) : Models class distributions, finds the best linear separator. Classify types of flowers based on petal and sepal measurements.  

**2. Tree-Based Models**    
These models use decision trees to make predictions.           
Examples:      
- Decision Tree : Predict whether a loan will be approved based on multiple features.
- Random Forest : Many decision trees voting together. Predict if a customer will churn based on behavior data.
- XGBoost (Extreme Gradient Boosting) Builds trees one after another, correcting previous errors. Ex : Predict housing prices based on many features (popular in Kaggle).
- LightGBM
- CatBoost

**Note That `XGBoost` is one of the most powerful and commonly used methods in Kaggle competitions and real-world projects.**  

**3. Ensemble Methods :**   
Combine many models to improve performance.      
Examples:          
- Bagging → Random Forest
- Boosting → XGBoost, LightGBM, AdaBoost      
- Stacking     
- Voting Classifier     

**4. Neural Networks (Deep Learning) :**       
Models inspired by the human brain.    
Examples:    

Multilayer Perceptron (MLP)

Convolutional Neural Networks (CNN)

Recurrent Neural Networks (RNN)

Transformers

Used for images, text, audio, etc.

5. Probabilistic Models

Based on statistics and probability.

Examples:

Naive Bayes

Gaussian Mixture Models

Hidden Markov Models (HMM)

6. Support Vector Machines (SVM)

Can be:

Linear SVM

Non-linear SVM (using kernels like RBF)

7. Clustering Models (Unsupervised)

Used when data isn’t labeled.

Examples:

K-Means

DBSCAN

Hierarchical clustering

8. Dimensionality Reduction Models

Used to reduce number of features.

Examples:

PCA

t-SNE

UMAP
