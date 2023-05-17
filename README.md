# COGS118A Project | Group 14
## Customer Churn Predictor Using the Telco Customer Churn Dataset from IBM Company

## Group Members
- [Yilin Zhu]()
- [Gavin Roberts](https://github.com/empire-penguin)
- []()
- []()

----------------------------

Customer Churn 
: the percentage of customers who stopped purchasing your business's products or services during a certain period of time.

----------------------------

### Dataset
The dataset we used is the Telco Customer Churn Dataset from IBM Company. This dataset is a widely used one in customer analytics and churn prediction studies. It was collected to analyze customer behavior and predict churn. The dataset offers valuable insights into various aspects of customer interactions and characteristics. The dataset consists of 7043 rows representing customers and 21 columns representing features. These features include important customer information such as tenure, total charges, subscriptions, and contract type. The final prediction variable is customer churn. 

### EDA and Data Visualization
Summarize the main characteristics of the dataset by visualizing the variables. This provides an insight about the data. The goal of EDA is to understand the distribution of the data and detect any outliers. Fortunately, some variables could be removed directly by observing the distributions when patterns are identified among variables.

### Data Preprocessing
Data Cleaning. Perform one-hot encoding.

Since there are 21 variables, it potentially includes a lot of noise. So I may want to remove the features with little impact in decision making. Ideally, Principal Component Analysis should be performed. I may want to use XGBoost. Or I could use the result from EDA and remove some features based on reasoning.

### Model Selection

**Algorithms:** \
    Since customer churn is a classification problem, we could use the following algorithms.
    **Support Vector Machine**\
        SVM is a good fit for classification and regression tasks. It will try to find the hyperplane that best separates (maximizes the margin) the data into different classes. In the context of customer churn, aregular SVM would work fine. We can use it to predict whether a customer is likely to churn or not based on various features like their tenure, demographics, usage patterns, etc. It is also a great algorithm for data that need to draw nonlinear decision boundaries. We could use the kernel trick to map the original features to a higher dimensional space.
    **K Nearest Neighbors**\
        KNN is a non-parametric algorithm that could be used in classification tasks. The output is based on the k-nearest data points in the training set. The training process is basically storing the data. The testing could consume more time as it needs to calculate the distance pairwise. In the context of customer churn, we can use KNN to predict whether a customer is likely to churn based on the similar customers in the training set.
    **Logistic Classification**\
         It is a statistical algorithm used for classifications. The underlying mechanics is maximum likelihood optimization. In the context of customer churn, we can use logistic regression to predict whether a customer is likely to churn or not.
    **Decision Tree**\
        It is a non-parametric algorithm. Specifically, the algorithm creates a tree-like model of decisions and their possible consequences. Mathematically, it tries different threshold to maximize the information gain. Meaning that at each spliting stage, it finds the feature that best seperate customer from churn and stay. 
    
**K-Fold Cross-Validation:**
  Ensures the model's performance is reliable and robust. Try different algorithms and tune the hyperparameter to enhance the result. Hyperparameters such as regularization strength or number of trees in the random forest could be changed using grid search or randomized search to improve model performance.

**Algorithm Evaluation:**
    Employ nested cross validation. Use different metrics for evaluation. See below.

### Model Evaluation

For the model and algorithm evaluation metrics, we will use recall, sensitivity, F1-score, accuracy, and AUC-ROC. 

- $\textbf{Recall} = \frac{\text{TP}}{\text{TP + FN}}$

    It is the proportion of actual positives that are correctly identified by the model. In our context, it is the probability of successful identification of customer churns given that customer churns happen. Essentially, we want to maximize this metric to have a more accurate model.
    

- $\textbf{Precision} = \frac{\text{TP}}{\text{TP+FP}}$

    It is the fraction of actual positives given among all positive predictions made by the model. In our project, it helps us see the performance of our model because it measure the probability of correct predictions given that the model makes a positive prediction. Namely, it refers to the proportion of customers who are predicted to churn and actually do churn, out of all the customers who are predicted to churn.
 
 
- $\textbf{Accuracy} = \frac{\text{TP+TN}}{\text{TP+TN+FP+FN}}$

    This would evaluate the fraction of correct predictions among all predictions.
    
    
- $\textbf{F1-score} = \frac{\text{2 * Precision * Recall}}{\text{Precision + Recall}}$

    When comparing 2 models, it is likely that model A has a higher precision and model B has a higher recall. When this happen, we want to pick a model that have a relatively balanced value between precision and recall. We could compare the F1-scores because it represents the harmonic mean of precision and recall.
    
    
- $\textbf{Receiver Operating Characteristic (AUC-ROC)}$

    ROC is generated by plotting the true positive rate against the false positive rate at various thresholds.\
    In this context, the threshold is a value between 0 and 1. For example, when a logistic model output 0.75 as a probability that a customer churn may happen, the final decision really depends on the threshold value: if we set the threshold to be 0.8, then it would still be classified as negative even with a high value.\
    Ideally, we want to pick a threshold that make the TPR to be 1 for any FPR.\
    The AUC is the area under the ROC curve. A good classifier would have an AUC of 1: it rank all positive samples higher than negative samples for any threshold value. This is what we want because we want the model to have a strong ability to identify actual customer churns. We won't care too much about the negatives (customers stay) as it is not a loss for the company.

### Conclusion

### References

----------------------------