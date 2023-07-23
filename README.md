# CLAIM FRAUD DETECTOR #

## Business Problem and Objectives ##
In the insurance industry, claims are one of the common aspects that are noticed by the company. An insurance claim is a formal request to an insurance company asking for a payment based on the terms of the insurance policy. The insurance company reviews the claim for its validity and then pays out to the insured or requesting party (on behalf of the insured) once approved<sup>[1](https://www.gicouncil.in/insurance-education/insurance-claims/)</sup>.

But, due to personal or group interest, in some cases, a person or group can intentionally commit fraud to obtain advantages by the claim. Indeed, this behavior inflicts financial loss for the company. Costs for dealing with fraud are significant and go well beyond the loss claim itself. They can include assessment, detection, and investigation<sup>[2](https://www.counterfraud.gov.au/total-impacts-fraud)</sup>. In addition, the company's reputation is also at stake if the public knows that the fraud cases can't be handled by the company and can become the client's loss if a true claim is wrongly predicted as fraud by the company.

As a preventive way, initial detection to avoid fraud cases can be done by building a system that can predict whether a claim is a fraud or not fraud. With this solution, an unwanted claim can be prevented and minimize financial loss. Furthermore, by adopting such an approach, the company's reputation can be safeguarded, ensuring a positive experience among clients while simultaneously minimizing client loss.

## Business Metrics ##
Based on the objective above, the business metrics are:
1. Minimized and reduce the total financial loss for the fraud claim at a period of time before and after the prediction system built
2. Increase customer satisfaction associated with the claim process

## Machine Learning Workflow ##
![machine_learning_workflow](https://github.com/etikawdywt16/fraud-detection-project/assets/91242818/0871a186-70eb-46c2-a327-73a2ea4eafef)


## Machine Learning Metrics ##
![confusion_matrix](https://github.com/etikawdywt16/fraud-detection-project/assets/91242818/2a21717d-36f0-4114-9ae5-91f2d9c1b7dd)

In this case project and in terms of business, the False Positive will return a bad experience for clients and a bad reputation for the company because the company wrongly predicted the true claim as fraud, so the claim that should have been paid was not paid by the company. Meanwhile, the False Negative will return an unwanted financial loss for the company because the company paid the claim that should not be paid. By all those business objectives, we want to minimize both the False Negatives and False Positives.

The dataset used is the vehicle insurance claim dataset that corresponds with the fraud reported. The data itself contains 36 columns in the features that contain policy detail, insured detail, and claim detail.

![fraud_reported](https://github.com/etikawdywt16/fraud-detection-project/assets/91242818/9f990626-ad2c-4b00-8ed7-c6856222df4c)

The machine learning metric used is the F-1 score because the label data fraud_reported has imbalanced distribution, with the N label more than the Y label. In addition, the precision-recall curve is also used to see how well the model predicted the data in every threshold.

## Modeling ##
### Baseline Model ###
In the baseline model, the model used is DummyClassifier using the most frequent label in the dataset, which is the N label. So, all data predicted as Not Fraud claim.

![baseline_matrix](https://github.com/etikawdywt16/fraud-detection-project/assets/91242818/3c89f8ea-433e-439f-83ae-22f174d1aff4)

This baseline model return an F-1 score of 0.43

![baseline_report](https://github.com/etikawdywt16/fraud-detection-project/assets/91242818/28ac1061-376e-4f8b-9971-821dfa0eb0a8)

### Model without Tuning ###
Because of the high dimensionality of data and because there's no linear correlation between the fraud_reported label and the features, then in this case we try tree-based models to predict the data:

1. Decision Tree Classifier
2. Random Forest Classifier
3. AdaBoost Classifier
4. Gradient Boosting Classifier
5. XGB Classifier

Besides, the data used is various in the re-sampling method and without the re-sampling method.

![model_without_tuning_report](https://github.com/etikawdywt16/fraud-detection-project/assets/91242818/88d89661-b1f5-4dc4-bef7-df2cab8a65e0)

By the report, all models are overfitted in the training dataset and need to be tuned.

### Model with Hyperparameter Tuning ###

K-fold cross-validation with 5-fold returns score that handled overfit in the model before, but some are still overfitted to the training dataset.

![model_with_tuning_report](https://github.com/etikawdywt16/fraud-detection-project/assets/91242818/dd48efda-253e-4365-9abe-f7dfc51db9ed)

### Evaluation Model ###

In the evaluation, DecisionTreeClassifier and XGBClassifier are models with higher F-1 scores, but considering the fit time, DecisionTreeClassifier takes the training process faster than XGBClassifier. So, decided **DecisionTreeClassifer** as the final model.

![evaluation_report](https://github.com/etikawdywt16/fraud-detection-project/assets/91242818/446b8374-c3b7-4bee-ab0f-bffe51275aba)

Decision tree parameter

![decision_tree_params](https://github.com/etikawdywt16/fraud-detection-project/assets/91242818/053e06ca-e5ec-4b7b-8e9a-925b214276d6)

Decision tree plot

![decision_tree_plot](https://github.com/etikawdywt16/fraud-detection-project/assets/91242818/efa342f2-b019-470f-a4b1-546da5c1062b)

Based on the final model, the fraud claims can be classified by the incident_severity, insured_hobbies, and insured_education_level.

## API and Streamlit Service ##
### How to run the API and Streamlit Service ###
On local computer:
1. Run `docker compose up` to run API and Streamlit service
2. Go to `localhost:8501`
   
On online server:
1. Go to `184.72.120.208:8501`
   
All the input in the Streamlit will send to and return response from API in JSON format

### Interface ###
The API as a back-end service and Streamlit as a front-end service are built using Docker and return an interface as below:

![interface](https://github.com/etikawdywt16/fraud-detection-project/assets/91242818/9fb32839-49fd-4b23-b426-dce0782faa52)

## Conclusion and Future Work ##
### Conclusion ###
After all those processes, concluded:
1. The final model is DecisionTreeClassifier with parameters criterion = entropy and max_depth = 3
2. Based on the model, the claim classify as fraud and not fraud by incident_severity, insured_hobbies, and insured_education_level
3. The final model returns the F-1 score of 0.83
   
### Future Work ###
Hope the next work will:
1. Develop a model with a more advanced model that can classify claim more perfectly
2. Develop a Streamlit interface that split policy, insured, and claim detail into 3 separate step forms to be more clearly

## Dataset ##
[Claim Fraud Dataset](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/4954928053318020/1058911316420443/167703932442645/latest.html)
