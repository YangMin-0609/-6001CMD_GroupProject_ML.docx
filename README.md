6001CMD Group Project — Credit Card Default Prediction

We use the UCI Default of Credit Card Clients dataset to predict whether a customer will default in the following month.


We submits one shared Jupyter notebook. It combine together data exploration, preprocessing, model tuning, and final evaluation.

[Open the shared notebook on the Random-Forest branch](https://github.com/YangMin-0609/-6001CMD_GroupProject_ML.docx/blob/Random-Forest/credit_default_full.ipynb)

## Team contributions

Member Contribution
Random Forest and tuning - Goay Wei Jun
Logistic Regression and tuning - Woon Foo Yang
HistGradientBoosting and tuning - Jeremy Tan Jian Pin
Ensemble development and final comparison - Ong Yang Min

## How the models fit together

We tuned three base models using the same validation setup from "main" after then we combine the final version and push by Goey Wei Jun -[link]


HistGradientBoosting and Random Forest ranked highest by validation ROC-AUC, so we combined them using Soft Voting. We then compared the ensemble and the three tuned base models on the same held-out test set.

[View run instructions](https://github.com/YangMin-0609/-6001CMD_GroupProject_ML.docx/blob/Random-Forest/RUNNING.md)
