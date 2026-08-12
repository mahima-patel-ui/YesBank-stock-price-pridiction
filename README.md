# Yes Bank Stock Price Prediction — Streamlit


Project Type: RegressionDomain: Banking / Financial AnalyticsTarget Variable: CloseFinal Model: Linear RegressionDeployment: Streamlit

1. Project Overview

This project develops an end-to-end machine learning solution for predicting the closing price of Yes Bank stock using historical stock-price data.

The project follows a complete data science workflow:

Data Understanding → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Feature Selection → Train/Test Split → Model Building → Hyperparameter Tuning → Model Evaluation → Model Explainability → Streamlit Deployment

The historical dataset contains 185 monthly observations with the original variables:

Date

Open

High

Low

Close

The machine learning model uses:

Open

High

Low

Month

Year

to predict:

Close

The analysis compares Linear Regression, Random Forest Regressor and XGBoost Regressor. Based on the evaluation results documented in the project notebook, Linear Regression was selected as the final model.

2. Business Problem

Stock prices are influenced by a combination of historical price behaviour, company performance, market conditions, investor sentiment, economic conditions and unexpected events.

The objective of this project is to investigate whether historical Yes Bank price information can be used to estimate the closing price.

Problem Statement

Build a supervised machine learning regression model that predicts the closing price of Yes Bank stock using historical Open, High, Low and time-derived features.

The project is intended as a machine learning and financial analytics exercise. The resulting predictions should not be interpreted as guaranteed investment recommendations.

3. Project Objectives

The major objectives are:

Understand the structure and quality of the historical Yes Bank dataset.

Identify trends and relationships among stock-price variables.

Extract useful temporal features from the Date column.

Prepare the data for machine learning.

Build multiple regression models.

Compare model performance using appropriate regression metrics.

Apply hyperparameter tuning to ensemble models.

Select the best-performing model.

Interpret feature influence using SHAP.

Build an interactive Streamlit application.

Make the solution deployable through Streamlit Community Cloud.

4. Dataset Description

The project notebook documents a dataset containing 185 rows and 5 original columns, with monthly observations.

Column

Description

Role

Date

Month and year of observation

Used for feature extraction

Open

Opening stock price

Feature

High

Highest stock price during the period

Feature

Low

Lowest stock price during the period

Feature

Close

Closing stock price

Target

Time Period

The notebook describes historical Yes Bank stock-price data covering approximately July 2005 to November 2020.

Data Quality

The project analysis reports:

No missing values

No duplicate records

No categorical variables

No missing-value imputation required

5. Important Data Consideration: Price Drop

A major movement occurs in the historical stock-price series, including a sharp decline in Yes Bank's share price.

This movement was not removed as an artificial outlier.

The notebook treats this as a genuine market event rather than a data-entry error.

This is important because removing genuine market movements could remove meaningful information from the historical dataset and distort the model.

The project documentation associates the major decline with issues including aggressive/risky lending practices, increasing non-performing assets, regulatory intervention and loss of investor confidence.

6. Exploratory Data Analysis

Several visual analyses were performed.

6.1 Price Distribution

A distribution plot was used to understand the spread of closing prices.

Observation

The closing-price distribution is right-skewed, with many observations concentrated at comparatively lower price levels and fewer observations at substantially higher levels.

6.2 Closing Price Over Time

A line chart was used to examine the historical movement of the closing price.

Observation

The project identifies:

Lower prices during the earlier period

A substantial rise over time

A high-price period around 2018

A sharp decline after the high-price period

Further decline toward 2020

This visualization demonstrates why time-series context is important when interpreting the dataset.

6.3 Open vs Close

A scatter plot was used to examine the relationship between opening and closing prices.

Observation

The project found a strong positive linear relationship between Open and Close.

6.4 High vs Close

A scatter plot was used to investigate the relationship between High and Close.

Observation

The project found a strong linear relationship.

6.5 Low vs Close

A scatter plot was used to investigate the relationship between Low and Close.

Observation

The project found a strong linear relationship between Low and Close.

6.6 Correlation Heatmap

The correlation heatmap showed very strong positive relationships among:

Open

High

Low

Close

The notebook reports correlations in approximately the 0.98–1.00 range among the price variables.

This strong correlation explains why a relatively simple linear model can perform extremely well on this dataset.

6.7 Pair Plot

The pair plot provided a combined view of the numerical relationships.

Key observations documented in the notebook include:

Strong relationships among Open, High, Low and Close

Right-skewed price distributions

No clear seasonal pattern associated with Month

A long-term price movement across Year

Slightly uneven representation across months

7. Feature Engineering

The original Date variable was converted into a datetime representation.

Two additional variables were extracted:

Month
Year

Therefore, the final feature set became:

Open
High
Low
Month
Year

The target remained:

Close

Why Month and Year?

The project considers time-related information relevant because financial results, quarterly comparisons, yearly performance and other time-dependent business events can influence investor expectations and stock prices.

At the same time, the notebook's exploratory analysis found that Month itself did not show a strong visible seasonal pattern.

8. Feature Selection

Independent Variables

Open
High
Low
Month
Year

Dependent Variable

Close

The price variables were selected because they have a strong relationship with the closing price.

9. Data Preprocessing

The following preprocessing decisions were made.

Missing Values

No missing values were reported, so no imputation was required.

Duplicate Values

No duplicate records were reported.

Categorical Encoding

No categorical variables were present, so encoding was unnecessary.

Outlier Treatment

The major price decline was retained because it represents genuine market behaviour rather than an obvious data-quality error.

Feature Scaling

StandardScaler was applied before the Linear Regression model used in the deployed application.

The scaler is fitted on the training data and then applied to the test data and prediction inputs.

10. Train-Test Split

The project uses:

Training data: 80%
Testing data: 20%
Random state: 42

The purpose is to train the model on one portion of the data and evaluate its performance on unseen observations.

11. Machine Learning Models

Three main regression algorithms were evaluated.

Model 1 — Linear Regression

Linear Regression models the relationship between the selected independent variables and the closing price.

Conceptually:

Close =
β0
+ β1(Open)
+ β2(High)
+ β3(Low)
+ β4(Month)
+ β5(Year)

The project applies feature scaling before fitting the model.

Reported Performance

Metric

Linear Regression

R²

99.07%

RMSE

9.17

MSE

84.01–84.02

MAE

≈ 5.90

12. Ridge Regression and GridSearchCV

Ridge Regression was evaluated because the price features are highly correlated.

Ridge adds regularization to the linear regression objective and can help control coefficient magnitude in the presence of multicollinearity.

GridSearchCV was used to search for an appropriate alpha.

Reported Result

The notebook reports:

Best alpha = 0.001
Cross-validation score ≈ 0.9946

The tuned Ridge model maintained very strong performance.

13. Model 2 — Random Forest Regressor

Random Forest is an ensemble algorithm that combines predictions from multiple decision trees.

It can capture nonlinear relationships and interactions between features.

The project evaluated both a baseline Random Forest and a tuned Random Forest using GridSearchCV.

Tuned Random Forest Performance

Metric

Result

R²

97.90%

RMSE

13.77

MSE

189.59

MAE

9.083

14. Model 3 — XGBoost Regressor

XGBoost is a gradient-boosting algorithm that builds trees sequentially to improve predictive performance.

GridSearchCV was used to search for better hyperparameter combinations.

Tuned XGBoost Performance

Metric

Result

R²

97.52%

RMSE

14.96

MSE

223.82

MAE

9.716

15. Final Model Comparison

The final model comparison documented in the notebook is:

Model

R²

RMSE

MSE

MAE

Linear Regression

99.07%

9.17

84.01

5.89–5.90

Random Forest + GridSearchCV

97.90%

13.77

189.59

9.083

XGBoost + GridSearchCV

97.52%

14.96

223.82

9.716

Final Selection: Linear Regression

Linear Regression was selected because it achieved the strongest reported performance across the main evaluation metrics.

Reasons for Selection

1. Best predictive performance

It achieved the highest R² and the lowest RMSE and MAE among the compared final models.

2. Strong linear relationships

EDA showed extremely strong relationships among Open, High, Low and Close.

3. Interpretability

The coefficients can be directly inspected and explained.

4. Computational efficiency

Linear Regression is lightweight and straightforward to deploy.

5. Simpler production architecture

The final Streamlit application only needs the preprocessing pipeline and selected model rather than a large ensemble stack.

16. Evaluation Metrics

Four regression metrics were considered.

R² Score

R² measures the proportion of variance in the target explained by the model.

The project reports:

R² ≈ 99.07%

for Linear Regression.

RMSE

Root Mean Squared Error penalizes larger errors more strongly.

Reported Linear Regression RMSE:

≈ ₹9.17

MAE

Mean Absolute Error measures the average absolute difference between predicted and actual closing price.

Reported Linear Regression MAE:

≈ ₹5.90

This is particularly easy to communicate because it is expressed in the same unit as the stock price.

MSE

Mean Squared Error squares prediction errors before averaging them.

Reported Linear Regression MSE:

≈ 84.01–84.02

17. Model Explainability with SHAP

SHAP (SHapley Additive exPlanations) was used to understand feature contributions.

The notebook reports the following approximate mean absolute SHAP contributions:

| Feature | Approx. Mean |SHAP| ||---|---:|| Low | ≈ 70 || High | ≈ 43 || Open | ≈ 38 || Month | ≈ 1 || Year | ≈ 0 |

Interpretation

Low was identified as the most influential feature, followed by High and Open.

The temporal variables contributed comparatively little according to the documented SHAP analysis.

This is consistent with the strong relationship among the monthly price variables.

18. Streamlit Application

The project includes an interactive Streamlit application.

Application Sections

Overview

Displays:

Dataset size

Key model metrics

Historical closing-price trend

Prediction

The user provides:

Open price

High price

Low price

Month

Year

The application then applies the trained preprocessing pipeline and Linear Regression model to estimate the closing price.

Model Performance

Displays:

R²

RMSE

MAE

MSE

Actual vs predicted values

Linear Regression coefficients

Data

Displays a preview of the processed dataset.

19. Application Architecture

                    User
                     |
                     v
             Streamlit Interface
                     |
                     v
              Input Validation
                     |
                     v
              Feature Creation
             Month + Year from Date
                     |
                     v
              StandardScaler
                     |
                     v
             Linear Regression
                     |
                     v
          Predicted Closing Price

For the dataset workflow:

CSV
 |
 v
Pandas DataFrame
 |
 v
Data Cleaning
 |
 v
Date Processing
 |
 v
Feature Engineering
 |
 v
Train/Test Split
 |
 v
Scaling
 |
 v
Linear Regression
 |
 v
Evaluation
 |
 v
Streamlit Deployment

20. Project Structure

Yes-Bank-Stock-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── data_YesBank_StockPrices.csv
├── .gitignore
│
└── venv/                 # Local only — DO NOT upload to GitHub

File Description

File

Purpose

app.py

Streamlit application and prediction pipeline

requirements.txt

Python dependencies required for the app

README.md

Project documentation

data_YesBank_StockPrices.csv

Historical Yes Bank stock-price dataset

.gitignore

Prevents local/environment files from being committed

venv/

Isolated local Python environment; not deployed

21. Local Setup

Step 1 — Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL
cd Yes-Bank-Stock-Prediction

Step 2 — Create a virtual environment

Windows:

python -m venv venv

Activate:

venv\Scripts\activate

Step 3 — Install dependencies

pip install -r requirements.txt

Step 4 — Run Streamlit

streamlit run app.py

If the streamlit command is not recognized:

python -m streamlit run app.py

22. Requirements

The Streamlit application requires:

streamlit
pandas
numpy
matplotlib
scikit-learn

The deployed application does not need the full set of experimentation libraries used in the original notebook unless additional notebook functionality is being deployed.

23. GitHub Workflow

Do not commit the virtual environment.

Recommended .gitignore:

venv/
__pycache__/
*.pyc
.env
.ipynb_checkpoints/

Initialize Git:

git init

Add files:

git add .

Commit:

git commit -m "Add Yes Bank stock price prediction project"

Connect repository:

git remote add origin YOUR_GITHUB_REPOSITORY_URL

Push:

git branch -M main
git push -u origin main

24. Streamlit Community Cloud Deployment

Push the project to GitHub.

Open Streamlit Community Cloud.

Connect the GitHub repository.

Select the main branch.

Select app.py as the main application file.

Deploy the application.

Verify the prediction workflow after deployment.

The requirements.txt file allows the deployment environment to install the required dependencies.

25. Important Deployment Note About the Dataset

The application supports uploading the CSV through the Streamlit interface.

If the dataset is included in the repository, it can also be loaded locally.

Do not hard-code a personal Windows path such as:

C:\Users\YourName\Downloads\data.csv

because that path will not exist on another computer or on Streamlit Cloud.

Use a relative path or Streamlit's file uploader instead.

26. Limitations

This project has several important limitations.

1. Historical price features only

The model uses Open, High, Low, Month and Year.

It does not include:

Trading volume

Market index movements

Interest rates

Macroeconomic indicators

Company financial statements

News sentiment

Investor sentiment

Technical indicators

Broader market conditions

Therefore, it cannot capture every factor affecting stock prices.

2. Strong feature correlation

Open, High, Low and Close are extremely highly correlated.

This contributes to the excellent reported model performance but also means the model should not automatically be interpreted as a robust real-world forecasting system.

3. Future-input limitation

The deployed prediction interface asks for Open, High and Low.

For a genuinely future period, those values are not known in advance.

Therefore, the application is best described as a historical-data-based closing-price estimation application, not a guaranteed future stock-price forecasting system.

4. Small dataset

The documented dataset contains only 185 monthly observations.

A larger dataset containing more observations and additional market variables would be preferable for a production-grade financial forecasting system.

5. Financial risk

Machine learning predictions should not be treated as investment advice or as a guarantee of future stock performance.

27. Future Improvements

The project could be extended by adding:

Trading volume

Moving averages

RSI

MACD

Bollinger Bands

Market-index features

Banking-sector indicators

Company financial ratios

News sentiment

Macroeconomic indicators

Lag features

Rolling-window statistics

Time-series-specific validation

Walk-forward validation

LSTM/GRU models

Prophet/ARIMA-type time-series approaches

Model monitoring and drift detection

A future version should also ensure that only information available before the prediction period is used as an input feature.

28. Key Business Insights

Based on the project analysis:

Yes Bank's historical closing price has a strong relationship with Open, High and Low prices.

The closing-price distribution is right-skewed.

The stock experienced substantial historical price movement.

The major decline was retained because it represents genuine market behaviour.

The price variables show extremely strong correlations.

Month does not show a strong visible seasonal relationship in the documented EDA.

Linear Regression performed better than the evaluated Random Forest and XGBoost models.

The final Linear Regression model provides a strong and interpretable baseline for this dataset.

SHAP analysis identifies Low, High and Open as the dominant features.

More external financial and market variables would be required for a stronger real-world forecasting system.

29. Conclusion

This project demonstrates a complete supervised machine learning workflow for Yes Bank stock-price analysis.

The process began with data understanding and exploratory analysis, followed by feature engineering, preprocessing, model training, hyperparameter tuning, evaluation and explainability.

Three major regression approaches were compared:

Linear Regression

Random Forest Regressor

XGBoost Regressor

The documented results show that Linear Regression produced the strongest performance:

R²    ≈ 99.07%
RMSE  ≈ 9.17
MAE   ≈ ₹5.90
MSE   ≈ 84.01–84.02

The strong performance is consistent with the very high correlation among the historical price variables.

The project was then converted into an interactive Streamlit application, allowing users to inspect the data, review model performance and generate a closing-price estimate from supplied input features.

Overall, the project demonstrates the complete transition from raw financial data to an evaluated and deployable machine learning application.

30. Disclaimer

This project is created for educational and machine learning demonstration purposes.

The predictions generated by the application are estimates based on historical data and the selected model features. They do not constitute financial, investment or trading advice.

Past stock-price behaviour does not guarantee future performance.

Author

Name: YOUR NAMEProject: Yes Bank Stock Price PredictionCourse: Data Science / Machine Learning CapstoneProject Type: Supervised Learning — Regression

Replace YOUR NAME w
