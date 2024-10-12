import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV, Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set seaborn theme for consistency
sns.set_theme(style="whitegrid")

st.title('Regularize Your Real Estate: Introduction to Ridge and Lasso')

st.subheader("Overfitting?")

st.write("""
Have you ever prepared for a test and felt super confident, only to be completely thrown off guard when you're given a problem you didn’t study for? 

Well, you may have just experienced something similar to a fundamental machine learning concept: **overfitting**!

Overfitting in linear regression happens when the model learns the training data too well, capturing noise instead of the underlying pattern.

For instance, if you were interested in predicting house prices, what factors might be important? Let me pitch some: **size**, **location**, and **construction date**. That should be good enough to get started, right?

“Wait,” you’re probably thinking. “Aren’t there a lot of other important factors too? What about **crime rates**, **walkability**, **proximity to good schools**, **whether it’s renovated**, or **how far it is from the nearest Trader Joe’s**?”

All of these are valid! But you can see how adding every single important factor would lead to your prediction process becoming overly complicated and, ultimately, narrow in scope. Just like test preparation, if you spend all your time diving into the minutiae of factors you have in front of you, instead of focusing on broad trends and underlying concepts, you’ll be unprepared when you’re presented with information that isn’t in your very specific scope.

This happens in linear regression models as well. Instead of focusing on an overall trend, the model tries to fit every little fluctuation in the data, including random outliers.

This issue is known as **overfitting** and is a serious problem when it comes to making predictions. While your model might perform very well on its training data, given new information, it will struggle to make a prediction. (1)

""")

st.subheader("Introducing Regularization Techniques")

st.write("""
This is where regularization techniques like Ridge and Lasso Regression come in. They adjust the loss function by adding a penalty for complexity, encouraging the model to be simpler and more generalizable.

The loss function is essentially a scorecard that helps you measure how well your model is performing. Continuing along the same housing model example, the loss function tells you how far off your predictions are from the actual prices of the houses in your training data.

In terms of such a housing model, if the loss function is very low on the training data but significantly higher when predicting new house prices, it indicates overfitting. The model learned the noise in the training data rather than the underlying patterns, leading to poor performance in real-world scenarios. (2)

In simple linear regression, our model looks like this:
""")

st.latex(r'y = \beta_0 + \beta_1 x + \epsilon')

st.write("""
The β's are called parameters. They represent the ground truth that we are trying to estimate using our data. We estimate the parameters using a technique called Ordinary Least Squares (OLS) estimators. Essentially, OLS estimators find the line that minimizes the sum of the squared differences between the observed values and the values predicted by the model. They are found using concepts from calculus and statistics with a formula that looks like this:
""")

st.latex(r'\text{minimize} \quad \sum_{i=1}^{n} (y_i - \hat{y}_i)^2')

st.write("""
If this formula seems daunting, don’t worry! You don’t need to understand the math to grasp the intuition. Think of it like trying to find the best-fitting line on a scatter plot of house prices: you want the line to be as close as possible to all the points, which represent actual house prices. Notice how the left graph fits the data much better than the horizontal average line on the right.
""")

st.image('OLS.png', caption='Ordinary Least Squares (OLS) Method', use_column_width=True)

st.write("""
By minimizing the differences between observed and predicted values, OLS estimators provide a simple yet powerful way to fit a model to the data. However, as we discussed earlier, this method can lead to overfitting, particularly in complex models.

Regularization addresses this issue by introducing a small bias, enhancing the model’s robustness. It’s essentially a reminder: **don’t let perfect be the enemy of good**. While we may not predict every training example perfectly, regularization equips us to better handle new data in the long-term.

This trade-off is crucial for developing models that perform well not only on the training data but also on new, unseen data.

Two popular forms of regularization in linear regression are **Ridge Regression** and **Lasso Regression**.
""")

st.subheader("Ridge Regression")

st.write("""
Ridge Regression is a type of linear regression that includes an L2 penalty, which is the sum of the squares of the coefficients multiplied by a regularization parameter (often denoted as λ or α) that can take any value from 0 to infinity:
""")

st.latex(r'\text{minimize} \quad \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2')

st.write("""
Notice how this formula is slightly different from the previous one for OLS? This difference arises from the addition of the L2 penalty, including the λ parameter, which determines the severity of that penalty.

When λ is set to zero, Ridge Regression is equivalent to OLS regression, meaning no regularization is applied. As λ increases, the penalty for large coefficients grows, effectively shrinking them towards zero. This can help reduce the model's complexity and mitigate overfitting, leading to better performance on unseen data.
""")

st.image('Ridge.png', caption='Ridge Regression Illustration', width=400) 

st.write("""
The key takeaway here is that the blue line (our Ridge Regression line) is closer to the green data points (the data we are testing our model on) than the red line (our OLS linear model). The Ridge Regression line has a smaller slope than the OLS line.

More generally, as we increase λ, the slope gets closer and closer to 0. Small slopes indicate flatter lines. In these cases, increases along the x-axis don't significantly change the y-axis.

This means that as we increase λ, the y-variable (what we are predicting: our response variable) becomes less sensitive to changes in the x-variable (the predictor).

Ultimately, Ridge Regression results in a model that is less sensitive to our training dataset, helping us address the overfitting problem and leading to better predictions.
""")

st.subheader("Coefficient Shrinkage")

st.write("""
But, Ridge Regression is a gift that keeps on giving! It also plays a crucial role in shrinking the coefficients of the model. This shrinking effect occurs because the L2 penalty penalizes larger coefficients more heavily, which encourages the model to distribute weights more evenly among the predictors rather than allowing a few to dominate.

Recall, as λ increases, the slope gets smaller, which means the coefficient estimates (the parameters that determine the slope) of the predictor variables become smaller. This is because we are not only minimizing the loss function, but the additional penalty as well. See the illustration below to better visualize what happens to the distribution of the coefficients as λ changes.
""")

st.image('RidgeParameter.png', caption='Ridge Parameter Visualization', width=400) 

st.write("""
The shrinking of coefficients can be thought of as the model pulling the coefficient values closer to zero, which can lead to a more generalized model. This shrinkage effect is particularly pronounced in cases where multicollinearity is present, as it helps stabilize the estimates and reduce variance.

In practice, this coefficient shrinkage helps improve the model’s ability to generalize to unseen data, as it reduces the complexity and reliance on individual predictors. Ultimately, this can lead to more reliable predictions and a model that is less prone to overfitting.
""")

st.subheader("Lasso Regression")

st.write("""
In many ways, Lasso Regression, or Least Absolute Shrinkage and Selection Operator, is similar to Ridge Regression. It is another form of linear regression that incorporates an L1 penalty:
""")

st.latex(r'\text{minimize} \quad \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j|')

st.write("""
This formula looks really similar to the one from Ridge, but instead of squaring the coefficients, Lasso uses their absolute values. This key difference leads to distinct characteristics in how Lasso affects the coefficients of the model.

Lasso Regression has a similar relation to λ. When λ is set to zero, Lasso is also equivalent to OLS regression, meaning that no regularization is applied. However, as λ increases, the penalty for large coefficients not only shrinks them but can also force some coefficients to exactly zero.

This allows us to virtually exclude the effects of useless predictors from our model. Note that in cases of extreme multicollinearity, this could lead to the model arbitrarily selecting one variable from a group of correlated predictors, potentially ignoring others.
""")

st.image('LassoParameter.png', caption='Lasso Parameter Visualization', width=400) 

st.subheader("Ridge and Lasso Comparison")

st.write("""
**When to Use Ridge Regression:** (tldr: when most variables are useful)
* **Multicollinearity:** Ridge is ideal when predictors are highly correlated. It stabilizes coefficient estimates and helps manage the variance in the model.
* **Predictive Accuracy:** If your primary goal is to improve predictive performance, especially on new data, Ridge can help by constraining coefficient sizes.
* **High-Dimensional Data:** Ridge is computationally efficient and works well when you have a large number of predictors relative to the number of observations.
* **No Feature Selection Needed:** Use Ridge when you want to retain all features in your model, even if some may not be highly informative.
""")

st.write("""
**When to Use Lasso Regression:** (tldr: when you have a lot of useless variables)
* **Feature Selection:** If you need a simpler model with fewer predictors, Lasso is advantageous because it performs automatic feature selection by driving some coefficients to zero.
* **Model Interpretability:** Lasso can enhance interpretability, making it easier to understand which variables are most important, especially in high-dimensional datasets.
* **Sparse Solutions:** When you suspect that many predictors are irrelevant or redundant, Lasso can help isolate the most impactful features.
---
""")

st.title("Regularize Your Real Estate: (Demo)")

st.write("""
Let’s explore these concepts in a real dataset. The Kaggle Housing Prices dataset is a well-known resource used for exploring advanced regression techniques. It contains information on residential homes in Ames, Iowa.
""")

df = pd.read_csv('train.csv') 
st.write('### Dataset Overview')
st.write(df.head())

st.write("""
#### Objective:
The primary goal is to predict the final sale price of homes based on a variety of characteristics.

#### Features:
The dataset includes 79 explanatory variables covering a wide range of factors such as:

* **Structural Features**: Year built, total square footage, number of bedrooms and bathrooms.
* **Location**: Neighborhood, proximity to amenities, and lot size.
* **Quality Metrics**: Overall condition, quality ratings, and specific attributes like basement and garage presence.
* **Miscellaneous**: Features such as swimming pools, fences, and the presence of a central air system.

Whew... that's a lot of features. A data scientist’s nightmare. But, a party for Ridge and Lasso!

Based on the sheer amount of predictors, it can be assumed there is overfitting present. Now, let’s check for multicollinearity.
""")

df = df.drop(columns=['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'])
df = df.rename(columns={"1stFlrSF": "FirstFlrSF", "2ndFlrSF": "SecondFlrSF"})

selected_features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'FirstFlrSF', 
    'YearBuilt', 'YearRemodAdd', 'Neighborhood', 'ExterQual', 'KitchenQual', 'Fireplaces',
    'LotArea', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'BsmtQual', 'BsmtFinSF1', 
    'HeatingQC', 'CentralAir', 'PavedDrive'
]

X = df[selected_features]
y = df['SalePrice']

X = pd.get_dummies(X, drop_first=True)
X.fillna(X.mean(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# LR for baseline comparison
linear_reg = LinearRegression().fit(X_train, y_train)

formula = 'SalePrice ~ OverallQual + GrLivArea + GarageCars + GarageArea + TotalBsmtSF + FirstFlrSF + YearBuilt + YearRemodAdd + Neighborhood + ExterQual + KitchenQual + Fireplaces + LotArea + FullBath + HalfBath + TotRmsAbvGrd + BsmtQual + BsmtFinSF1 + HeatingQC + CentralAir + PavedDrive'

y, X = dmatrices(formula, data=df, return_type='dataframe')

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

vif_sorted = vif.sort_values(by="VIF Factor", ascending=False)

vif_no_intercept = vif_sorted[vif_sorted["features"] != "Intercept"]

colors = ['darkblue' if vif > 10 else 'skyblue' for vif in vif_no_intercept["VIF Factor"]]

st.write("### Variance Inflation Factor (VIF)")
plt.figure(figsize=(12, 8))
plt.barh(vif_no_intercept["features"], vif_no_intercept["VIF Factor"], color=colors, height=0.8)  # Adjusting bar width
plt.axvline(x=4, color='orange', linestyle='--', label='VIF = 4')
plt.axvline(x=10, color='red', linestyle='--', label='VIF = 10')
plt.xlabel('Variance Inflation Factor (VIF)')
plt.title('VIF Scores for Selected Features (Without Intercept)')
# Improving the y-label readability
plt.gca().invert_yaxis()
plt.yticks(fontsize=10, rotation=0, ha="right")  # Aligning y-ticks to the right
plt.tight_layout()
plt.legend()
st.pyplot(plt) 

st.write("""
An initial examination for multicollinearity using the Variance Inflation Factor (VIF) scores reveals there are a number of moderately correlated features (4 < VIF < 10), which is to be expected with such a complex model.

Notably, there are a couple of features with severe multicollinearity (VIF > 10), including: the categorical variables ExterQual (Exterior material quality) and Neighborhood. 

Given the presence of multicollinearity, an initial thought would be that Ridge might be a better option for this dataset, but let’s keep exploring by applying Ridge and Lasso techniques we’ve discussed!
""")

# Fit LassoCV model
# Apply Lasso regression with cross-validation to find the best alpha (regularization parameter)
lasso_cv = LassoCV(cv=5, random_state=42).fit(X_train, y_train)

# Get the coefficients from Lasso
lasso_coefficients = pd.DataFrame({
    "Feature": X_train.columns,  # Now using X_train with column names
    "Coefficient": lasso_cv.coef_
}).sort_values(by="Coefficient", ascending=False)

# Filter the selected features (features with non-zero coefficients)
selected_features_lasso = lasso_coefficients[lasso_coefficients["Coefficient"] != 0]


# Performance metrics
lasso_performance = {
    "Best Alpha (Lambda)": lasso_cv.alpha_,
    "Number of Selected Features": len(selected_features_lasso)
}

# Evaluate Lasso model on test data.
# Predicting on the test set
y_pred = lasso_cv.predict(X_test)

# Calculating mean squared error (MSE) and R-squared (R²) score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Displaying the performance metrics
lasso_test_performance = {
    "Mean Squared Error (MSE)": mse,
    "R-squared (R²)": r2
}

# Plotting the comparison between Lasso and Ridge feature importance
st.write("### Feature Importance: Lasso vs Ridge")
plt.figure(figsize=(10, 8))
lasso_sorted = selected_features_lasso.set_index('Feature')['Coefficient'].abs().sort_values(ascending=False)
ridge_cv = Ridge(alpha=lasso_cv.alpha_).fit(X_train, y_train)
ridge_coefficients = pd.DataFrame({
    "Feature": X_train.columns,  # Use X_train_df.columns to get feature names
    "Coefficient": ridge_cv.coef_
}).sort_values(by="Coefficient", ascending=False)
ridge_sorted = ridge_coefficients.set_index('Feature')['Coefficient'].abs().sort_values(ascending=False)

lasso_sorted.plot(kind='bar', alpha=0.7, color='blue', label='Lasso')
ridge_sorted.plot(kind='bar', alpha=0.7, color='orange', label='Ridge')

plt.title("Feature Importance: Lasso vs Ridge")
plt.ylabel("Absolute Coefficient Value")
plt.xlabel("Feature")
plt.legend()
plt.tight_layout()
st.pyplot(plt)  

st.write("""
**Lasso's ability to zero out less important coefficients seems to be beneficial here**. It's likely identifying the key factors that truly drive house prices.
* Robustness: Lasso's more gradual performance decline suggests it might be more robust to different levels of regularization.
* Interpretability: With Lasso potentially eliminating some features, our model becomes more interpretable. We can focus on the features it deems most important for predicting house prices.

#### Let’s summarize:
* Want a simpler model? Lean towards higher parameter values with Lasso.
* Need to retain all features? Ridge might be your best bet.

""")

# Perform a grid search over a range of alpha values to test different levels of regularization and 
# find the optimal alpha for better performance.

# Defining the range of alpha values for grid search
alpha_grid = {'alpha': [0.1, 1, 10, 50, 100, 200, 500, 1000, 2000]}

# Setting up the Lasso model
lasso_model = Lasso(max_iter=10000)

# Performing grid search with 5-fold cross-validation
grid_search = GridSearchCV(estimator=lasso_model, param_grid=alpha_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Getting the best alpha and the best performance score
best_alpha = grid_search.best_params_['alpha']
best_r2 = grid_search.best_score_

# Applying the best alpha to the Lasso model and evaluating on the test set
best_lasso = Lasso(alpha=best_alpha, max_iter=10000)
best_lasso.fit(X_train, y_train)
y_pred_best = best_lasso.predict(X_test)

# Get predictions for all models
y_pred_ridge = ridge_cv.predict(X_test)
y_pred_linear = linear_reg.predict(X_test)

# Plot predicted vs actual for Lasso, Ridge, and Linear Regression
st.write("### Prediction vs. Actual Scatter Plot (Lasso, Ridge, Linear Regression Comparison)")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6, label='Lasso', color='blue')
plt.scatter(y_test, y_pred_ridge, alpha=0.6, label='Ridge', color='orange')
plt.scatter(y_test, y_pred_linear, alpha=0.6, label='Linear Regression', color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Line for perfect prediction

plt.title("Predicted vs Actual House Prices")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.legend()
plt.tight_layout()
st.pyplot(plt)

st.write("""
#### High-level observations:
* All models show a strong positive correlation between predicted and actual prices.
* Linear Regression seems to have slightly more spread than Lasso or Ridge.
* Ridge and Lasso perform similarly, with Ridge possibly having a slight edge for mid-range prices, which adds up given the multicollinearity present.
""")

# Residuals Plot
st.write("### Residuals vs Predicted Values")
residuals_lasso = y_test - y_pred
residuals_ridge = y_test - ridge_cv.predict(X_test)
y_pred_linear = linear_reg.predict(X_test)
residuals_linear = y_test - y_pred_linear

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals_lasso, alpha=0.6, label='Lasso', color='blue')
plt.scatter(ridge_cv.predict(X_test), residuals_ridge, alpha=0.6, label='Ridge', color='orange')
plt.scatter(y_pred_linear, residuals_linear, alpha=0.6, label='Linear Regression', color='green')
plt.axhline(0, color='r', linestyle='--', linewidth=2)  # Line for zero residuals

plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted SalePrice")
plt.ylabel("Residuals")
plt.legend()
plt.tight_layout()
st.pyplot(plt)  

st.write("""
A tool to diagnose problems with our model is called the residuals plot. Let's take a closer look at what our residuals are telling us about our models.

As a quick refresher: residuals are the differences between our predicted house prices and the actual house prices. In an ideal world, these residuals would be randomly scattered around zero, showing no particular pattern.

Looking at our residuals plot, we can see some interesting patterns:
* **The Fan or Funnel Effect**: Notice how the spread of residuals increases as the predicted price increases? This 'funnel' shape is present in all three models (Ridge, Lasso, and Linear Regression), suggesting that our models are less accurate for higher-priced homes. This is a common issue in real estate prediction – luxury homes often have unique features that make them harder to price accurately.
* **Regularization at Work**: Comparing the OLS residuals to those of Ridge and Lasso, it seems that the regularized models (Ridge and Lasso) have slightly tighter clusters of residuals? This is regularization doing its job, reducing the impact of outliers and making our predictions more robust.
* **Lasso Edging Out Ridge?**: It seems Lasso's residuals are slightly more compact than Ridge's, especially for mid-range house prices.

All three models show larger residuals for the most expensive houses (top right of the plot). This suggests that no matter which regression technique we use, predicting prices for high-end properties remains challenging.

This indicates a couple things:
* **Prediction nd Confidence Intervals**: We should be more cautious with our predictions for very expensive houses. It might be wise to provide wider intervals for these predictions.
* **Feature Engineering**: The funnel shape suggests that we might be missing some important features that explain the variability in high-end home prices. We might have to engineer new features or collect additional data for luxury properties.
* **Non-Linear Relationships**: The persistent funnel shape across all models hints that there might be some non-linear relationships in our data that our linear models, teeing us up to explore non-linear modeling techniques next.
""")

st.subheader("But, what even is λ?")

st.write("""
So, we’ve discussed this very important parameter, λ. But how do we actually find it?

Choosing the right value of λ is important: it directly impacts the model’s performance and predictive accuracy. A poorly chosen λ can lead to overfitting or underfitting, which defeats the purpose of applying Ridge or Lasso.

Cross-validation is a powerful technique used to assess how a statistical analysis will generalize to an independent dataset. By dividing the data into subsets, we can train the model on some of these subsets while validating it on others. This process provides a more reliable estimate of the model’s performance and helps in selecting the optimal λ.

One of the most widely used methods for cross-validation is k-fold cross-validation. In this approach, the dataset is divided into k subsets (or folds). The model is trained on k-1 folds and then validated on the remaining one. This process is repeated until every fold is rotated through. This method not only helps in reducing variance but also ensures that every data point gets to be in both training and validation sets.

""")

# Simulating CV results for Ridge and Lasso over a range of alpha values
alphas = np.logspace(-2, 3, 10)
lasso_scores = [Lasso(alpha=a).fit(X_train, y_train).score(X_test, y_test) for a in alphas]
ridge_scores = [Ridge(alpha=a).fit(X_train, y_train).score(X_test, y_test) for a in alphas]

# Plotting the cross-validation results
plt.figure(figsize=(10, 6))
plt.plot(alphas, lasso_scores, label='Lasso', color='blue', marker='o')
plt.plot(alphas, ridge_scores, label='Ridge', color='orange', marker='o')

st.write("#### Now, let’s apply these concepts to our housing dataset.")
plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Cross-Validation R² Score')
plt.title('Cross-Validation Results: Lasso vs Ridge')
plt.legend()
plt.tight_layout()
st.pyplot(plt)

st.write("""
First, we’ll implement cross-validation to evaluate how different λ values impact model performance.

The initial analysis suggests that values between 1 and 100 work well for both Ridge and Lasso models, with Lasso showing a slight edge.
* Similar Performance at Low λ: For small values of λ, both Ridge and Lasso perform similarly well. This indicates that at low levels of regularization, both models can capture the relationships in the data effectively.
* Lasso's Edging out Ridge (Again)?: As λ increases, Lasso maintains its performance longer than Ridge. This behavior suggests that Lasso is proficient at selecting the most significant features while keeping the model robust.
* The Cliff: However, at very high λ values, both models exhibit a dramatic drop in performance. Ridge, in particular, shows a steeper decline, indicating that excessive regularization can severely limit the model's ability to fit the data.

To optimize predictive power, you should select the λ value that yields the highest cross-validation score. This ensures that your model is not only well-fitted to the training data but also performs well on unseen data. (4)
""")

# Interactive app

df = pd.read_csv('train.csv')  # Provide the path to your file here
# st.write('### Dataset Overview')
# st.write(df.head())

# Drop columns with more than 80% null values
df = df.drop(columns=['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'])
df = df.rename(columns={"1stFlrSF": "FirstFlrSF", "2ndFlrSF": "SecondFlrSF"})

# Select relevant features for the analysis
selected_features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'FirstFlrSF', 
    'YearBuilt', 'YearRemodAdd', 'Neighborhood', 'ExterQual', 'KitchenQual', 'Fireplaces',
    'LotArea', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'BsmtQual', 'BsmtFinSF1', 
    'HeatingQC', 'CentralAir', 'PavedDrive'
]

X = df[selected_features]
y = df['SalePrice']

# Handle categorical variables and missing values
X = pd.get_dummies(X, drop_first=True)
X.fillna(X.mean(), inplace=True)

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled array back into DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

st.write("""
### See λ in Action!

Let's actually play around with what we've been discussing! Below, you can adjust the parameters to understand how it affects Ridge and Lasso regression better. Change the regularization strength and see how it affects the model's performance and feature importance.
""")

# User input for regularization strength
alpha = st.slider('Select regularization strength (alpha)', 0.1, 100.0, 1.0, 0.1)

# User input for regression type
regression_type = st.radio('Select regression type', ('Ridge', 'Lasso', 'Both'))

# Fit the selected model(s)
if regression_type == 'Ridge' or regression_type == 'Both':
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)

if regression_type == 'Lasso' or regression_type == 'Both':
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)
    y_pred_lasso = lasso_model.predict(X_test)

# Calculate and display metrics
if regression_type == 'Ridge':
    mse = mean_squared_error(y_test, y_pred_ridge)
    r2 = r2_score(y_test, y_pred_ridge)
    st.write(f'Ridge - Mean Squared Error: {mse:.2f}')
    st.write(f'Ridge - R-squared Score: {r2:.2f}')
elif regression_type == 'Lasso':
    mse = mean_squared_error(y_test, y_pred_lasso)
    r2 = r2_score(y_test, y_pred_lasso)
    st.write(f'Lasso - Mean Squared Error: {mse:.2f}')
    st.write(f'Lasso - R-squared Score: {r2:.2f}')
else:
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)
    st.write(f'Ridge - Mean Squared Error: {mse_ridge:.2f}')
    st.write(f'Ridge - R-squared Score: {r2_ridge:.2f}')
    st.write(f'Lasso - Mean Squared Error: {mse_lasso:.2f}')
    st.write(f'Lasso - R-squared Score: {r2_lasso:.2f}')

# Plot feature importance
plt.figure(figsize=(14, 6))
if regression_type == 'Ridge' or regression_type == 'Both':
    feature_importance_ridge = pd.DataFrame({
        'feature': X.columns,
        'importance': abs(ridge_model.coef_)
    }).sort_values('importance', ascending=False)

if regression_type == 'Lasso' or regression_type == 'Both':
    feature_importance_lasso = pd.DataFrame({
        'feature': X.columns,
        'importance': abs(lasso_model.coef_)
    }).sort_values('importance', ascending=False)

if regression_type == 'Both':
    # Plot side-by-side bars
    x = np.arange(10)
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, feature_importance_ridge['importance'][:10], width, label='Ridge', color='#1E88E5')
    ax.bar(x + width/2, feature_importance_lasso['importance'][:10], width, label='Lasso', color='#FFC107')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_importance_ridge['feature'][:10], rotation=45, ha='right')
    ax.legend()
else:
    if regression_type == 'Ridge':
        plt.bar(feature_importance_ridge['feature'][:10], feature_importance_ridge['importance'][:10], 
                color='#1E88E5', alpha=0.7)
    else:
        plt.bar(feature_importance_lasso['feature'][:10], feature_importance_lasso['importance'][:10], 
                color='#FFC107', alpha=0.7)
    plt.xticks(rotation=45, ha='right')

plt.title(f'Top 10 Feature Importance - {regression_type}')
plt.tight_layout()
st.pyplot(plt)

# Plot predicted vs actual
if regression_type == 'Both':
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    ax1.scatter(y_test, y_pred_ridge, alpha=0.6, color='#1E88E5')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Price')
    ax1.set_ylabel('Predicted Price')
    ax1.set_title('Ridge: Predicted vs Actual')
    
    ax2.scatter(y_test, y_pred_lasso, alpha=0.6, color='#FFC107')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual Price')
    ax2.set_ylabel('Predicted Price')
    ax2.set_title('Lasso: Predicted vs Actual')
else:
    plt.figure(figsize=(10, 6))
    if regression_type == 'Ridge':
        plt.scatter(y_test, y_pred_ridge, alpha=0.6, color='#1E88E5')
    else:
        plt.scatter(y_test, y_pred_lasso, alpha=0.6, color='#FFC107')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Predicted vs Actual - {regression_type}')

plt.tight_layout()
st.pyplot(plt)

# Plot residuals
if regression_type == 'Both':
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    residuals_ridge = y_test - y_pred_ridge
    ax1.scatter(y_pred_ridge, residuals_ridge, alpha=0.6, color='#1E88E5')
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Price')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Ridge: Residuals Plot')
    
    residuals_lasso = y_test - y_pred_lasso
    ax2.scatter(y_pred_lasso, residuals_lasso, alpha=0.6, color='#FFC107')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Price')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Lasso: Residuals Plot')
else:
    plt.figure(figsize=(10, 6))
    if regression_type == 'Ridge':
        residuals_ridge = y_test - y_pred_ridge
        plt.scatter(y_pred_ridge, residuals_ridge, alpha=0.6, color='#1E88E5')
    else:
        residuals_lasso = y_test - y_pred_lasso
        plt.scatter(y_pred_lasso, residuals_lasso, alpha=0.6, color='#FFC107')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title(f'Residuals Plot - {regression_type}')

plt.tight_layout()
st.pyplot(plt)

st.write("""
---
## Wrapping Up

As we've seen, the problem of overfitting in machine learning is like preparing for a test by memorizing the material instead of actually understanding underlying principles. Our journey through Ridge and Lasso regression has shown us how to strike a balance between model complexity and generalization.

The **Housing Prices dataset** served as our real-world classroom, demonstrating the practical application of these techniques. Here’s a summary of some key takeaways:
* **Regularization techniques** like Ridge and Lasso can significantly improve model performance, especially in complex datasets with numerous features.
* The choice between **Ridge and Lasso** depends on the specific characteristics of your data and your modeling goals.
* **Cross-validation** is crucial for finding the optimal regularization strength (λ) and ensuring our model generalizes well to unseen data.

But our exploration doesn't end here. The world of machine learning is vast, and there are many more advanced techniques we can apply to further improve our housing price predictions:
* **Elastic Net**: This method combines Ridge and Lasso penalties, potentially offering the best of both worlds: balancing feature selection with handling multicollinearity. (5)
* **Polynomial Features**: We could explore adding polynomial terms to capture non-linear relationships between features and house prices.
* **Feature Engineering**: As stated before, we could create new features or transform current ones based on domain knowledge.

As we continue to investigate this dataset and apply more advanced techniques, we'll gain deeper insights into the factors that truly drive housing prices. Remember, the goal isn't just to predict prices accurately, but to **understand the underlying market dynamics**. 

After all, we don’t want to overfit our understanding of generalization!

So, let's keep our minds open, our regularization parameters tuned, and our passion for learning alive as we go about our data science adventures!

---
## References:
1. **Dataset**: [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
2. [IBM: Understanding Overfitting](https://www.ibm.com/topics/overfitting)
3. [DataRobot: Introduction to Loss Functions](https://www.datarobot.com/blog/introduction-to-loss-functions/)
4. [YouTube: Regression Analysis Tutorial](https://www.youtube.com/watch?v=Q81RR3yKn30&t=85s)
5. [Towards Data Science: The Power of Ridge Regression](https://towardsdatascience.com/the-power-of-ridge-regression-4281852a64d6)
6. [Medium: Elastic Net Regression](https://medium.com/@abhishekjainindore24/elastic-net-regression-combined-features-of-l1-and-l2-regularization-6181a660c3a5)
""")



