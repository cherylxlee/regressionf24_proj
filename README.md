# Regularize Your Real Estate: Ridge and Lasso Regression

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)

An interactive Streamlit application that explains and demonstrates Ridge and Lasso regression techniques with a focus on real estate price prediction.

## Overview

This application serves as both an educational tool and a practical demonstration of regularization techniques in machine learning. It's designed to help users understand the concepts of Ridge and Lasso regression and how they can be applied to solve real-world problems like predicting housing prices.

The app consists of two main parts:
1. **Educational Content**: Clear explanations of concepts like overfitting, regularization, coefficient shrinkage, and the mathematics behind Ridge and Lasso regression.
2. **Interactive Demo**: A hands-on section where users can adjust regularization parameters and observe their effects on model performance and feature importance.

## Features

- **Comprehensive explanations** of Ridge and Lasso regression with visualizations
- **Interactive controls** to adjust regularization strength and compare models
- **Real-time visualizations** showing:
  - Feature importance
  - Predicted vs. actual values
  - Residual plots
- **Performance metrics** including Mean Squared Error (MSE) and R-squared
- **Multicollinearity analysis** using Variance Inflation Factor (VIF)
- **Practical guidance** on when to use Ridge vs. Lasso

## Dataset

The application uses the [Kaggle Housing Prices dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) which contains information on residential homes in Ames, Iowa. The dataset includes 79 explanatory variables covering:

- Structural features (year built, square footage, bedrooms, etc.)
- Location information
- Quality metrics
- Various amenities and features

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/regularize-real-estate.git
cd regularize-real-estate

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Statsmodels
- Patsy

## Usage

```bash
streamlit run app.py
```

This will start the Streamlit server and open the application in your default web browser.

## Learning Resources

The application references several learning resources:

1. [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
2. [IBM: Understanding Overfitting](https://www.ibm.com/topics/overfitting)
3. [DataRobot: Introduction to Loss Functions](https://www.datarobot.com/blog/introduction-to-loss-functions/)
4. [YouTube: Regression Analysis Tutorial](https://www.youtube.com/watch?v=Q81RR3yKn30&t=85s)
5. [Towards Data Science: The Power of Ridge Regression](https://towardsdatascience.com/the-power-of-ridge-regression-4281852a64d6)
6. [Medium: Elastic Net Regression](https://medium.com/@abhishekjainindore24/elastic-net-regression-combined-features-of-l1-and-l2-regularization-6181a660c3a5)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
