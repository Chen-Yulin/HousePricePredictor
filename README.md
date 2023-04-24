# ECE4710J Project

<!--toc:start-->
- [ECE4710J Project](#ece4710j-project)
  - [Feature Engineering](#feature-engineering)
    - [Bedrooms](#bedrooms)
  - [Total Features](#total-features)
  - [Pipeline](#pipeline)
    - [ColumnTransformer](#columntransformer)
    - [RandomForestRegressor](#randomforestregressor)
<!--toc:end-->

*Describe your what have you done in your project (feature engineering, models, etc.), section by section.*

## Feature Engineering

### Bedrooms

The information of bedrooms can be found in `Description`.
I use regular expression of extract the number of bedrooms and add them to be a new feature column.

## Total Features
- Bedrooms
- Building Square Feet
- Land Square Feet
- Age Decade
- Garage Indicator
- Floodplain
- Road Proximity
- Sale Year
- Repair Condition
- Estimate (Building)
- Estimate (Land)
- Apartments
- Wall Material
- Basement
- Basement Finish

## Pipeline

### ColumnTransformer

Since `Random Forest Regressor` can divide the numeric data in a unlinear way, there is no need for one hot encoding these categorical features.
Therefore, I just let these data `passthrough`.

### RandomForestRegressor

Choose `n_estimators` as 100 to enhance the robustness of model and avoid overfitting on the training data

Choose `max_depth` as 15 to reduce the complexity of model and enhance generalization.

## Preprocessing on the training data

Remove outliers at a degree of 0.1, i.e. removing the head and tail 0.1% house price.
