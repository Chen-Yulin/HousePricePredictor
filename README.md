# ECE4710J Project

<!--toc:start-->
- [ECE4710J Project](#ece4710j-project)
  - [Feature Engineering](#feature-engineering)
    - [Bedrooms](#bedrooms)
  - [Total Features](#total-features)
  - [Pipeline](#pipeline)
    - [ColumnTransformer](#columntransformer)
      - [OneHotEncoder](#onehotencoder)
    - [StandardScaler](#standardscaler)
    - [GradientBoostingRegressor](#gradientboostingregressor)
  - [Preprocessing on the training data](#preprocessing-on-the-training-data)
<!--toc:end-->

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

```python
ct = ColumnTransformer([
        ('linear_num', "passthrough",["Building Square Feet","Bedrooms","Age Decade","Sale Year","Repair Condition",
                                      "Estimate (Building)","Estimate (Land)","Apartments","Land Square Feet"]),
        ('ohe', OneHotEncoder(handle_unknown='ignore'), ["Garage Indicator","Floodplain","Road Proximity",
                                  "Wall Material","Basement","Basement Finish","Sale Month of Year",
                                  "Pure Market Filter","Porch","Property Class"])
    ])
```

#### OneHotEncoder

For `GBRT`, to avoid misleading the model to think the category number has something to do with level, I onehotencode all the categorical feature and set `handle_unknown='ignore'` to handle unknown category name in test set.

For other numeric features I just let them `passthrough`

### StandardScaler

It is necessary to use standard scaler before the data is passed to the model, because `GBRT` is scaler sensitive and standard scale can help it to convergence faster. 

### GradientBoostingRegressor

I choose GradientBoostingRegressor as my model.

```python
ensemble.GradientBoostingRegressor(n_estimators=500, learning_rate=0.3)
```

Choose `n_estimators` as 500 to enhance the robustness of model and avoid overfitting on the training data

## Preprocessing on the training data

Remove outliers at a degree of 0.1, i.e. removing the head and tail 0.1% house price.

