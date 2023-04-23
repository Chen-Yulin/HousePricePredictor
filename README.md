# ECE4710J Project

<!--toc:start-->
- [ECE4710J Project](#ece4710j-project)
  - [Feature Engineering](#feature-engineering)
    - [Bedrooms](#bedrooms)
  - [Total Features](#total-features)
  - [Pipeline](#pipeline)
    - [ColumnTransformer](#columntransformer)
<!--toc:end-->

*Describe your what have you done in your project (feature engineering, models, etc.), section by section.*

## Feature Engineering

### Bedrooms

The information of bedrooms can be found in `Description`.
I use regular expression of extract the number of bedrooms and add them to be a new feature column.

## Total Features
- Bedrooms
- Building Square Feet
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

