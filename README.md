# Used Car Price Prediction with Linear Regression

A machine learning project that predicts the **selling price of a used car** from its details (brand, mileage, engine size, body type and more), using linear regression in scikit-learn.

## The goal

Used car prices vary a lot. This project builds a model that estimates a fair price from a car's features, and shows which features push the price up or down.

## Dataset

4,345 used car listings with these columns:

| Column | Description |
|--------|-------------|
| `Price` | Selling price (the value to predict) |
| `Brand` | Audi, BMW, Mercedes-Benz, Mitsubishi, Renault, Toyota or Volkswagen |
| `Body` | Sedan, crossover, hatch, van, wagon or other |
| `Mileage` | Distance the car has driven |
| `EngineV` | Engine volume |
| `Engine Type` | Petrol, diesel, gas or other |
| `Registration` | Whether the car is registered (yes/no) |
| `Year` | Year of manufacture |
| `Model` | Car model (dropped: too many unique values to use directly) |

## What I did

1. **Handled missing values.** 172 prices and 150 engine volumes were missing, so I removed those rows (4,345 down to 4,025).
2. **Removed outliers.** I cut the top 1% of prices and mileages, the bottom 1% of years, and engine volumes above 6.5 (values like 99.99 are clearly data errors). 3,867 cars remained.
3. **Checked the linear regression assumptions.** Price was heavily skewed and its relationships with the other variables were curved. Taking the **log of price** made the relationships roughly linear.
4. **Checked multicollinearity.** `Year` had the highest VIF (10.4), meaning it overlaps strongly with mileage and engine volume, so I dropped it.
5. **Prepared the features.** Text columns became dummy variables, all features were standardized, and the data was split 80% training and 20% testing.
6. **Trained a linear regression** and compared its predictions with the real values.
7. **Tested on 774 unseen cars**, converting predictions from log values back to real prices.

## Results

- **R-squared on training data: 0.745.** The model explains about 75% of the variation in (log) price.
- **On the unseen test cars, the typical prediction is off by about 23%** (median). A quarter of predictions land within 12% of the real price.
- The average error is higher (about 36%) because a few predictions are far off.

**What affects the price most** (from the model weights):
- **Mileage** has the strongest effect: more miles means a lower price
- **Registered cars** and **larger engines** are priced higher
- **BMW, Mercedes-Benz and Audi** are priced highest, while **Renault and Mitsubishi** are lowest
- Diesel cars are priced higher than petrol and gas cars

## Conclusion

A simple linear regression captures the main drivers of used car prices, and mileage, registration and engine size matter most. Taking the log of price and removing outliers made the model much more reliable. It is a good baseline, but not accurate enough to price an individual car with confidence.

## Limitations and next steps

- Dropping `Year` removes useful information. Keeping it (for example as car age) and handling the overlap differently could improve accuracy.
- The notebook reports R-squared only on training data. Adding the test R-squared would give a fairer score.
- The scaler was fitted on all the data before splitting. Fitting it on the training data only is cleaner.
- Next steps: try regularization (Ridge or Lasso) and tree-based models such as Random Forest.

## How to run

1. Install the requirements:
   ```
   pip install numpy pandas matplotlib seaborn scikit-learn statsmodels jupyter
   ```
2. Keep `1.04. Real-life example.csv` in the same folder as the notebook
3. Open `Used_Car_Price_Prediction.ipynb` in Jupyter and run the cells from top to bottom

## Tech used

Python, pandas, NumPy, scikit-learn, statsmodels, Matplotlib, Seaborn, Jupyter Notebook
