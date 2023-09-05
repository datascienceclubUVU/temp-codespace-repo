import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

training_data = pd.read_parquet('preprocessed_car_listings.parquet')

X = training_data.drop(columns=['price'])
y = training_data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

used_car_predictor = DecisionTreeRegressor()
used_car_predictor.fit(X_train, y_train)