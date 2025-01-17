import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('dataset.csv')

df['AvgPricePerUnit'] = df['Total Price'] / df['Units']

label_encoder = LabelEncoder()
df['AllocationEncoded'] = label_encoder.fit_transform(df['Allocation'])

X = df[['Total Price', 'Units', 'AvgPricePerUnit', 'AllocationEncoded']]
y = df['Total Sales']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# X_future = ...
# future_sales_predictions = model.predict(X_future)
