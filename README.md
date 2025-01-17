# Code-for-Predicting-Future-Sales
**Overview**
This project predicts total sales for truck shipments using a Random Forest Regressor model. The model uses features like total price, units, average price per unit, and allocation to predict sales.

**Dataset**

The dataset (dataset.csv) includes:

Total Price: Total shipment price

Units: Number of units in the shipment

Allocation: Product category (e.g., "HV", "MAS")

Total Sales: Target variable (total sales)

**Features**

AvgPricePerUnit: Calculated as Total Price / Units

AllocationEncoded: Categorical Allocation encoded numerically

**Steps**

Preprocessing: Derive AvgPricePerUnit and encode Allocation.

Model: Train a Random Forest Regressor.

Evaluation: Use Mean Squared Error (MSE) to assess performance.

Future Predictions: Predict future sales with the trained model.

**Requirements**
Install dependencies:
pip install pandas numpy scikit-learn
