# ============================================================
# 🍽️ RESTAURANT RATING PREDICTION SYSTEM
# ML CA2 PROJECT
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("Libraries imported successfully! ✅")


# ============================================================
# 2. CREATE RESTAURANT DATASET
# ============================================================

np.random.seed(42)

n = 500

cuisines = [
    "Indian",
    "Chinese",
    "Italian",
    "Fast Food",
    "Mexican",
    "Continental"
]

locations = [
    "Mumbai",
    "Pune",
    "Delhi",
    "Bangalore",
    "Hyderabad",
    "Chennai"
]

data = pd.DataFrame({
    "Cuisine": np.random.choice(cuisines, n),
    "Location": np.random.choice(locations, n),
    "Cost_for_Two": np.random.randint(200, 2501, n),
    "Votes": np.random.randint(20, 5001, n),
    "Online_Delivery": np.random.choice([0, 1], n),
    "Table_Booking": np.random.choice([0, 1], n),
    "Preparation_Time": np.random.randint(10, 61, n)
})


# ============================================================
# 3. CREATE RATING COLUMN
# ============================================================

rating = (
    2.5
    + 0.00025 * data["Votes"]
    + 0.00015 * data["Cost_for_Two"]
    + 0.25 * data["Online_Delivery"]
    + 0.15 * data["Table_Booking"]
    - 0.008 * data["Preparation_Time"]
    + np.random.normal(0, 0.20, n)
)

data["Rating"] = np.clip(rating, 1.0, 5.0)


# ============================================================
# 4. DISPLAY DATASET
# ============================================================

print("\n" + "=" * 60)
print("RESTAURANT DATASET")
print("=" * 60)

print(data.head(10))

print("\nDataset Shape:")
print(data.shape)


# ============================================================
# 5. DATA CLEANING
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

print("Missing values before cleaning:")
print(data.isnull().sum())

data = data.drop_duplicates()
data = data.dropna()

print("\nMissing values after cleaning:")
print(data.isnull().sum())

print("\nData cleaning completed! ✅")


# ============================================================
# 6. ENCODE CATEGORICAL VARIABLES
# ============================================================

cuisine_encoder = LabelEncoder()
location_encoder = LabelEncoder()

data["Cuisine_Code"] = cuisine_encoder.fit_transform(
    data["Cuisine"]
)

data["Location_Code"] = location_encoder.fit_transform(
    data["Location"]
)

print("\n" + "=" * 60)
print("ENCODED DATA")
print("=" * 60)

print(data.head())


# ============================================================
# 7. SELECT FEATURES
# ============================================================

features = [
    "Cuisine_Code",
    "Location_Code",
    "Cost_for_Two",
    "Votes",
    "Online_Delivery",
    "Table_Booking",
    "Preparation_Time"
]

X = data[features]
y = data["Rating"]

print("\nFeatures used for prediction:")
print(features)

print("\nTarget variable:")
print("Rating")


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 9. CREATE MACHINE LEARNING MODELS
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=8,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
}


# ============================================================
# 10. TRAIN MODELS AND EVALUATE
# ============================================================

results = []
trained_models = {}
predictions = {}

for model_name, model in models.items():

    # Train model
    model.fit(X_train, y_train)

    # Predict
    pred = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    r2 = r2_score(
        y_test,
        pred
    )

    results.append([
        model_name,
        mae,
        rmse,
        r2
    ])

    trained_models[model_name] = model
    predictions[model_name] = pred


# ============================================================
# 11. MODEL EVALUATION TABLE
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2 Score"
    ]
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(
    results_df.round(4).to_string(index=False)
)


# ============================================================
# 12. FIND BEST MODEL
# ============================================================

best_index = results_df["R2 Score"].idxmax()

best_model_name = results_df.loc[
    best_index,
    "Model"
]

best_model = trained_models[
    best_model_name
]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Best Model:", best_model_name)

print(
    "Best R2 Score:",
    round(
        results_df.loc[
            best_index,
            "R2 Score"
        ],
        4
    )
)


# ============================================================
# 13. MODEL COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Model"],
    results_df["R2 Score"]
)

plt.xlabel("Machine Learning Model")
plt.ylabel("R² Score")

plt.title(
    "Restaurant Rating Prediction - Model Comparison"
)

plt.xticks(rotation=15)

plt.tight_layout()
plt.show()


# ============================================================
# 14. ACTUAL VS PREDICTED VALUES
# ============================================================

best_predictions = predictions[
    best_model_name
]

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.6
)

plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")

plt.title(
    "Actual Rating vs Predicted Rating"
)

plt.tight_layout()
plt.show()


# ============================================================
# 15. RATING DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    data["Rating"],
    bins=20
)

plt.xlabel("Restaurant Rating")
plt.ylabel("Number of Restaurants")

plt.title(
    "Restaurant Rating Distribution"
)

plt.tight_layout()
plt.show()


# ============================================================
# 16. COST VS RATING
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    data["Cost_for_Two"],
    data["Rating"],
    alpha=0.5
)

plt.xlabel("Average Cost for Two (₹)")
plt.ylabel("Restaurant Rating")

plt.title(
    "Cost for Two vs Restaurant Rating"
)

plt.tight_layout()
plt.show()


# ============================================================
# 17. VOTES VS RATING
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    data["Votes"],
    data["Rating"],
    alpha=0.5
)

plt.xlabel("Number of Votes")
plt.ylabel("Restaurant Rating")

plt.title(
    "Votes vs Restaurant Rating"
)

plt.tight_layout()
plt.show()


# ============================================================
# 18. FEATURE IMPORTANCE
# ============================================================

if best_model_name == "Random Forest":

    importance = best_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    print(
        importance_df.round(4).to_string(index=False)
    )

    plt.figure(figsize=(9, 5))

    plt.bar(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.xlabel("Features")
    plt.ylabel("Importance")

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ============================================================
# 19. RESTAURANT RATING PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("RESTAURANT RATING PREDICTION")
print("=" * 60)


# ------------------------------------------------------------
# USER INPUT
# ------------------------------------------------------------

user_cuisine = "Indian"

user_location = "Mumbai"

user_cost = 600

user_votes = 1000

user_online_delivery = 1

user_table_booking = 1

user_preparation_time = 30


# ============================================================
# 20. ENCODE USER INPUT
# ============================================================

cuisine_code = cuisine_encoder.transform(
    [user_cuisine]
)[0]

location_code = location_encoder.transform(
    [user_location]
)[0]


# ============================================================
# 21. CREATE INPUT DATA
# ============================================================

user_input = pd.DataFrame({
    "Cuisine_Code": [cuisine_code],
    "Location_Code": [location_code],
    "Cost_for_Two": [user_cost],
    "Votes": [user_votes],
    "Online_Delivery": [user_online_delivery],
    "Table_Booking": [user_table_booking],
    "Preparation_Time": [user_preparation_time]
})


# ============================================================
# 22. PREDICT RATING
# ============================================================

predicted_rating = best_model.predict(
    user_input
)[0]

predicted_rating = np.clip(
    predicted_rating,
    1.0,
    5.0
)


# ============================================================
# 23. DISPLAY PREDICTION
# ============================================================

print("\nRestaurant Details")
print("-" * 40)

print(
    "Cuisine:",
    user_cuisine
)

print(
    "Location:",
    user_location
)

print(
    "Average Cost for Two: ₹",
    user_cost
)

print(
    "Number of Votes:",
    user_votes
)

print(
    "Online Delivery:",
    "Yes" if user_online_delivery == 1 else "No"
)

print(
    "Table Booking:",
    "Yes" if user_table_booking == 1 else "No"
)

print(
    "Preparation Time:",
    user_preparation_time,
    "minutes"
)

print("\n" + "-" * 40)

print(
    "⭐ Predicted Restaurant Rating:",
    round(predicted_rating, 2),
    "/ 5.0"
)


# ============================================================
# 24. RATING INTERPRETATION
# ============================================================

if predicted_rating >= 4.5:

    print("🌟 Excellent Restaurant")

elif predicted_rating >= 4.0:

    print("😊 Very Good Restaurant")

elif predicted_rating >= 3.0:

    print("👍 Good Restaurant")

elif predicted_rating >= 2.0:

    print("😐 Average Restaurant")

else:

    print("⚠️ Low Rated Restaurant")


# ============================================================
# 25. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print("""
Project Title:
Restaurant Rating Prediction System Using Machine Learning

Machine Learning Algorithms:
1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression

Evaluation Metrics:
1. MAE - Mean Absolute Error
2. RMSE - Root Mean Squared Error
3. R² Score

Target Variable:
Restaurant Rating

The system predicts a restaurant's expected rating
based on restaurant-related features.
""")

print("Project completed successfully! ✅")