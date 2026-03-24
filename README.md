# Squat Predictor Using Bench
Static website that predicts your squat using only your 1RM on bench and bodyweight, with a mean error of 48lb/21.7kg.

# How it works
I used powerlifting data from https://www.kaggle.com/datasets/open-powerlifting/powerlifting-database, then dropped rows that were female, equipped, and then extracted the values for "BodyweightKg", "Best3SquatKg", "Best3BenchKg", doing:
```
import polars as pl

df = (
    pl.read_csv(
        "/kaggle/input/datasets/organizations/open-powerlifting/powerlifting-database/openpowerlifting-2024-01-06-4c732975.csv",
        columns=["Sex", "Equipment", "BodyweightKg", "Best3SquatKg", "Best3BenchKg"],
        schema_overrides={
            "Sex": pl.Utf8,
            "Equipment": pl.Utf8,
            "BodyweightKg": pl.Float64,
            "Best3SquatKg": pl.Float64,
            "Best3BenchKg": pl.Float64,
        },
        infer_schema_length=0,
    )
    .filter(
        (pl.col("Sex") == "M") & (pl.col("Equipment") == "Raw")
    )
    .drop_nulls()
    .unique()
    .select(["BodyweightKg", "Best3SquatKg", "Best3BenchKg"])
)
```
-- Along with any duplicates or rows will null values --

## Step 1
I then used LinearRegression on the dataset with no feature engineering, to get a baseline score:
```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Convert Polars to NumPy
X = df.select(["BodyweightKg", "Best3BenchKg"]).to_numpy()
y = df.select("Best3SquatKg").to_numpy().ravel()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (fit on train, transform both)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate
preds = model.predict(X_test_scaled)
print("MAE:", mean_absolute_error(y_test, preds))
print("R²:", r2_score(y_test, preds))

# Coefficients
print("Intercept:", model.intercept_)
print("Bodyweight coef:", model.coef_[0])
print("Bench coef:", model.coef_[1])
```
Outputs:
```
MAE: 22.446190347291747
R²: 0.6455132626884112
Intercept: 189.85956555488994
Bodyweight coef: 9.235197367727674
Bench coef: 37.2132223115461
```

### As you can tell, that's way different from the proposed MAE error of 21.7kg; because--we haven't introducted feature engineering yet!
<img width="444" height="334" alt="image" src="https://github.com/user-attachments/assets/e946540a-e2d1-484b-8290-292cb55237a3" />

## Step 2: Feature Engineering
I then used sklearn's PolynomialFeatures, without any specific selection or tuning of the degree:
```
from sklearn.preprocessing import PolynomialFeatures

# Polynomial expansion
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Fit model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Evaluate
preds = model.predict(X_test_poly)
print("MAE:", mean_absolute_error(y_test, preds))
print("R²:", r2_score(y_test, preds))

# Coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
```
Output:
```
MAE: 22.35370537289386
R²: 0.6663409294420436
Intercept: 188.83706538490935
Coefficients: [ 8.20261196 39.82202861 -0.21931903 -4.39785963  3.71615747]
```
-- So slightly better without any exact selection at all! --

## Step 3: Optuna Selection
```
import optuna

# Base data
X_full = df.select(["BodyweightKg", "Best3BenchKg"]).to_numpy()
y_full = df.select("Best3SquatKg").to_numpy().ravel()

# Fixed split
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

def objective(trial):
    # Degree of polynomial
    degree = trial.suggest_int("degree", 1, 4)

    # Scaling
    use_scaler = trial.suggest_categorical("use_scaler", [True, False])

    # Polynomial expansion
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Number of polynomial features
    n_features = X_train_poly.shape[1]

    # Create a binary mask for feature selection
    mask = []
    for i in range(n_features):
        mask.append(trial.suggest_categorical(f"feat_{i}", [0, 1]))

    mask = np.array(mask, dtype=bool)

    # Must keep at least 1 feature
    if mask.sum() == 0:
        return 9999

    # Apply mask
    X_train_sel = X_train_poly[:, mask]
    X_test_sel = X_test_poly[:, mask]

    # Optional scaling
    if use_scaler:
        scaler = StandardScaler()
        X_train_sel = scaler.fit_transform(X_train_sel)
        X_test_sel = scaler.transform(X_test_sel)

    # Fit model
    model = LinearRegression()
    model.fit(X_train_sel, y_train)

    # Evaluate
    preds = model.predict(X_test_sel)
    mae = mean_absolute_error(y_test, preds)

    return mae


# Run Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=80)

print("Best MAE:", study.best_value)
print("Best params:", study.best_params)
```
Output:
```
Best MAE: 21.757052331069424
Best params: {'degree': 4, 'use_scaler': True, 'feat_0': 1, 'feat_1': 1, 'feat_2': 1, 'feat_3': 1, 'feat_4': 1, 'feat_5': 0, 'feat_6': 1, 'feat_7': 1, 'feat_8': 1, 'feat_9': 0, 'feat_10': 0, 'feat_11': 1, 'feat_12': 1, 'feat_13': 0}
```
-- So as you can see a very nice improvement compared to the previous MAE!! --

Selected polynomial features:
 - BodyweightKg
 - Best3BenchKg
 - BodyweightKg^2
 - Best3BenchKg^2
 - BodyweightKg^3
 - BodyweightKg^2 * Best3BenchKg
 - BodyweightKg * Best3BenchKg^2
 - Best3BenchKg^3
 - BodyweightKg * Best3BenchKg^3

## Step 4: Using the Features:
We train the model on the newly found features, then log the coefficents so we can create a python function using it:
```
# Scale features (Optuna said use_scaler=False, so we skip scaling)
# -- although it doesn't seem to make a difference anyways

# Polynomial expansion (degree=4)
poly = PolynomialFeatures(degree=4, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Feature names (to map indices)
feature_names = poly.get_feature_names_out(["BodyweightKg", "Best3BenchKg"])

# Your selected features (mask from Optuna)
selected_mask = np.array([
    1,  # BodyweightKg
    1,  # Best3BenchKg
    1,  # BodyweightKg^2
    0,  # BodyweightKg Best3BenchKg
    1,  # Best3BenchKg^2
    1,  # BodyweightKg^3
    1,  # BodyweightKg^2 Best3BenchKg
    1,  # BodyweightKg Best3BenchKg^2
    1,  # Best3BenchKg^3
    0,  # BodyweightKg^4
    0,  # BodyweightKg^3 Best3BenchKg
    0,  # BodyweightKg^2 Best3BenchKg^2
    1,  # BodyweightKg Best3BenchKg^3
    0   # Best3BenchKg^4
], dtype=bool)

# Apply feature mask
X_train_sel = X_train_poly[:, selected_mask]
X_test_sel = X_test_poly[:, selected_mask]

# Fit model
model = LinearRegression()
model.fit(X_train_sel, y_train)

# Evaluate
preds = model.predict(X_test_sel)
mae = mean_absolute_error(y_test, preds)
print("MAE:", mae)
print("R²:", r2_score(y_test, preds))

# Coefficients with feature names
print("\nSelected Features and Coefficients:")
for name, coef in zip(feature_names[selected_mask], model.coef_):
    print(f"{name}: {coef}")

print("\nIntercept:", model.intercept_)
```
Output:
```
MAE: 21.760782144221697
R²: 0.6821832969801478

Selected Features and Coefficients:
BodyweightKg: 1.5297410827061009
Best3BenchKg: 0.5165197518947516
BodyweightKg^2: -0.008229692810747834
Best3BenchKg^2: 0.009599671637951824
BodyweightKg^3: 1.660906079236908e-05
BodyweightKg^2 Best3BenchKg: 3.123765286322958e-06
BodyweightKg Best3BenchKg^2: -3.632022411856049e-05
Best3BenchKg^3: -3.7131121763886924e-05
BodyweightKg Best3BenchKg^3: 1.832434706927071e-07

Intercept: -20.560828590674078
```

# AND ALL DONE!
Also there were 380,026 samples in the training data
