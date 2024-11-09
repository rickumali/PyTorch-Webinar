import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_scaled = StandardScaler().fit_transform(cancer.data)
print("Original data (rows, features):", X_scaled.shape)

# TIME
# Generating to polynomial features is not that time consuming
poly = PolynomialFeatures(2)
X_poly = poly.fit_transform(X_scaled)
print("All polynomial features (order 2):", X_poly.shape)

# TIME
# A fairly generic random forest
rfc = RandomForestClassifier(max_depth=7, n_estimators=10, random_state=1)

# Do some work to pick the optimal number of features
# "Recursive feature elimination using cross-validation"
rfecv = RFECV(estimator=rfc, cv=5, n_jobs=-1)
X_poly_top = rfecv.fit_transform(X_poly, cancer.target)

# The "top" features selected for the model
print("Best polynomial features", X_poly_top.shape)

# TIME
# Do a train/test split on the "poly_top" features
X_train, X_test, y_train, y_test = train_test_split(
    X_poly_top, cancer.target, random_state=42)

# Train the selected RFC model
rfc = RandomForestClassifier(max_depth=7, n_estimators=10, random_state=1)
print("Test accuracy:", rfc.fit(X_train, y_train).score(X_test, y_test))
