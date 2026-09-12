"""Pytest session configuration and backward-compatibility shims."""

try:
    import xgboost as xgb
    if hasattr(xgb, "XGBClassifier") and not hasattr(xgb.XGBClassifier, "_estimator_type"):
        xgb.XGBClassifier._estimator_type = "classifier"
    if hasattr(xgb, "XGBRegressor") and not hasattr(xgb.XGBRegressor, "_estimator_type"):
        xgb.XGBRegressor._estimator_type = "regressor"
    if hasattr(xgb, "XGBModel") and not hasattr(xgb.XGBModel, "_estimator_type"):
        xgb.XGBModel._estimator_type = "classifier"
except ImportError:
    pass
