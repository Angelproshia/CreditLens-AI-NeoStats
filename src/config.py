from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT / "data/application_train.csv"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "artifacts/model_bundle.joblib"))
TARGET = "TARGET"
ID_COLUMN = "SK_ID_CURR"

# Compact, business-relevant feature set keeps training and explanations fast.
PREFERRED_FEATURES = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_INCOME_TOTAL",
    "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH",
    "DAYS_EMPLOYED", "DAYS_ID_PUBLISH", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT_W_CITY", "OWN_CAR_AGE", "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE", "CODE_GENDER", "DAYS_LAST_PHONE_CHANGE",
]

