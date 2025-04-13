# filename: app.py

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from math import radians, cos, sin, asin, sqrt
import os

# --- Configuration ---
MODEL_DIR = "saved_models_multi"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
# Cluster model path - might be DBSCAN or KMeans depending on notebook outcome
CLUSTER_MODEL_PATH = os.path.join(MODEL_DIR, "best_cluster_model.joblib")
# <<< NEW: Path for the KNN predictor >>>
KNN_PREDICTOR_PATH = os.path.join(MODEL_DIR, "knn_cluster_predictor.joblib")

# Load Donor Data (Same as before)
try:
    DONOR_DATA_PATH = "synthetic_donors_profiled.csv"
    donors_df_unscaled = pd.read_csv(
        DONOR_DATA_PATH, parse_dates=["last_donation_date"]
    )
    # Re-engineer eligibility if needed (same logic as before)
    if "is_eligible" not in donors_df_unscaled.columns:
        donors_df_unscaled["last_donation_date"] = pd.to_datetime(
            donors_df_unscaled["last_donation_date"], errors="coerce"
        )
        if pd.api.types.is_datetime64_any_dtype(
            donors_df_unscaled["last_donation_date"]
        ):
            now_ts = pd.Timestamp.now()
            time_diff_series = now_ts - donors_df_unscaled["last_donation_date"]
            donors_df_unscaled["days_since_last_donation"] = (
                time_diff_series.dt.days.fillna(9999)
            )
            eligibility_days = 56
            donors_df_unscaled["days_since_last_donation"] = pd.to_numeric(
                donors_df_unscaled["days_since_last_donation"]
            )
            donors_df_unscaled["is_eligible"] = (
                donors_df_unscaled["days_since_last_donation"] > eligibility_days
            )
        else:
            donors_df_unscaled["is_eligible"] = True
    # Ensure other needed columns exist
    for col in [
        "latitude",
        "longitude",
        "blood_type",
        "age",
        "days_since_last_donation",
        "is_regular_donor",
    ]:
        if col not in donors_df_unscaled.columns:
            donors_df_unscaled[col] = None
    print(f"Loaded {len(donors_df_unscaled)} donors from CSV.")
except FileNotFoundError:
    print(f"Error: Donor data file not found at {DONOR_DATA_PATH}")
    donors_df_unscaled = pd.DataFrame()

# --- Load Models ---
scaler = None
cluster_model = None  # The original cluster model (KMeans or DBSCAN)
knn_predictor = None  # The KNN model to predict cluster labels
SCALER_FEATURES = []

try:
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded successfully.")
    if hasattr(scaler, "feature_names_in_"):
        SCALER_FEATURES = scaler.feature_names_in_
    else:
        SCALER_FEATURES = [
            "latitude",
            "longitude",
            "age",
            "days_since_last_donation",
        ]  # Fallback

    # Try loading the original best cluster model (for context/info if needed)
    if os.path.exists(CLUSTER_MODEL_PATH):
        cluster_model = joblib.load(CLUSTER_MODEL_PATH)
        print(f"Original Cluster Model ({type(cluster_model).__name__}) loaded.")

    # <<< NEW: Load the KNN Predictor >>>
    if os.path.exists(KNN_PREDICTOR_PATH):
        knn_predictor = joblib.load(KNN_PREDICTOR_PATH)
        print("KNN Cluster Predictor loaded successfully.")
    else:
        print(
            "KNN Cluster Predictor model not found (expected if best model wasn't DBSCAN)."
        )

except FileNotFoundError:
    print("Error: Scaler or required model file not found.")
except Exception as e:
    print(f"Error loading models: {e}")


# --- Helper Functions (Haversine, is_compatible) ---
# (Keep the functions as defined before)
# --- Helper Functions (Haversine, is_compatible) ---
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers.
    return c * r


compatibility = {
    # Donor Type -> List of acceptable Recipient Types
    "O-": ["O-", "O+", "B-", "B+", "A-", "A+", "AB-", "AB+"],
    "O+": ["O+", "B+", "A+", "AB+"],
    "B-": ["B-", "B+", "AB-", "AB+"],
    "B+": ["B+", "AB+"],
    "A-": ["A-", "A+", "AB-", "AB+"],
    "A+": ["A+", "AB+"],
    "AB-": ["AB-", "AB+"],
    "AB+": ["AB+"],
}


def is_compatible(donor_type, requested_type):
    """Checks if donor blood type is compatible with the requested type."""
    if requested_type == "ALL":
        return True
    if donor_type == requested_type:
        return True
    if donor_type == "O-":
        return True  # Universal donor
    # Simplified compatibility checks (adjust if more complex logic is needed)
    if requested_type.endswith("+"):
        if donor_type == requested_type[:-1] + "-":
            return True
    if requested_type == "A+" and donor_type in ["A-", "O-", "O+"]:
        return True
    if requested_type == "B+" and donor_type in ["B-", "O-", "O+"]:
        return True
    if requested_type == "AB+" and donor_type in [
        "A-",
        "B-",
        "O-",
        "O+",
        "A+",
        "B+",
        "AB-",
    ]:
        return True  # Approx Universal Recipient
    if requested_type == "O+" and donor_type == "O-":
        return True
    if requested_type == "A-" and donor_type == "O-":
        return True
    if requested_type == "B-" and donor_type == "O-":
        return True
    if requested_type == "AB-" and donor_type in ["A-", "B-", "O-"]:
        return True
    return False


# --- Matching Function ---
# --- Matching Function ---
def find_matching_donors(
    bank_need_row, all_donors_df_unscaled, max_distance_km=50, top_n=20
):
    """
    Finds and ranks eligible, compatible donors for a bank need.
    Assumes input dataframe has original scale features needed.
    """
    bank_lat = bank_need_row["latitude"]
    bank_lon = bank_need_row["longitude"]
    required_type = bank_need_row["required_blood_type"]
    required_units = bank_need_row["required_units"]

    candidate_donors = all_donors_df_unscaled.copy()  # Work on a copy

    # Filters (Eligibility, Compatibility, Availability)
    # (Filters remain the same - ensure columns exist first)
    if "is_eligible" not in candidate_donors.columns:
        return pd.DataFrame()
    candidate_donors = candidate_donors[candidate_donors["is_eligible"]]
    if candidate_donors.empty:
        return pd.DataFrame()

    if "blood_type" not in candidate_donors.columns:
        return pd.DataFrame()
    candidate_donors["compatible"] = candidate_donors["blood_type"].apply(
        lambda dt: is_compatible(dt, required_type)
    )
    candidate_donors = candidate_donors[candidate_donors["compatible"]]
    if candidate_donors.empty:
        return pd.DataFrame()

    if "availability_status" in candidate_donors.columns:
        candidate_donors = candidate_donors[
            candidate_donors["availability_status"] == True
        ]
        if candidate_donors.empty:
            return pd.DataFrame()

    # Calculate Distance
    if not all(col in candidate_donors.columns for col in ["latitude", "longitude"]):
        return pd.DataFrame()
    candidate_donors["distance_km"] = candidate_donors.apply(
        lambda donor: haversine(
            bank_lon, bank_lat, donor["longitude"], donor["latitude"]
        ),
        axis=1,
    )

    # Filter by Distance Radius & Rank
    nearby_donors = candidate_donors[
        candidate_donors["distance_km"] <= max_distance_km
    ].copy()
    if nearby_donors.empty:
        return pd.DataFrame()

    rank_cols = ["distance_km"]
    if "is_regular_donor" in nearby_donors.columns:
        rank_cols.append("is_regular_donor")
    if "days_since_last_donation" in nearby_donors.columns:
        rank_cols.append("days_since_last_donation")
    ascending_flags = [True]
    if "is_regular_donor" in rank_cols:
        ascending_flags.append(False)
    if "days_since_last_donation" in rank_cols:
        ascending_flags.append(False)

    # Ensure all ranking columns actually exist before sorting
    rank_cols_exist = [col for col in rank_cols if col in nearby_donors.columns]
    if len(rank_cols_exist) != len(rank_cols):
        print(
            f"Warning: Missing some ranking columns. Sorting only by {rank_cols_exist}"
        )
        # Adjust ascending_flags if needed based on which columns remain
        if "is_regular_donor" not in rank_cols_exist and len(ascending_flags) > 1:
            ascending_flags.pop(1)
        if (
            "days_since_last_donation" not in rank_cols_exist
            and len(ascending_flags) > 1
        ):
            ascending_flags.pop(-1)

    nearby_donors.sort_values(
        by=rank_cols_exist, ascending=ascending_flags, inplace=True
    )

    # Return Top N
    num_to_return = max(top_n, int(required_units * 1.5))
    num_to_return = min(num_to_return, top_n)

    # <<< CORRECTED output_columns LIST >>>
    # Select columns NEEDED later (for scaling/prediction) AND columns for display
    output_columns = [
        "donor_id",
        "latitude",
        "longitude",
        "blood_type",
        "distance_km",
        "age",
        "days_since_last_donation",  # ADDED columns needed for scaling
    ]
    # <<< END CORRECTION >>>

    # Ensure only existing columns are selected
    nearby_donors_output = nearby_donors[
        [col for col in output_columns if col in nearby_donors.columns]
    ]
    return nearby_donors_output.head(num_to_return)


# --- Flask App ---
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route('/match', methods=['POST'])
def match_donors_api():
    # Check if essential components are loaded
    if donors_df_unscaled.empty:
        return jsonify({"error": "System not ready. Donor data missing."}), 500
    if scaler is None:
        return jsonify({"error": "System not ready. Scaler missing."}), 500
    if knn_predictor is None:
        print("Warning: KNN Predictor not loaded. Cannot predict cluster labels.")

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        required_fields = ['latitude', 'longitude', 'required_blood_type', 'required_units']
        if not all(field in data for field in required_fields):
            return jsonify({
                "error": "Missing required fields",
                "required": required_fields
            }), 400

        # Create a bank need dict for the function
        bank_need = {
            'latitude': float(data['latitude']),
            'longitude': float(data['longitude']),
            'required_blood_type': str(data['required_blood_type']),
            'required_units': int(data['required_units'])
        }

        # Core Matching Logic
        matched_donors_df = find_matching_donors(bank_need, donors_df_unscaled)

        # Add default cluster column
        if not matched_donors_df.empty:
            matched_donors_df['cluster'] = -100  # Default: not predicted

        # Proceed with ML prediction only if everything is available
        if (
            not matched_donors_df.empty and
            knn_predictor is not None and
            scaler is not None and
            len(SCALER_FEATURES) > 0
        ):
            try:
                # Ensure all required features are present
                missing_features = [f for f in SCALER_FEATURES if f not in matched_donors_df.columns]
                if missing_features:
                    app.logger.error(f"Missing features for scaling/prediction: {missing_features}")
                    matched_donors_df['cluster'] = -98
                else:
                    donor_features = matched_donors_df[SCALER_FEATURES].copy()

                    # Impute NaNs
                    if donor_features.isnull().values.any():
                        print(f"NaNs found. Imputing with median...")
                        for col in donor_features.columns:
                            if donor_features[col].isnull().any():
                                median_val = donor_features[col].median()
                                if pd.isna(median_val):
                                    median_val = 0
                                    print(f"Warning: Median for '{col}' is NaN, filling with 0.")
                                donor_features[col].fillna(median_val, inplace=True)

                    # Double check no NaNs remain
                    if donor_features.isnull().values.any():
                        app.logger.error("NaNs still present after imputation!")
                        matched_donors_df['cluster'] = -97
                    else:
                        # Scale and predict clusters
                        matched_donors_scaled = scaler.transform(donor_features)
                        predicted_clusters = knn_predictor.predict(matched_donors_scaled)
                        matched_donors_df['cluster'] = predicted_clusters
                        print(f"Predicted cluster labels for {len(matched_donors_df)} donors.")

            except Exception as e:
                app.logger.error(f"Error predicting cluster labels: {e}", exc_info=True)
                matched_donors_df['cluster'] = -99

        elif not matched_donors_df.empty:
            if knn_predictor is None:
                print("KNN Predictor not loaded, cannot assign cluster labels.")
            elif not SCALER_FEATURES:
                print("Scaler feature list is empty, cannot assign cluster labels.")

        # Prepare Final Output
        output_columns = ['donor_id', 'latitude', 'longitude', 'blood_type', 'distance_km', 'cluster']
        if 'cluster' not in matched_donors_df.columns and not matched_donors_df.empty:
            matched_donors_df['cluster'] = -101

        result = matched_donors_df[
            [col for col in output_columns if col in matched_donors_df.columns]
        ].to_dict(orient='records')

        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Error during matching request processing: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during matching."}), 500



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
