import json
from pathlib import Path
import numpy as np
from typing import Tuple

def get_drift_angle_stokes(stokes1, stokes2):
    return np.arccos(
        np.clip(
            np.dot(stokes1 / np.linalg.norm(stokes1), stokes2 / np.linalg.norm(stokes2)),
            -1.0, 1.0
        )
    )

# Function for fidelity prediction based on angle difference
def fidelity_from_angle(angle_diff: float) -> float:
    return np.cos(angle_diff / 2) ** 2

def model(wind, sun):
    return wind*np.float64(8.226996379794377e-08) + \
            sun*np.float64(1.2441978018637697e-07) + \
                np.float64(6.987705632914636e-18)

_Q90_MODEL_CACHE = None


def _load_q90_model(path: str = "variance_model_q90.json"):
    """Load and cache the quantile (τ=0.90) variance model artifact."""
    global _Q90_MODEL_CACHE
    if _Q90_MODEL_CACHE is None:
        artifact_path = Path(__file__).with_name(path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Q90 model artifact not found at {artifact_path}")
        _Q90_MODEL_CACHE = json.loads(artifact_path.read_text())
    return _Q90_MODEL_CACHE


def model_q90(wind, sun, path: str = "variance_model_q90.json"):
    """Predict variance using the saved τ=0.90 quantile regression model."""
    artifact = _load_q90_model(path)
    coeffs = artifact["coefficients"]
    return (
        np.float64(coeffs["wind_squared"]) * wind
        + np.float64(coeffs["sunlight"]) * sun
        + np.float64(coeffs["offset"])
    )


class ModelPredictor:
    def __init__(self, time, initial_stokes=[1, 1, 1]):
        self.last_time = time
        self.last_drift_rate = 0.0
        self.last_stokes = initial_stokes
        self.last_drift_rate = None

    def start_recalibration(self, time, current_stokes):
        self.last_drift_rate = get_drift_angle_stokes(self.last_stokes, current_stokes) / (time - self.last_time)
        self.last_time = time

    def finish_recalibration(self, new_stokes):
        self.last_stokes = new_stokes

    def run_fidelity_check(self, current_stokes, time):
        pass

    def predict_fidelity(self, time, wind, sun, num_stds, last_drift_rate=None):
        time_diff = time - self.last_time
        
        if last_drift_rate is not None:
            predicted_drift = last_drift_rate * time_diff
        else:
            predicted_drift = self.last_drift_rate * time_diff

        # The angle diff (make this fidelity diff)
        predicted_stddev = model(wind, sun) * np.sqrt(time_diff)
        predicted_fidelity = fidelity_from_angle(predicted_drift + num_stds * predicted_stddev)

        return predicted_fidelity
        

        

        