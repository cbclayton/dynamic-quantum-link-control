import pandas as pd
import numpy as np
import os

from drift_predictor import DriftPredictor

model_path = os.path.join("models", "drift_prediction_model.npz")
drift_predictor = DriftPredictor(load_path=model_path)

def angle_between_stokes(stokes1, stokes2):
    # Calculate the angle between two stokes vectors
    stokes1_norm = stokes1 / np.linalg.norm(stokes1)
    stokes2_norm = stokes2 / np.linalg.norm(stokes2)
    dot_product = np.clip(np.dot(stokes1_norm, stokes2_norm), -1.0, 1.0)
    return np.arccos(dot_product)

def fidelity_between_stokes(stokes1, stokes2):
    theta = angle_between_stokes(stokes1, stokes2)
    return angle2fid(theta)

def angle2fid(theta):
    return 0.5 * (1 + np.cos(theta))

def fid2angle(fid):
    return np.arccos(2*fid - 1)


#################################################
############ Polarization prediction ############
#################################################
def polarization_fidelity_prediction(state, confidence, dt=None):
    if dt == None:
        # Estimate from the last check
        dt = (state.current_time - state.last_measurement_time()).total_seconds()
    if confidence:
        predicted_drift = _drift_prediction_confidence(state, confidence, dt)
    else:
        predicted_drift = _drift_prediction_historical(state, dt)
    
    last_measured_angle = fid2angle(state.last_measured_fidelity())
    fid = angle2fid(last_measured_angle + predicted_drift)
    return fid

def _drift_prediction_linear(dt):
    """
    Predict the polarization fidelity of the quantum link.
    """
    return 0.02 * dt # Prediction for fidelity decrease per second

def _drift_prediction_historical(state, dt):
    """
    Predict the polarization fidelity at the current time based on historical data.
    """
    raise NotImplementedError("Need to implement last_n_drift_rates in DriftTracker.")
    n = min(NUM_PREV_CALIBRATION_INTERVALS, len(drift_rates_rad))
    if n == 0:
        return _drift_prediction_linear(dt) # If no data, use constant linear prediction

    # Predicted drift rate is the mean of the previous n measured drift rates
    predicted_drift_rate = np.mean(drift_rates_rad[-n:])

    return predicted_drift_rate * dt

def _drift_prediction_confidence(state, confidence, dt):
    if state.drift_tracker.is_empty():
        return _drift_prediction_linear(dt)
    
    theta_prev = state.drift_tracker.recent_drift_angle()
    t_prev = state.drift_tracker.recent_drift_interval()
    theta = drift_predictor(theta_prev, t_prev, dt, percentile=confidence*100)
    
    return theta

def expected_transmission_fidelity_after_fid_check(state, settings):
    # If we were to do a fidelity check right now, what would be the expected F_trans up to our desired confidence?
    F_expected = polarization_fidelity_prediction(state, confidence=0.5)
    F_last_measured = state.last_measured_fidelity()
    theta_last_measured = fid2angle(F_last_measured)
    theta_expected = fid2angle(F_expected) - theta_last_measured
    t_prev = (state.current_time - state.last_measurement_time()).total_seconds()
    F_trans_expected_next = angle2fid(theta_last_measured + drift_predictor(theta_expected, t_prev, 0, percentile=settings.confidence*100)) # The value we will assume based on our confidence interval
    return F_trans_expected_next