import pandas as pd
import numpy as np
from enum import Enum
import os
from filelock import FileLock


from hardware import SOURCE_RATE, FID_CHECK_DURATION
from hardware import MIN_PUMP_POWER_MW, MAX_PUMP_POWER_MW
from hardware import fidelity_sd, p_success_sd, p_success_transmission, power_from_fid, power_from_rate, fid_to_rate
from polarization_drift import polarization_fidelity_prediction, angle_between_stokes, fid2angle, angle2fid
from polarization_drift import expected_transmission_fidelity_after_fid_check, drift_predictor
from apc_sim import predict_compensation_time, simulate_apc
from utils import load_data, progressBar
from plot_sim import plot_simulation, plot_simulation_detailed, plot_calibration_durs

PARAM_UPDATE_INTERVAL = .1 # in seconds
MIN_FID_CHECK_INTERVAL = .5 # DELETE
N_FIT_POINTS = 5 # Minimum number of points required to estimate rate gradient


class SimulationSettings:
    def __init__(self, **kwargs):
        self.sim_time_seconds = kwargs['sim_time_seconds']
        self.resample_ms = kwargs['resample_ms']
        self.adaptive = kwargs['adaptive']
        self.label = kwargs['label']
        self._F_min = kwargs['F_min']
        self.source_fidelity = kwargs['source_fidelity']
        self.fid_check_freq = kwargs['fid_check_freq']
        self.F_trigger = kwargs['F_trigger']
        self.F_target = kwargs['F_target']
        self.timeout = kwargs['timeout']
        self.confidence = kwargs['confidence']
        self.labels = ['label','sim_time_seconds', 'resample_ms', 'F_min', 'adaptive', 'source_fidelity', 'fid_check_freq', 'F_trigger', 'F_target', 'timeout', 'confidence']
        self.arg_check()
        if not isinstance(self._F_min, (int, float)):
            self._init_variable_Fmin(kwargs['data'])
    
    def settings_dict(self):
        d = {label: getattr(self, label) for label in self.labels}
        d['F_min'] = self._F_min if isinstance(self._F_min, (int, float)) else None
        return d
    
    def arg_check(self):
        assert self._F_min == 'sine' or 0 <= self._F_min <= 1
        if self.adaptive:
            assert self.source_fidelity == None
            assert self.fid_check_freq == None
            assert self.F_trigger == None
            assert self.F_target == 1
            assert 0 < self.timeout
            assert 0 < self.confidence < 1
        else:
            assert 0 <= self.source_fidelity <= 1
            assert 0 <= self.fid_check_freq 
            assert 0 <= self.F_trigger <=1
            assert 0 <= self.F_target <= 1
            assert 0 < self.timeout
            assert self.confidence == None
    
    def _init_variable_Fmin(self, data):
        self._F_min_start = data.index[0]
        _F_min_end = data.index[-1]
        self._F_min_period = (_F_min_end - self._F_min_start).total_seconds()
    
    def F_min(self, t):
        if isinstance(self._F_min, (int, float)):
            # Constant F_min
            return self._F_min
        
        elif self._F_min == 'sine':
            # Sine wave F_min
            t_offset = (t - self._F_min_start).total_seconds()
            return 0.88 + 0.05 * np.sin(2 * np.pi * t_offset / self._F_min_period)
        
        raise ValueError(f"Invalid F_min type: {type(self._F_min)}")

class EventType(Enum):
    RECALIBRATION = 0
    PARAMETER_CHANGE = 1
    FID_CHECK = 2

class Event():
    def __init__(self, event_type: EventType, time: pd.Timestamp, **kwargs):
        self.event_type = event_type
        self.time = time
        if event_type == EventType.PARAMETER_CHANGE:
            self.link_params = kwargs['link_params']
            self.overall_p_success = kwargs['overall_p_success']
            self.plotting_params = {'predicted_F_trans': kwargs['pfp'], 'r_avg': kwargs['r_avg']} # Only used for plotting purposes
        elif event_type == EventType.RECALIBRATION:
            self.end_time = kwargs['end_time']
            self.duration = (self.end_time - self.time).total_seconds()
            self.reference_stokes = kwargs['reference_stokes']
            self.final_fidelity = kwargs['final_fidelity']
        elif event_type == EventType.FID_CHECK:
            self.end_time = kwargs['end_time']
            self.duration = (self.end_time - self.time).total_seconds()
            self.measured_stokes = kwargs['measured_stokes']
            self.measured_fidelity = kwargs['measured_fidelity']
            self.triggering_threshold = kwargs['triggering_threshold']
            self.target_threshold = kwargs['target_threshold']
    
    def __repr__(self):
        if self.event_type == EventType.PARAMETER_CHANGE:
            return f"PARAMETER CHANGE at {self.time}: power={self.link_params[0]:.1f}, p_success={self.overall_p_success:.2f}, rate={SOURCE_RATE * self.overall_p_success:.1f}"
        elif self.event_type == EventType.RECALIBRATION:
            return f"RECALIBRATION    at {self.time}: duration={self.duration}, F_final={self.final_fidelity:.3f}"
        elif self.event_type == EventType.FID_CHECK:
            return f"FIDELITY CHECK   at {self.time}: F={self.measured_fidelity:.3f}, F_trigger={self.triggering_threshold:.3f}"

# Simulation state
class State():
    def __init__(self, start_time, data):
        self.current_time = start_time
        self.event_type = EventType.PARAMETER_CHANGE
        self.success_agg_calibration = 0 # The expected number of successful entanglement distributions since the last recalibration, based on link parameters; required for calbration scheduling
        self.success_agg_fid_check = 0 # The expected number of successful entanglement distributions since the last fidelity check, based on link parameters; required for fidelity check scheduling
        self.drift_tracker = DriftTracker(start_time, get_stokes_by_time(data, start_time))
        self.rate_projection = None
        self.events = []
        self.last_parameter_change = Event(EventType.PARAMETER_CHANGE, 
                                           start_time-pd.Timedelta(days=365), 
                                           link_params=None, # Will be set in first iteration
                                           overall_p_success=0, # Determined by link_params; will be set in first iteration
                                           pfp=0, r_avg=0 # Only used for plotting purposes
                                           )
        self.last_calibration = Event(EventType.RECALIBRATION,
                                      start_time,
                                      end_time=start_time,
                                      reference_stokes=get_stokes_by_time(data, start_time), # Last calibration before start_time
                                      final_fidelity=.99
                                      )
        self.last_fid_check = Event(EventType.FID_CHECK, 
                                    pd.Timestamp(0),
                                    end_time=pd.Timestamp(0),
                                    measured_stokes=get_stokes_by_time(data, start_time),
                                    measured_fidelity=.99,
                                    triggering_threshold=0,
                                    target_threshold=0
                                    )

    def new_event(self, event: Event):
        self.events.append(event)
        match event.event_type:
            case EventType.PARAMETER_CHANGE:
                self.last_parameter_change = event
            case EventType.RECALIBRATION:
                self.last_calibration = event
            case EventType.FID_CHECK:
                self.last_fid_check = event
    
    def last_parameter_change_time(self):
        return self.last_parameter_change.time

    def last_calibration_time(self):
        return self.last_calibration.end_time

    def last_fid_check_time(self):
        return self.last_fid_check.end_time
    
    def link_params(self):
        return self.last_parameter_change.link_params

    def overall_p_success(self):
        return self.last_parameter_change.overall_p_success

    def reference_stokes(self):
        return self.last_calibration.reference_stokes

    def last_measured_stokes(self):
        if self.last_calibration_time() > self.last_fid_check_time():
            return self.last_calibration.reference_stokes
        else:
            return self.last_fid_check.measured_stokes
    
    def last_measured_fidelity(self):
        if self.last_calibration_time() > self.last_fid_check_time():
            return self.last_calibration.final_fidelity
        else:
            return self.last_fid_check.measured_fidelity
    
    def last_measurement_time(self):
        return max(self.last_calibration_time(), self.last_fid_check_time())
    
    def _recent_param_changes(self):
        rpc = []
        for event in reversed(self.events):
            if event.event_type != EventType.PARAMETER_CHANGE:
                break
            assert event.time >= self.last_measurement_time()
            rpc.append(event)
        return rpc
    
    def estimate_rate_gradient(self):
        param_changes = self._recent_param_changes()
        if len(param_changes) < N_FIT_POINTS:
            # Not enough points to compute gradient
            return None

        times = [e.time.timestamp() for e in param_changes]
        rates = [SOURCE_RATE*e.overall_p_success for e in param_changes]
        coeffs = np.polyfit(times, rates, 1)
        grad = coeffs[0]
        return grad

    def expected_gain(self, s, Teff, r_exp):
        # Compute the expected rate gain of doing a fidelity check
        # s seconds since last check/compensation, Teff seconds til the next one, r_exp expected rate after check
        if not self.rate_projection:
            return 0
        ts, rs, integration = self.rate_projection
        c = rs[0] - r_exp # expected decrease in rate (comparing the rate after the last check to the rate I would get right after this check)
        i = np.searchsorted(ts, s)
        j = np.searchsorted(ts, Teff)
        k = np.searchsorted(ts, s+Teff)
        term1 = integration[j] - c*Teff # int_0^Teff [r(t) - c] dt
        term2 = integration[k] - integration[i] # int_s^{s+Teff} r(t) dt
        
        return term1 - term2 



class DriftTracker:
    def __init__(self, first_time, first_stokes):
        # Track measured stokes and their timestamps. 
        # Each inner list corresponds to a compensation cycle, where the first elements are the reference stokes vector and the time of its measurement.
        self.measurement_times = [[first_time]]
        self.measured_stokes = [[first_stokes]]
    
    def update_fid_check(self, time, current_stokes):
        self.measurement_times[-1].append(time)
        self.measured_stokes[-1].append(current_stokes)
        dt = (time - self.measurement_times[-1][-2]).total_seconds()
        assert dt > 0, f"dt: {dt} is non-positive at time {time}"
    
    def update_compensation(self, time, reference_stokes):
        self.measurement_times.append([time])
        self.measured_stokes.append([reference_stokes])
    
    def last_drift_angle(self):
        if self.is_empty():
            return 0
        
        stokes = self.measured_stokes[-1] if len(self.measured_stokes[-1]) > 1 else self.measured_stokes[-2]
        theta = angle_between_stokes(stokes[-1], stokes[-2])
        return theta

    def last_drift_interval(self):
        if self.is_empty():
            return 0
        
        times = self.measurement_times[-1] if len(self.measurement_times[-1]) > 1 else self.measurement_times[-2]
        return (times[-1] - times[-2]).total_seconds()

    def last_drift_rate(self):
        if self.is_empty():
            return 0
        
        dt = self.last_drift_interval()
        theta = self.last_drift_angle()
        return theta / dt

    # Attempt to find the drift angle/interval over the last ~window seconds
    def _recent_drift(self, window):
        if self.is_empty():
            return 0, 0
        
        recent_times = self.measurement_times[-1]
        if len(self.measurement_times) == 1 or (len(recent_times) > 1 and (recent_times[-1] - recent_times[0]).total_seconds() > 1.0):
            stokes = self.measured_stokes[-1]
            times = self.measurement_times[-1]
        else:
            stokes = self.measured_stokes[-2]
            times = self.measurement_times[-2]
        
        t_prev = 0
        i = 1
        while t_prev < window and i < len(times):
            i += 1
            t_prev = (times[-1] - times[-i]).total_seconds()
            
        recent = stokes[-i]
        latest = stokes[-1]

        theta = angle_between_stokes(latest, recent)
        return theta, t_prev
    
    def recent_drift_angle(self, window=10):
        return self._recent_drift(window)[0]

    def recent_drift_interval(self, window=10):
        return self._recent_drift(window)[1]
    
    def is_empty(self):
        return len(self.measurement_times) == 1 and len(self.measurement_times[-1]) <= 1



# Returns the stokes parameters at the given timestamp t
def get_stokes_by_time(data, t):
    idx = data.index.get_indexer([t], method='nearest')[0]
    return data.iloc[idx][['S1', 'S2', 'S3']].values

# Get transmission fidelity at current time relative to reference Stokes vector.
def get_transmission_fidelity(data, current_time, last_calibration_event):
    current_stokes = get_stokes_by_time(data, current_time)
    reference_stokes = last_calibration_event.reference_stokes
    theta = angle_between_stokes(current_stokes, reference_stokes)
    dtheta = fid2angle(last_calibration_event.final_fidelity) # Account for imperfect recalibration
    return angle2fid(theta + dtheta)

def get_transmission_success_probability():
    return p_success_transmission()

# Returns the next time at which link parameters will be changed.
def get_next_parameter_change_time(state, settings: SimulationSettings):
    # Update paramters now if there has been a fidelity check or recalibration
    if state.last_parameter_change_time() < state.last_measurement_time():
        return state.current_time
    
    # Don't update parameters if they are set to be constant
    if settings.source_fidelity != None:
        return state.current_time + pd.Timedelta(days=365)
    
    # Update paramters at regular intervals
    return max(state.last_parameter_change_time() + pd.Timedelta(seconds=PARAM_UPDATE_INTERVAL), state.current_time)

def get_next_fid_check_time(state: State, settings: SimulationSettings):
    """
    Returns the next time at which a fidelity check should be performed.
    """
    if settings.fid_check_freq:
        last = state.last_measurement_time()
        next = last + pd.Timedelta(seconds=settings.fid_check_freq)
        assert next >= state.current_time
        return next
    
    # Check the fidelity if the pump power is at the minimum, we likely need to compensate for drift
    current_power = state.link_params()[0]
    if current_power == MIN_PUMP_POWER_MW and (state.current_time - state.last_measurement_time()).total_seconds() >= MIN_FID_CHECK_INTERVAL:
        return state.current_time
    
    # Adaptive fidelity check timing:
    #   Perform a fidelity check when the instantaneous rate dips below the average rate
    s = (state.current_time - state.last_calibration_time()).total_seconds()
    c = predict_compensation_time(state.drift_tracker.last_drift_rate(), state.last_measured_fidelity(), compute_target_threshold(settings), timeout_s=compute_timeout(settings))
    r_avg = state.success_agg_calibration / (s + c) # Average rate including a future calibration
    r = state.overall_p_success() * SOURCE_RATE # Instantaneous rate
    if r <= r_avg and (state.current_time - state.last_measurement_time()).total_seconds() >= MIN_FID_CHECK_INTERVAL:
        return state.current_time
    
    # Else, check if gain from doing a check exceeds loss
    # To compute this, need to estimate time until next check
    d = state.estimate_rate_gradient()
    if not d or d >= 0:
        return state.current_time + pd.Timedelta(days=365)
    
    a = state.success_agg_calibration
    T_eff = (-np.sqrt((2*c*d + 2*d*s)**2 - 4*d*(-2*a + 2*c*r + 2*r*s)) - 2*c*d - 2*d*s)/(2*d) # Estimated time until next fid check, assuming constant rate gradient into future
    T_eff = np.clip(T_eff, 0, s)

    F_trans_expected = expected_transmission_fidelity_after_fid_check(state, settings)
    F_source_expected = settings.F_min(state.current_time)/F_trans_expected
    r_expected = fid_to_rate(F_source_expected) * get_transmission_success_probability()
   
    gain = state.expected_gain((state.current_time - state.last_measurement_time()).total_seconds(), T_eff, r_expected)
    loss = FID_CHECK_DURATION * r

    # if T_eff * (r_expected - r) >= FID_CHECK_DURATION * r:
    if gain >= loss:
        # Expected gain from fidelity check outweighs the cost, do one now
        return state.current_time

    return state.current_time + pd.Timedelta(days=365)




def sample_next_event(state: State, settings: SimulationSettings):
    # Recalibrate now if we just did a fidelity check that was below the triggering threshold
    if state.last_fid_check_time() == state.current_time and state.last_fid_check.measured_fidelity <= state.last_fid_check.triggering_threshold:
        return EventType.RECALIBRATION, state.current_time
    
    # Calculate next event times
    next_param_change_time = get_next_parameter_change_time(state, settings)
    next_fid_check_time = get_next_fid_check_time(state, settings)
    
    # Find the next event
    next_event_time = min(next_param_change_time, next_fid_check_time)
    assert next_event_time >= state.current_time, f"Next event time {next_event_time} is before current time {state.current_time}."
    if next_event_time == next_fid_check_time:
        return EventType.FID_CHECK, next_event_time
    elif next_event_time == next_param_change_time:
        return EventType.PARAMETER_CHANGE, next_event_time
    else:
        raise RuntimeError(f"Error sampling next event: next_event_time = {next_event_time}")

def compute_triggering_threshold(state: State, settings: SimulationSettings):
    """
    Compute the triggering threshold for a fidelity check event.
    """
    if settings.F_trigger != None:
        return settings.F_trigger
    
    # Trigger compensation if the pump power is at the minimum
    current_power = state.link_params()[0]
    if current_power == MIN_PUMP_POWER_MW and (state.current_time - state.last_fid_check_time()).total_seconds() >= MIN_FID_CHECK_INTERVAL:
        return 1.0

    F_target = compute_target_threshold(settings)
    c = predict_compensation_time(state.drift_tracker.last_drift_rate(), state.last_measured_fidelity(), F_target, timeout_s=compute_timeout(settings))
    s = (state.current_time - state.last_calibration_time()).total_seconds()
    r_avg = state.success_agg_calibration / (s+c)


    # Check if the last two intervals have been bad
    r = state.overall_p_success() * SOURCE_RATE # Instantaneous rate
    if r <= r_avg:
        interval_is_bad = (state.current_time - state.last_measurement_time()).total_seconds() == MIN_FID_CHECK_INTERVAL
        last_interval_is_bad = (state.drift_tracker.last_drift_interval() == MIN_FID_CHECK_INTERVAL+FID_CHECK_DURATION) and (state.last_fid_check_time() > state.last_calibration_time())
        if interval_is_bad and last_interval_is_bad:
            # Last two intervals crashed out, perform a calibration now
            return 1.0


    
    # Need to find the polarization fidelity which corresponds to the overall rate r_avg: overall rate -> source rate -> source fidelity -> polarization fidelity
    r_source = r_avg / get_transmission_success_probability()
    power = power_from_rate(r_source)
    F_source = fidelity_sd([power])

    # Adjust F_trigger by (an estimate of) the amount of drift that will be predicted at time 0 after the check.
    # This corresponds to the rate that will be achieved once the check is complete (if compensation is not triggered).
    theta_prev = state.drift_tracker.recent_drift_angle()
    t_prev = state.drift_tracker.recent_drift_interval()
    theta_pred = drift_predictor(theta_prev, t_prev, 0, percentile=settings.confidence*100)
    F_trigger = angle2fid(fid2angle(settings.F_min(state.current_time) / F_source) - theta_pred)

    return F_trigger

def compute_target_threshold(settings: SimulationSettings):
    return settings.F_target

def compute_timeout(settings: SimulationSettings):
    return settings.timeout

def project_rates(state: State, settings: SimulationSettings):
    # Project drift angles into the future
    ts = []
    rs = []
    for t_next in np.arange(0, 10, PARAM_UPDATE_INTERVAL):
        pfp = polarization_fidelity_prediction(state, settings.confidence, dt=t_next)
        source_fidelity = settings.F_min(state.current_time)/pfp
        r = fid_to_rate(source_fidelity) * get_transmission_success_probability()
        rs.append(r)
        ts.append(t_next)
        if power_from_fid(source_fidelity) == MIN_PUMP_POWER_MW:
            # The protocol will do a fidelity check at this point
            break
    
    integration = np.cumsum([0]+rs) * PARAM_UPDATE_INTERVAL # cumulative throughput at each interval
    return np.array(ts), np.array(rs), integration


def compute_stats(events, data, settings: SimulationSettings):

    def update_stats(stats, t, data, last_calibration_event, source_detector_fidelity, p, pfp, r_avg, min_fidelity):
        """
        Update the simulation statistics at time t.
        """
        current_transmission_fidelity = get_transmission_fidelity(data, t, last_calibration_event)
        overall_fidelity = source_detector_fidelity * current_transmission_fidelity # TODO: update fidelity calculation
        stats['time_history'].append(t)
        stats['F_source_history'].append(source_detector_fidelity)
        stats['F_trans_history'].append(current_transmission_fidelity)
        stats['F_trans_predicted_history'].append(pfp)
        stats['fidelity_history'].append(overall_fidelity)
        stats['expected_rate_history'].append(SOURCE_RATE * p)
        stats['r_avg_history'].append(r_avg)
        if min_fidelity is not None and overall_fidelity < min_fidelity:
            stats['rate_history'].append(0)
        else:
            stats['rate_history'].append(SOURCE_RATE * p)

    print("Computing statistics...")
    stats = {
        'time_history': [],
        'fidelity_history': [],
        'F_source_history': [],
        'F_trans_history': [],
        'F_trans_predicted_history': [],
        'expected_rate_history': [],
        'rate_history': [],
        'r_avg_history': [],
        'compensation_event_times': [],
        'compensation_event_durations': [],
        'time_compensating': 0,
        'average_rate': 0,
        'time_below_threshold': 0,
    }

    sim_time_seconds = (data.index[-1] - data.index[0]).total_seconds()
    start_time = data.index[0]
    last_calibration_event = State(start_time, data).last_calibration # Initialize same as simulation
    event_idx = 0
    next_event = events[event_idx]
    assert next_event.event_type == EventType.PARAMETER_CHANGE and next_event.time == start_time
    compensating_until = start_time
    last_progress_time = start_time

    for t in data.index:
        while next_event.time <= t and event_idx < len(events):
            # Process event
            assert t >= compensating_until
            e = next_event
            match e.event_type:
                case EventType.PARAMETER_CHANGE:
                    p = e.overall_p_success
                    pfp = e.plotting_params['predicted_F_trans']
                    r_avg = e.plotting_params['r_avg']
                    source_detector_fidelity = fidelity_sd(e.link_params)
                    update_stats(stats, e.time, data, last_calibration_event, source_detector_fidelity, p, pfp, r_avg, settings.F_min(e.time))
                case EventType.RECALIBRATION | EventType.FID_CHECK:
                    duration_s = e.duration
                    stats['compensation_event_times'].append(e.time)
                    stats['compensation_event_durations'].append(duration_s)
                    compensating_until = e.end_time
                    if e.event_type == EventType.RECALIBRATION:
                        last_calibration_event = e
                    stats['time_history'] += [e.time, e.time]
                    stats['fidelity_history'] += [stats['fidelity_history'][-1], 0]
                    stats['F_source_history'] += [stats['F_source_history'][-1], 0]
                    stats['F_trans_history'] += [stats['F_trans_history'][-1], 0]
                    stats['F_trans_predicted_history'] += [stats['F_trans_predicted_history'][-1], 0]
                    stats['expected_rate_history'] += [SOURCE_RATE * p, SOURCE_RATE * p]
                    stats['rate_history'] += [stats['rate_history'][-1], 0]
                    stats['r_avg_history'] += [stats['r_avg_history'][-1], 0]

            event_idx += 1
            if event_idx < len(events):
                next_event = events[event_idx]
        
        if t == events[event_idx-1].time:
            # Don't duplicate data points in stats
            continue
        
        if t < compensating_until:
            # We are still compensating
            continue
        
        # Compute fidelity/rate for current time t
        update_stats(stats, t, data, last_calibration_event, source_detector_fidelity, p, pfp, r_avg, settings.F_min(t))

        if (t - last_progress_time).total_seconds() >= sim_time_seconds/1000:
            elapsed = (t - data.index[0]).total_seconds()
            total = sim_time_seconds
            elapsed = min(elapsed, total)
            progressBar(elapsed, total)
            last_progress_time = t
    
    
    # Compute final stats
    total_weighted_rate = 0.0
    time_below_threshold = 0.0
    total_time = 0.0
    
    for i in range(len(stats['time_history']) - 1):
        dt = (stats['time_history'][i+1] - stats['time_history'][i]).total_seconds()
        total_weighted_rate += stats['rate_history'][i] * dt
        total_time += dt
        if stats['fidelity_history'][i] < settings.F_min(stats['time_history'][i]):
            time_below_threshold += dt
    
    # Compute list of durations between compensation events
    stats['compensation_intervals'] = []
    comp_times = stats['compensation_event_times']
    comp_durs = stats['compensation_event_durations']
    for i in range(len(comp_times)-1):
        t0 = comp_times[i]
        t1 = comp_times[i+1]
        dur = comp_durs[i]
        interval = (t1 - t0).total_seconds() - dur
        if interval > 0:
            stats['compensation_intervals'].append(interval)

        
    stats['time_compensating'] = sum(stats['compensation_event_durations'])
    stats['time_below_threshold'] = time_below_threshold - stats['time_compensating']
    stats['average_rate'] = total_weighted_rate / total_time

    recalibration_durs = np.array([x for x in stats['compensation_event_durations'] if x > FID_CHECK_DURATION])
    n_fid_checks = sum([1 for d in stats['compensation_event_durations'] if d <= FID_CHECK_DURATION])
    n_compensations = len(recalibration_durs)
    median_compensation_time = 1000*np.median(recalibration_durs)
    mean_compensation_time = 1000*np.mean(recalibration_durs)
    stats['total_time'] = total_time
    stats['n_fid_checks'] = n_fid_checks
    stats['n_compensations'] = n_compensations
    stats['median_compensation_time'] = median_compensation_time
    stats['mean_compensation_time'] = mean_compensation_time
    stats['F_mins'] = [settings.F_min(t) for t in stats['time_history']]

    progressBar(1, 1, done=True)
    
    return stats



def run_simulation(data, settings : SimulationSettings):
    # Run a full simulation of quantum link operation.
    
    # Get time range from data
    start_time = data.index[0]
    end_time = data.index[-1]
    sim_time_seconds = (end_time - start_time).total_seconds()
    
    # Initialize simulation state
    state = State(start_time, data)

    # Initialize progress tracking
    last_progress_time = start_time
    progress_interval = pd.Timedelta(seconds=max(sim_time_seconds/1000,10))

    succ_agg_xx = []
    succ_agg_yy = []
    

    # Simulation loop - event-driven approach
    while state.current_time <= end_time:
        match (state.event_type):
            case EventType.PARAMETER_CHANGE:
                # Update aggregated success probability (expected number of successful entanglement distributions since the last recalibration event)
                dt_calibration = (state.current_time - max(state.last_parameter_change_time(), state.last_calibration_time())).total_seconds()
                dt_fid_check = (state.current_time - max(state.last_parameter_change_time(), state.last_measurement_time())).total_seconds()
                if state.last_fid_check_time() == state.current_time:
                    dt_calibration -= FID_CHECK_DURATION # time spent checking fidelity should not count toward aggregate successes
                    assert dt_calibration >= 0, f"dt_calibration: {dt_calibration} is negative at time {state.current_time}"
                state.success_agg_calibration += SOURCE_RATE * state.overall_p_success() * dt_calibration
                state.success_agg_fid_check += SOURCE_RATE * state.overall_p_success() * dt_fid_check
                
                ### This block is only used for plotting purposes
                drift = state.drift_tracker.last_drift_rate()
                c = predict_compensation_time(drift, state.last_measured_fidelity(), compute_target_threshold(settings), timeout_s=compute_timeout(settings))
                s = (state.current_time - state.last_calibration_time()).total_seconds()
                r_avg = state.success_agg_calibration / (s+c) if state.success_agg_calibration > 0 else 0
                succ_agg_xx.append(state.current_time)
                succ_agg_yy.append(r_avg)
                ### ### ### ### ### ###

                # Reoptimize link parameters
                if settings.source_fidelity == None:
                    pfp = polarization_fidelity_prediction(state, settings.confidence)
                    source_fidelity = settings.F_min(state.current_time)/pfp
                else:
                    source_fidelity = settings.source_fidelity
                
                # Optimize
                laser_power = power_from_fid(source_fidelity)
                link_params = np.array([laser_power])

                # Record event
                event = Event(EventType.PARAMETER_CHANGE, state.current_time, 
                              link_params=link_params, 
                              overall_p_success=p_success_sd(link_params) * get_transmission_success_probability(),
                              pfp=pfp if settings.source_fidelity == None else 0,
                              r_avg=r_avg
                              )
                state.new_event(event)

            case EventType.FID_CHECK:
                # Fidelity check event
                duration = FID_CHECK_DURATION
                fid_check_end_time = state.current_time + pd.Timedelta(seconds=duration)
                latest_stokes = get_stokes_by_time(data, fid_check_end_time)
                state.drift_tracker.update_fid_check(fid_check_end_time, latest_stokes)
                state.success_agg_fid_check = 0
                F = get_transmission_fidelity(data, state.current_time, state.last_calibration)
                event = Event(EventType.FID_CHECK, state.current_time, 
                              end_time=fid_check_end_time,
                              measured_stokes=latest_stokes,
                              measured_fidelity=F,
                              triggering_threshold=compute_triggering_threshold(state, settings),
                              target_threshold=compute_target_threshold(settings)
                              )
                state.new_event(event)
                
                # Jump to end of fidelity check period
                state.current_time = state.current_time + pd.Timedelta(seconds=duration)

                if settings.adaptive:
                    state.rate_projection = project_rates(state, settings)

            case EventType.RECALIBRATION:
                # Recalibration event
                F_0 = state.last_measured_fidelity()
                F_target = state.last_fid_check.target_threshold
                timeout = compute_timeout(settings)
                duration, F_final = simulate_apc(data, F_0=F_0, F_target=F_target, t_start=state.current_time, timeout_s=timeout)
                recalibration_end_time = state.current_time + pd.Timedelta(seconds=duration)
                reference_stokes = get_stokes_by_time(data, recalibration_end_time)
                state.drift_tracker.update_compensation(recalibration_end_time, reference_stokes)
                state.success_agg_calibration = 0
                state.success_agg_fid_check = 0
                
                # Recalibrate by resetting reference polarization vector
                event = Event(EventType.RECALIBRATION, state.current_time, 
                              end_time=recalibration_end_time,
                              reference_stokes=reference_stokes,
                              final_fidelity=F_final
                              )
                state.new_event(event)
                # Jump to end of recalibration period
                state.current_time = recalibration_end_time

                if settings.adaptive:
                    state.rate_projection = project_rates(state, settings)
        # Progress indicator
        if state.current_time - last_progress_time >= progress_interval:
            elapsed = (state.current_time - start_time).total_seconds()
            total = (end_time - start_time).total_seconds()
            elapsed = min(elapsed, total)
            progressBar(elapsed, total)
            last_progress_time = state.current_time

        # Sample next event
        state.event_type, state.current_time = sample_next_event(state, settings)
    
    progressBar(1, 1, done=True)
    stats = compute_stats(state.events, data, settings)
    
    stats['succ_agg_xx'] = succ_agg_xx
    stats['succ_agg_yy'] = succ_agg_yy
    
    return stats


def save_results(results_file, settings, stats):
    lock_path = f"{results_file}.lock"

    stats_cols = [
        "average_rate",
        "n_fid_checks",
        "n_compensations",
        "median_compensation_time",
        "mean_compensation_time",
        "time_compensating",
        "time_below_threshold",
    ]
    settings_cols = settings.labels
    all_cols = ['time'] + settings_cols + stats_cols

    with FileLock(lock_path):
        if os.path.exists(results_file):
            results = pd.read_csv(results_file)
            if not isinstance(results, pd.DataFrame):
                raise TypeError(f"Expected a pandas DataFrame in {results_file}, got {type(results)}.")
            # Ensure required columns exist (handles schema evolution).
            for col in all_cols:
                if col not in results.columns:
                    results[col] = pd.NA
        else:
            results = pd.DataFrame(columns=all_cols)

        row = {}
        row['time'] = pd.Timestamp.now()
        row.update(settings.settings_dict())
        row.update({col: stats.get(col, pd.NA) for col in stats_cols})

        results = pd.concat([results, pd.DataFrame([row], columns=all_cols)], ignore_index=True)
        results.to_csv(results_file, index=False)

def main(settings):
    import warnings
    warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")
    
    # Load data
    window_size_s = 5
    print("Loading data...")
    data = load_data(window_size_s, resample_ms=settings['resample_ms'], sim_time_seconds=settings['sim_time_seconds'], start_time=settings['start_time'])
    print(f"Loaded {len(data)} data points from {data.index[0]} to {data.index[-1]}")
    
    # Extract args
    results_file = os.path.join('results', settings['save_file_name']) if settings['save_file_name'] else None
    freqs = settings['fid_check_freq']
    source_fids = settings['source_fidelity']
    F_triggers = settings['F_trigger']
    F_targets = settings['F_target']
    timeouts = settings['timeout']
    confidences = settings['confidence']
    F_mins = settings['min_fidelity']

    if not results_file and not settings['show']:
        print("Warning: no results file and no show flag, so nothing will be saved, plotted, or printed.")

    # Run simulation
    for F_min in F_mins:
        for freq in freqs:
            for F_source in source_fids:
                for F_trigger in F_triggers:
                    for F_target in F_targets:
                        for timeout in timeouts:
                            for confidence in confidences:
                                run_settings = SimulationSettings(
                                    sim_time_seconds=settings['sim_time_seconds'],
                                    resample_ms=settings['resample_ms'],
                                    label=settings['label'],
                                    adaptive=settings['adaptive'],
                                    F_min=F_min,
                                    source_fidelity=F_source,
                                    fid_check_freq=freq,
                                    F_trigger=F_trigger,
                                    F_target=F_target,
                                    timeout=timeout,
                                    confidence=confidence,
                                    data=data
                                )
                                print(f"F_min: {F_min}, freq: {freq}, F_source: {F_source}, F_trigger: {F_trigger}, F_target: {F_target}, timeout: {timeout}, confidence: {confidence}")
                                print("Running simulation...")
                                stats = run_simulation(data, settings=run_settings)
                                if results_file:
                                    save_results(results_file, run_settings, stats)
                                if settings['show']:
                                    start = stats['time_history'][0] + pd.Timedelta('245.2s')
                                    end = stats['time_history'][0] + pd.Timedelta('263.8s')
                                    F_min_varying=not isinstance(F_min, (int, float))
                                    # plot_simulation(stats)
                                    # plot_calibration_durs(stats)
                                    plot_simulation_detailed(stats, start=start, end=end, F_min_varying=F_min_varying)

    

    # Plot results
    # print("Plotting results...")
    # min_fidelity = constraints.get('min_fidelity') if constraints and 'min_fidelity' in constraints else None
    # plot_simulation(stats, min_fidelity)
    # plot_rates(freqs, results)
    # plot_calibration_durs(stats)


if __name__ == '__main__':

    runs = {
        'normal': None,
        'day': '2023-09-30 10:00:00',
        'night': '2023-09-30 22:00:00',
    }

    run_label = 'normal'
    settings = {
        'sim_time_seconds': 270,
        'resample_ms': 100,
        'start_time': '2023-09-30 8:00:00', #runs[run_label],
        'save_file_name': None, #'results.csv',
        'show': True,

        'min_fidelity': [0.85],

        'adaptive': True, # False for static protocols
        'source_fidelity': [None], # Constant fidelity achieved by link parameters, None for adaptive parameter optimization
        'fid_check_freq': [None], # Recalibration frequency in seconds, None for adaptive recalibration
        
        'F_trigger': [None], # Triggering fidelity threshold, None for adaptive threshold
        'F_target': [1.0], # Target fidelity threshold
        'timeout': [1], # Recalibration timeout in seconds
        'confidence': [0.90], # Confidence level for drift predictions, only applicable for adaptive protocol
    }

    settings['label'] = run_label
    main(settings)