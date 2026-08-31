import enum
import os
import numpy as np
import pandas as pd
from glob import glob
from sunlight import solarInsolation
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime

class DependentVar(enum.Enum):
    TIMING = 'timing'
    POLARIZATION = 'polarization'

class WeatherAPI(enum.Enum):
    API1 = 'api1'
    API2 = 'api2'

class WeatherLocation(enum.Enum):
    LTS = 'LTS'
    NIST = 'NIST'

settings = {
    'experiment': "9_29-10_2_2023(Polarization) (Cs Reference, LTS GM, Star Top., ZEN_LEN)",
    'basedir': os.path.join("..", "data"),
    'polarization_dirname': '1539 Polarization Data',
    'weather_api': WeatherAPI.API1,
    'weather_location': WeatherLocation.LTS,
    'ws': 10,
    'ws2': 500,
    'weather_params': ['sunlight', 'avg-wind-speed squared'],
    'latitude': 39.005212,
    'longitude': -76.943433,
    'dependentVar': DependentVar.POLARIZATION
}


def EDT2UTC(datetime64, reverse=False):
    # convert EDT timestamp to UTC (or reverse)
    return datetime64 + np.timedelta64(4, 'h') if not reverse else datetime64 - np.timedelta64(4, 'h')

def UTC2EDT(datetime64):
    return EDT2UTC(datetime64, True)

# readWeatherFolder -> read_weather_data
def read_weather_data(dir : str, api : WeatherAPI) -> pd.DataFrame:
    '''
    Read in the weather data. For example, from a single experiment folder, we
    concatenate all the .txt files into a single dataframe. The text files will
    be found in WeatherData/api/location. Remember that the weather data is
    split across multiple files, each containing the same data but split up.
    '''
    dfs = []
    for filename in sorted(os.listdir(dir)):
        if filename.endswith('.txt'):
            df = pd.read_csv(os.path.join(dir, filename), sep='\t', engine='python')
            dfs.append(df)

    res = pd.concat(dfs)
    res.sort_values(by=['timestamp'], inplace=True)

    if "20230509_5Day_CP_GM_Cs_BC_BC (Cs Reference, LTS GM)" in dir:
        res['timestamp'] -= 60*60

    if api == WeatherAPI.API1:
        res = res.drop(
            labels=['cloud-base', 'cloud-ceiling'],
            axis=1
        )
    if api == WeatherAPI.API2:
        res = res.rename(columns={
            'w.s.': 'wind-speed',
            'dir.': 'wind-dir',
            '%T(K)': 'temperature',
            'press.': 'pressure',
            'hum.': 'humidity',
            'vis.': 'visibility'
        })
        res['temperature'] -= 273.15 # Convert Kelvin to Celsius

    return res


def add_sunlight_data(weather_data : pd.DataFrame, longitude : float,
                      latitude : float, api : WeatherAPI) -> pd.DataFrame:
    weather_data['sunlight'] = solarInsolation(
        weather_data['timestamp'],
        latitude,
        longitude
    )
    weather_data['daytime'] = (weather_data['sunlight'] > 0).astype(int)

    if api == WeatherAPI.API2:
        weather_data['isClear'] = (weather_data['cover'] == 'Clear').astype(int)
        weather_data['isCloudy'] = (weather_data['cover'] == 'Clouds').astype(int)
        weather_data['isFoggy'] = (weather_data['cover'] == 'Fog').astype(int)
        weather_data['isMisty'] = (weather_data['cover'] == 'Mist').astype(int)
        weather_data['isDrizzling'] = (weather_data['cover'] == 'Drizzle').astype(int)
        weather_data['isRaining'] = (weather_data['cover'] == 'Rain').astype(int)
        weather_data['isStorming'] = (weather_data['cover'] == 'Thunderstorm').astype(int)

    return weather_data

def get_average_weather_data(experiment_dir : str, param : str, timestamps,
                             smoothing : float=30):
    '''
    Compute the average value of `param` across all apis/locations.
    * `param`: weather parameter to average, e.g. 'temperature', 'wind-speed',
      'cloud-cover'
    * `timestamps`: array of timestamps at which to compute the average
    * `smoothing`: averaging window size in minutes
    '''

    path = os.path.join(
        experiment_dir,
        'WeatherData'
    )

    dfs = []

    for api in WeatherAPI:
        for loc in WeatherLocation:
            dir = os.path.join(path, api.value, loc.value)

            if os.path.exists(dir):
                dfs.append(read_weather_data(dir, api))

    timestamps = np.array(timestamps)
    sample_rate = round(timestamps[1] - timestamps[0])

    ys = []
    window_size = round(60 * smoothing / sample_rate)

    for df in dfs:
        ynew = pd.DataFrame(np.interp(
            timestamps,
            df['timestamp'],
            df[param],
            left=np.nan,
            right=np.nan)
        ).rolling(window_size, center=True).mean()
        ys.append(ynew)
    
    y_avg = np.nanmean(np.hstack(ys), axis=1)
    return y_avg

def reduce_window_by_factor(data : pd.DataFrame, ws : int) -> np.ndarray:
    if ws <= 1:
        return np.array(data)
    else:
        return np.array(data.rolling(ws, center=True).mean()[ws::ws])
    
def dxdt(data : pd.DataFrame, param : str, ws : int, ws2 : int=1000, absv : bool=True) -> np.ndarray:
    dpdt = np.gradient(reduce_window_by_factor(data[param], ws))
    if absv:
        dpdt = abs(dpdt)
    rolling_dpdt = np.array(pd.DataFrame(dpdt).rolling(ws2, center=True).mean())
    return rolling_dpdt

def construct_weather_df() -> pd.DataFrame:
    experiment_dir = os.path.join(
        settings['basedir'], 
        settings['experiment']
    )

    # Find the exact path to the weather data, given the api and location.
    weather_dir = os.path.join(
        settings['basedir'], 
        settings['experiment'],
        'WeatherData', 
        settings['weather_api'].value,
        settings['weather_location'].value
    )
    # Read in the weather data
    weather_data = read_weather_data(weather_dir, settings['weather_api'])
    weather_data = add_sunlight_data(
        weather_data, settings['longitude'],
        settings['latitude'],
        settings['weather_api']
    )
    weather_data['wind-speed squared'] = weather_data['wind-speed']**2

    if settings['weather_api'] == WeatherAPI.API1:
        pass
    if settings['weather_api'] == WeatherAPI.API2:
        weather_data['sunlight-adj'] = pd.DataFrame(
            weather_data['sunlight'] * (2*weather_data['isClear'] + weather_data['isCloudy'])
        ).rolling(120, center=True).mean()

    weather_data['avg-temperature'] = get_average_weather_data(
        experiment_dir,
        'temperature', 
        weather_data['timestamp']
    )
    weather_data['avg-temperature-change-rate'] = dxdt(
        weather_data, 
        'avg-temperature', 
        ws=1, 
        ws2=30, 
        absv=False
    )
    weather_data['avg-wind-speed'] = get_average_weather_data(
        experiment_dir,
        'wind-speed',
        weather_data['timestamp']
    )
    weather_data['avg-wind-speed squared'] = weather_data['avg-wind-speed']**2

    weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'], unit='s')
    weather_data.set_index('timestamp', inplace=True)
    weather_data = weather_data[~weather_data.index.duplicated(keep='last')]

    return weather_data

def getpkl(filepath : str):
    if os.path.exists(filepath):
        return pd.read_pickle(filepath)
    else:
        return None
    
def get_polarization_data(dir):
    print("-------------- Reading polarization data... --------------")

    pickle_path = os.path.join(dir, "PolarizationData.pkl")
    cached = getpkl(pickle_path)
    if isinstance(cached, pd.core.frame.DataFrame):
        return cached

    dfs = []
    for i, filepath in enumerate(sorted(glob(os.path.join(dir, '*.txt')))):
        if i == 0:
            with open(filepath, "r") as f:
                start_date = next(f)[len("Start Date: "):].strip()
                start_time = next(f)[len("Start Time: "):].strip()
            print(f"{start_date} {start_time}")
            start_datetime = datetime.strptime(f"{start_date} {start_time}", "%m/%d/%Y %H:%M:%S").timestamp()
            print(f"Start datetime: {start_datetime}")
        
        cols = [
            'timestamp',
            'S0(mW)',
            'S1',
            'S2',
            'S3',
            'DOP(%)',
            'Average DOP(%)'
        ]
        df = pd.read_csv(
            filepath,
            sep=' |\t',
            skiprows=15,
            header=None,
            engine='python'
        )
        
        df.columns = cols
        df['timestamp'] = df['timestamp'].astype(float) / 1000 + start_datetime
        dfs.append(df)

    all_dfs = pd.concat(dfs)
    all_dfs.sort_values(by=['timestamp'], inplace=True)
    all_dfs.attrs['filename'] = os.path.basename(dir)

    all_dfs['timestamp'] = UTC2EDT(pd.to_datetime(all_dfs['timestamp'], unit='s'))
    all_dfs.set_index('timestamp', inplace=True)
    all_dfs
    all_dfs.to_pickle(pickle_path)

    print("-------------- Finished reading polarization data. --------------")

    return all_dfs

def get_timing_data(dir):
    pickle_path = os.path.join(dir, "TimingData.pkl")
    cached = getpkl(pickle_path)
    if isinstance(cached, pd.core.frame.DataFrame):
        return cached

    dfs = []
    for filename in sorted(os.listdir(dir)):
        if filename.endswith('.dat'):
            filepath = os.path.join(dir, filename)
            cols = [
                'Record Counter',
                'timestamp',
                'Switch Delay',
                'Round Trip Delay',
                'Raw Delay',
                'Mean Delay'
            ]
            df = pd.read_csv(filepath, sep=' ', names=cols, engine='python')
            dfs.append(df)

    all_dfs = pd.concat(dfs)
    all_dfs.sort_values(by=['timestamp'], inplace=True)
    all_dfs.attrs['filename'] = os.path.basename(dir)
    all_dfs.to_pickle(pickle_path)

    return all_dfs

def toDatetime64(timestamps):
    return UTC2EDT((10**6*timestamps).astype('datetime64[us]'))

def plot_data_param(data, param):
    plt.figure(figsize=(12, 5))
    plt.plot(data.index, data[param])
    plt.xlabel('Time')
    plt.ylabel(param)
    plt.title(f'{param} vs Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_with_weather(data, param, weather_data, weather_params, ws=1,
                    ws2=1000, plotDerivatives=True, absvDerivative=True,
                    fit_func=None, y_label='default', title=None):
    
    print("-------------- Plotting with weather...")
    print(type(weather_data))

    def clipWeatherDomain(data, start, end):
        print("DEBUG:", type(data))        # check type
        print("DEBUG:", getattr(data, "columns", "no columns"))  # check columns if it's a DataFrame
        return data[(start <= data['timestamp']) & (data['timestamp'] <= end)]

    start_time = data['timestamp'].iloc[0]
    end_time = data['timestamp'].iloc[-1]
    weather_data = clipWeatherDomain(weather_data, start_time, end_time)

    dataxx = toDatetime64(data['timestamp'])
    weatherxx = toDatetime64(weather_data['timestamp'])

    c = 1.2
    plt.rcParams['figure.figsize'] = [12/c, 3/c]
    fig, ax1 = plt.subplots()

    
    # ax1.tick_params(axis='y', labelcolor=color)

    if plotDerivatives:
        yy = dxdt(data, param, ws, ws2=ws2, absv=absvDerivative)
        ax1.plot(dataxx, yy, label='dP/dt')
        ax1.set_ylabel(y_label) if y_label != 'default' else ax1.set_ylabel('Mean Polarization Drift Rate')
    else:
        ax1.plot(dataxx, data[param], label=param)
        ax1.set_ylabel(y_label) if y_label != 'default' else ax1.set_ylabel('Stokes Paramter Value')
        
            
    if fit_func != None:
        fitline = fit_func(weather_data)
        ax1.plot(weatherxx, fitline, color='#d62728', label='linear regression fit')
    
    colors = ['#d62728', '#9467bd', '#8c564b', 'orange']
    if len(weather_params) > 0:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Rel. Value' if len(weather_params) != 1 else weather_params[0])
    for i, yl in enumerate(weather_params):
        yy = weather_data[yl]
        if len(weather_params) > 1:
            yy = (yy-np.min(yy))/np.max(yy-np.min(yy)) # Normalize
        ax2.plot(weatherxx, yy, label=yl, color=colors[i])


    ax1.set_xlabel('Date/Time')
    #ax1.set_title(title) if title else ax1.set_title(data.attrs['filename'])
    ax1.set_title(title) if title else ax1.set_title('')
    save_file = ''
    bbox = None

    ###########################################################
    # # Timing
    # ax1.set_ylabel('Mean Delay (ns)')
    # ax2.set_ylabel(u'Temperature (\u00B0C)')
    # ax1.set_xlabel('Time (Date)')
    # ax1.set_title('')
    # bbox = (0.941, 0.948)
    # save_file = 'timing-drift.pdf'
    ###########################################################
    ###########################################################
    # # Polarization
    # ax1.set_ylabel('Mean Polarization\nDrift Rate (rad/sec)')
    # ax1.set_xlabel('Time (Date and Hour)')
    # ax1.set_title('')
    # bbox = (0.986, 0.948)
    # save_file = 'polarization-drift.pdf'
    ###########################################################


    fig.legend(loc='upper right', bbox_to_anchor=bbox)
    # plt.ylim(.988, .991)
    fig.tight_layout()
    if save_file: plt.savefig(save_file)
    plt.show()

def calculate_polarization_drift_rate(df):
    """
    Compute angle change (radians) and polarization drift rate (radians/sec)
    based on consecutive Stokes vectors (S1, S2, S3) in a time-indexed DataFrame.
    Assumes index is a sorted DatetimeIndex.
    """

    S = df[['S1', 'S2', 'S3']]

    # Next-row shifted vectors
    S_next = S.shift(-1)

    # Dot product and norms
    dot = (S * S_next).sum(axis=1)
    norm = np.linalg.norm(S.to_numpy(), axis=1)
    norm_next = np.linalg.norm(S_next.to_numpy(), axis=1)
    den = norm * norm_next

    # Angle between successive Stokes vectors (clipped for numerical stability)
    cos_theta = (dot / den).clip(-1, 1)
    angle_step_rad = np.arccos(cos_theta)

    # Time difference to next sample in seconds
    dt = (df.index.to_series().shift(-1) - df.index.to_series()).dt.total_seconds()

    # Drift rate (rad/s)
    drift_rate = angle_step_rad / dt

    # Fix invalid cases (zero norm or invalid dt)
    mask_invalid = (den == 0) | (~np.isfinite(den)) | (~np.isfinite(dt)) | (dt == 0)
    angle_step_rad[mask_invalid] = np.nan
    drift_rate[mask_invalid] = np.nan

    # Attach results to a copy of the dataframe
    df['delay-drift-rate'] = drift_rate

def main():
    weather_data = construct_weather_df()
    weather_data.to_csv("weather_data.csv")
     
    # weather_data['timestamp'] = weather_data['timestamp'].astype(float)

    if settings['dependentVar'] == DependentVar.POLARIZATION:
        polarization_dir = os.path.join(
            settings['basedir'],
            settings['experiment'], 
            settings['polarization_dirname']
        )
        polarizationData = get_polarization_data(polarization_dir)
        data = polarizationData

    elif settings['dependentVar'] == DependentVar.TIMING:
        timing_dir = os.path.join(settings['basedir'], settings['experiment'], 'TimingData')
        timingData = get_timing_data(timing_dir)
        data = timingData
        data.rename(columns={'Mean Delay': 'delay'}, inplace=True)
        data = data[['timestamp', 'delay']]

    # Reduce by factor of ws
    # data['timestamp'] = data['timestamp'].astype(float)

    # data = data.sort_values(by='timestamp').reset_index(drop=True)
    # data = data.groupby(data.index // settings['ws']).mean()
    # weather_data = weather_data.sort_values(by='timestamp')
    # #weather_data = weather_data[['timestamp'] + settings['weather_params']]

    # data = pd.merge_asof(data, weather_data, on='timestamp', direction='nearest')
    # data = data.dropna()

    data = data.sort_index()
    weather_data = weather_data.sort_index()

    #weather_data_aligned = weather_data.reindex(data.index)
    #data = pd.merge_asof(data, weather_data, on='timestamp', direction='nearest')
    #data = data.interpolate(method='time')

    for param in weather_data.columns:
        if weather_data[param].dtype in ['float64', 'int64']:
            col = np.interp(data.index, weather_data.index, weather_data[param])
            data[param] = col


    #data.to_csv("data-new.csv")


    #data = data.dropna()
    # Resample by settings['ws']*100ms
    #data = data.resample(f'{settings["ws"]*100}ms').mean()
    #data = data.rolling(settings['ws2'], center=True).mean()
    #data = data.dropna()
    #data.to_csv("merged_data.csv")


    if settings['dependentVar'] == DependentVar.POLARIZATION:
        #calculate_polarization_drift_rate(data)
        #data['delay'] = data['delay-drift-rate'].rolling(settings['ws2'], center=True).mean()
        #data['delay'] = data['delay-drift-rate']
        #data = data.dropna()
        pass
    elif settings['dependentVar'] == DependentVar.TIMING:
        data['delay-drift-rate'] = np.gradient(
            data['delay'],
            data['timestamp']
        )
        data['delay-drift-rate'] = data['delay-drift-rate'].rolling(settings['ws2'], center=True).mean()
        data = data.dropna()

    

    if settings['dependentVar'] == DependentVar.POLARIZATION:
        #data['delay-drift-rate'] = data['delay-drift-rate'].rolling('100s').mean()
        #data['predicted_delay-drift-rate'] = data.apply(f, axis=1)

        # plt.figure(figsize=(12, 5))
        # plt.plot(data.index, data['delay-drift-rate'], color='blue', label='Measured delay-drift-rate')
        # plt.plot(data.index, data['predicted_delay-drift-rate'], color='green', label='Predicted delay-drift-rate based on Weather')
        # plt.xlabel('Timestamp')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()


        data.index = data.index.round('ms')
        # data.index = data.index - pd.Timedelta(hours=4)

        data.to_csv("data.csv")
        data.to_pickle("data.pkl")
        

        # plot_with_weather(
        #     data=data,
        #     param='delay-drift-rate',
        #     weather_data=weather_data,
        #     weather_params=settings['weather_params'],
        #     ws=settings['ws'], ws2=settings['ws2'],
        #     plotDerivatives=True
        # )
        #weatherData, ['S1', 'S2', 'S3'], weatherparams=weather_params, ws=ws, ws2=ws2, plotDerivatives=True)
        #plot_with_weather(data, weatherData, ['S1', 'S2', 'S3'], weatherparams=[], ws=ws, ws2=ws2, plotDerivatives=True, fit_func=f)
        # plot_with_weather(df, weatherData, ['drift'], weatherparams=weather_params, ws=1, ws2=1, plotDerivatives=False)
    elif settings['dependentVar'] == DependentVar.TIMING:
        data['compensated-delay'] = data['delay'] - f(data)
        data['compensated-delay-drift-rate'] = np.gradient(
            data['compensated-delay'],
            data['timestamp']
        )
        data['compensated-delay-drift-rate'] = data['compensated-delay-drift-rate'].rolling(settings['ws2'], center=True).mean()

        data['smoothed-wss'] = data['wind-speed squared'].rolling(settings['ws2'], center=True).mean()

        print(" .................. printing weather data .................. ")
        
        weather_data = weather_data.dropna()
        print(type(weather_data))

        plot_with_weather(
            data=data,
            param='delay',
            weather_data=weather_data,
            weather_params=settings['weather_params'],
            ws=settings['ws'], ws2=settings['ws2'],
            plotDerivatives=False,
            y_label='Mean Delay (ps)'
        )
        plot_with_weather(
            data=data,
            param='delay',
            weather_data=weather_data,
            weather_params=[],
            ws=settings['ws'], ws2=settings['ws2'], 
            plotDerivatives=False, 
            fit_func=f, 
            y_label='Mean Delay (ps)'
        )
        plot_with_weather(
            data=data,
            param='compensated-delay',
            weather_data=data,
            weather_params=[],
            ws=settings['ws'], ws2=settings['ws2'],
            plotDerivatives=False, 
            y_label='Delay Drift (ps)', 
            title='Delay After Compensating for Temperature'
        )

        xparam2 = 'avg-wind-speed squared'
        yparam2 = 'compensated-delay-drift-rate'
        data = data.dropna()
        xx = data[xparam2]
        yy = data[yparam2]
        Xc = sm.add_constant(xx)
        est2_ = sm.OLS(yy, Xc)
        est2 = est2_.fit()
        print(est2.summary())
        reg2 = LinearRegression().fit(np.reshape(xx, (-1,1)), yy)

        f2 = lambda d : reg2.coef_[0]*d[xparam2] + reg2.intercept_
        # plot_with_weather(df, df, ['delay-drift'], weatherparams=[], ws=1, ws2=1, plotDerivatives=False, fit_func=f2)
        plot_with_weather(
            data=data,
            weather_data=data,
            param=yparam2, 
            weather_params=[], 
            ws=1, ws2=1, 
            plotDerivatives=False, 
            y_label='Mean Delay Drift Rate', 
            title='Rate of Delay Drift After Accounting for Temperature'
        )
        plot_with_weather(
            data=data, 
            weather_data=data, 
            param=yparam2,
            weather_params=[xparam2], 
            ws=1, ws2=1, 
            plotDerivatives=False, 
            y_label='Mean Delay Drift Rate', 
            title='Rate of Delay Drift After Accounting for Temperature'
        )

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel(xparam2)
        ax.set_ylabel('Mean Delay Drift Rate')
        ax.scatter(xx, yy, s=1)
        ax.plot(xx, reg2.coef_[0]*xx + reg2.intercept_, color='red')
        plt.show()

if __name__ == "__main__":
    main()

