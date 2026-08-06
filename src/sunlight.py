import numpy as np
import matplotlib.pyplot as plt

def deg2rad(theta):
    return np.pi * theta / 180

def rad2deg(theta):
    return 180 * theta / np.pi

def sind(theta):
    return np.sin(deg2rad(theta))

def cosd(theta):
    return np.cos(deg2rad(theta))

def tand(theta):
    return np.tan(deg2rad(theta))

def toDateTimeUTC(timestamps):
    return (10**6*timestamps).astype('datetime64[us]')

def dAndLTFromTimestamp(timestamps, deltaT_GMT):
    # d = day of the year
    # LT = local time (in hours)
    o = toDateTimeUTC(np.array(timestamps)).astype(object)
    months = np.array([x.month for x in o])
    days = np.array([x.day for x in o])
    hours = np.array([x.hour for x in o])
    minutes = np.array([x.minute for x in o])
    seconds = np.array([x.second for x in o])
    microseconds = np.array([x.microsecond for x in o])
    LT = hours + minutes/60 + seconds/60/60 + microseconds/60/60/10**6
    d = np.array([x.timetuple().tm_yday for x in o])
    return d, LT + deltaT_GMT


def solarInsolation(timestamps, latitude, longitude, deltaT_GMT=-4):
    # deltaT_GMT: difference in local time from GMT (-4 for EDT)
    d, LT = dAndLTFromTimestamp(timestamps, deltaT_GMT)
    LSTM = 15 * deltaT_GMT
    B = 360 * (d-81) / 365
    EoT = 9.87*sind(2*B) - 7.53*cosd(B) - 1.5*sind(B)
    TC = 4*(longitude - LSTM) + EoT
    delta = 23.45 * sind(B) # solar declination
    sunrise = 12 - 1/15 * rad2deg(np.arccos(-tand(latitude)*tand(delta))) - TC/60
    sunset  = 12 + 1/15 * rad2deg(np.arccos(-tand(latitude)*tand(delta))) - TC/60
    mask = np.logical_and(LT >= np.full_like(LT, sunrise), LT <= np.full_like(LT, sunset))
    LST = LT + TC/60
    HRA = 15 * (LST - 12)
    alpha = rad2deg(np.arcsin(sind(delta)*sind(latitude) + cosd(delta)*cosd(latitude)*cosd(HRA))) # solar elevation
    azimuth = rad2deg(np.arccos((sind(delta)*cosd(latitude) - cosd(delta)*sind(latitude)*cosd(HRA)) / cosd(alpha)))
    zenith = 90 - alpha
    zenith = np.clip(zenith, 0, 89.9)
    AM = 1/cosd(zenith)
    AM2 = 1/(cosd(zenith) + 0.50572*(96.07995-zenith)**-1.6364)
    I_D = 1.353 * 0.7**(AM2**0.678)
    return np.nan_to_num(I_D) * mask


latitude, longitude = 39.005212, -76.943433
xx = np.arange(4, 22, .1)
# yy = solarInsolation(xx)
# plt.plot(xx, yy)
# plt.xlabel("Local Time")
# plt.ylabel("Solar Insolation")
# plt.show()


