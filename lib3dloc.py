import numpy as np, scipy as sp
import matplotlib.pyplot as plt
import time

class Device:
    def __init__(self, x: float | np.ndarray | list = 0, y: float = 0, z: float  = 0):
        if isinstance(x, np.ndarray) or isinstance(x, list):
            if len(x) == 3:
                self.x = x[0]
                self.y = x[1]
                self.z = x[2]
                self.coords = x
                return
            else:
                x = 0
                y = 0
                z = 0   

        self.coords = np.array([x, y, z])
        self.x = x
        self.y = y
        self.z = z
    def load(self, signal: np.array):
        self.signal = signal
    def __rmul__(self, other: float):
        toret = Device()
        toret.coords = other * self.coords
        return toret
    def __mul__(self, other: float):
        return self.__rmul__(self, other)

def compute_delay(rec: Device, em: Device, v: float = 343) -> float:
    return np.linalg.norm(rec.coords - em.coords, 2) / v

def simulate_delay(signal: np.array, delay: float, sample_rate: float) -> np.array:
    """Simulates a delay applied to a signal. Assumes the signal is sampled according to the Nyquist-Shannon Theorem
    
    Parameters
    ----------
    signal: np.array
        Signal to be delayed.
    delay: float
        Time to delay it by. Unit is u.
    sample_rate: float
        Sample rate of the analog signal. Unit is samples/u.

    Returns
    -------
    delayed_signal: np.array
        The signal, with the simulated delay.
    """
    interpolator = sp.interpolate.interp1d(np.arange(0, len(signal) / sample_rate, 1/sample_rate), signal, fill_value=0, bounds_error=False)
    return interpolator(np.arange(0, len(signal) / sample_rate, 1/sample_rate) - delay)

def plot_devices(device_list: list[Device] | Device, ax: plt.Axes = None, color = 'k', marker = 'o') -> plt.Axes:
    """Plots specified devices 

    Parameters
    ----------
    device_list: list[Device] | Device
        Reciever or Device to plot. Can also be a list of Recievers or Device.
    ax: matplotlib.pyplot.Axes, optional
        Axes object to plot on. A new one will be created if none is specified.
    color: default='k'
        matplotlib compatible color
    marker: default='o'
        maptlotlib compatible marker

    Returns
    -------
    ax: matplotlib.pyplot.Axes
        Axes object plotted on.
    """
    if not ax:
        ax = plt.figure().add_subplot(projection='3d')
    
    if type(device_list) == Device:
        device_list = [device_list]

    if type(device_list) == list:
        for device in device_list:
            ax.scatter(device.x, device.y, device.z, c=color, marker=marker)

    return ax

def estimate_delays(Devices: list[Device]) -> np.matrix:
    if not Devices:
        raise ValueError 
    
    signals = []

    for rec in Devices:
        signals.append(rec.signal)

    nsignals = len(signals)

    dist_dif = np.zeros((nsignals, nsignals))

    for i in range(nsignals):
        for j in range(nsignals):
            xcorr = sp.signal.correlate(signals[i]/np.max(np.abs(signals[i])), 
                                        signals[j]/np.max(np.abs(signals[j])), "same")
            k = sp.signal.correlation_lags(len(signals[i]), len(signals[j]), "same")
            peak = np.argmax(xcorr)
            dist_dif[i][j] = k[peak]

    return dist_dif

def calc_gradient(Devices: list[Device], est_delays: np.matrix, v: float, em_est: np.array) -> np.array:
    toret = np.array([0,0,0,0], dtype=float)

    for i in range(len(Devices)):    
        for j in range(len(Devices)):
            toadd = np.array([0,0,0,0], dtype=float)

            toadd[0:3] += (em_est - Devices[i].coords)/np.linalg.norm(em_est - Devices[i].coords)
            toadd[0:3] -= (em_est - Devices[j].coords)/np.linalg.norm(em_est - Devices[j].coords)
            toadd[3] -= est_delays[i][j]

            toadd *= (np.linalg.norm(em_est - Devices[i].coords) - np.linalg.norm(em_est - Devices[j].coords) - v * est_delays[i][j])

            toret += toadd

    return 2*toret

def solve_position_itr(receivers: list[Device], est_delays: np.matrix, v: float = 343, reference_device: int = 0, max_time: float = .1) -> np.array:
    """Solves for position of receiver using the least squares method, then refines it iteratatively

    Parameters
    ----------
    receivers: list[Device] 
        List of receivers
    est_delays: np.matrix
        Matrix of time differences between receivers
    v: float default=343
        Speef of sound in meters / second
    reference_receiver: default=0
        The receiver that is used as reference in the LS algorithm
    max_time: float default=0.1
        How much time will be spent refining each solution, in seconds

    Returns
    -------
    estimate: np.array
        Refined coordinates of emitter and speed of sound estimate
    """
    init = np.squeeze(solve_position_lsq(receivers, est_delays * v, reference_device))

    alpha = 1
    lastsol = np.zeros(4)

    now = time.time()

    while time.time()-now < max_time:
        dJ = calc_gradient(receivers, est_delays, v, init)

        init -= alpha * np.array(dJ[0:3])
        v -= 10*alpha * dJ[3]

        lastsol[:3] = init
        lastsol[3] = v

    return lastsol


def solve_position_lsq(Devices: list[Device], dist_dif: np.matrix, reference_device: int = 0) -> np.array:
    """Solves for position of receiver using the least squares method

    Parameters
    ----------
    receivers: list[Device] 
        List of receivers
    dist_dif: np.matrix
        Matrix of distance differences between receivers
    reference_device: default=0
        The receiver that is used as reference in the LS algorithm

    Returns
    -------
    x: np.array
        Coordinates of emitter 
    """
    Devices[0], Devices[reference_device] = Devices[0], Devices[reference_device]

    r = np.zeros((len(Devices) - 1, 1))
    A = np.zeros((len(Devices) - 1, len(Devices[0].coords) + 1))

    for i in range(len(r)):
        d0i = dist_dif[0, i+1] 
        r[i] = np.linalg.norm(Devices[0].coords) ** 2 - np.linalg.norm(Devices[i+1].coords) ** 2+ d0i**2
        A[i, 0:3] = Devices[0].coords - Devices[i+1].coords
        A[i, 3] = d0i

    return np.linalg.lstsq(A, .5*r)[0][:3]

def solve_position_tlsq(Devices: list[Device], dist_dif: np.matrix, reference_device: int = 0) -> np.array:
    """Solves for position of receiver using the total least squares method

    Parameters
    ----------
    receivers: list[Device] 
        List of receivers
    dist_dif: np.matrix
        Matrix of distance differences between receivers
    reference_device: default=0
        The receiver that is used as reference in the TLS algorithm

    Returns
    -------
    x: np.array
        Coordinates of the emitter 
    """
    Devices[0], Devices[reference_device] = Devices[reference_device], Devices[0]

    r = np.zeros((len(Devices) - 1, 1))
    A = np.zeros((len(Devices) - 1, len(Devices[0].coords) + 1))

    for i in range(len(r)):
        d0i = dist_dif[0, i+1]
        r[i] = np.linalg.norm(Devices[0].coords) ** 2 - np.linalg.norm(Devices[i+1].coords) ** 2+ d0i**2
        A[i, 0:3] = Devices[0].coords - Devices[i+1].coords
        A[i, 3] = d0i

    r *= .5

    Ar = np.concatenate((A, r), axis=1)

    s = sp.linalg.svd(Ar)
    s = np.min(s[1])
    
    x = np.linalg.inv(A.transpose() @ A - s**2 * np.eye(len(A.transpose() @ A))) @ A.transpose() @ r

    return x[:3]