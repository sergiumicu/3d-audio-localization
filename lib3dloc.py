import numpy as np, scipy as sp
import matplotlib.pyplot as plt

class Device:
    def __init__(self, x: float = 0, y: float = 0, z: float  = 0):
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

class Emitter(Device):
    def __init__(self, x: float = 0, y: float = 0, z: float  = 0):
        super().__init__(x, y, z)

class Receiver(Device):
    def __init__(self, x: float = 0, y: float = 0, z: float  = 0):
        super().__init__(x, y, z)

def compute_delay(rec: Receiver, em: Emitter, v: float = 343) -> float:
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

def plot_devices(device_list: list[Receiver] | list[Emitter] | Receiver | Emitter, ax: plt.Axes = None, color = 'k', marker = 'o') -> plt.Axes:
    """Plots specified devices 

    Parameters
    ----------
    device_list: list[Receiver] | list[Emitter] | Receiver | Emitter
        Reciever or Emitter to plot. Can also be a list of Recievers or Emitter.
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
    
    if type(device_list) == Receiver or type(device_list) == Emitter:
        device_list = [device_list]

    if type(device_list) == list:
        for device in device_list:
            ax.scatter(device.x, device.y, device.z, c=color, marker=marker)

    return ax

def estimate_delays(receivers: list[Receiver]) -> np.matrix:
    if not receivers:
        raise ValueError 
    
    signals = []

    for rec in receivers:
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

            # print(k[peak])
            # plt.plot(np.unwrap(np.unwrap(np.angle(np.fft.fft(signals[i]))) - np.unwrap(np.angle(np.fft.fft(signals[j])))))
            # plt.show()

    return dist_dif

def solve_position_lsq(receivers: list[Receiver], dist_dif: np.matrix, reference_receiver: int = 0) -> np.array:
    """Solves for position of emitter using the least squares method

    Parameters
    ----------
    receivers: list[Receiver] 
        List of receivers
    dist_dif: np.matrix
        Matrix of distance differences between receivers
    reference_receiver: default=0
        The receiver that is used as reference in the LS algorithm

    Returns
    -------
    x: np.array
        Coordinates of emitter
    """
    receivers[0], receivers[reference_receiver] = receivers[0], receivers[reference_receiver]

    r = np.zeros((len(receivers) - 1, 1))
    A = np.zeros((len(receivers) - 1, len(receivers[0].coords) + 1))

    for i in range(len(r)):
        d0i = dist_dif[0, i+1] 
        r[i] = np.linalg.norm(receivers[0].coords) ** 2 - np.linalg.norm(receivers[i+1].coords) ** 2+ d0i**2
        A[i, 0:3] = receivers[0].coords - receivers[i+1].coords
        A[i, 3] = d0i

    return np.linalg.lstsq(A, .5*r)[0][:3]

def solve_position_tlsq(receivers: list[Receiver], dist_dif: np.matrix, reference_receiver: int = 0) -> np.array:
    """Solves for position of emitter using the total least squares method

    Parameters
    ----------
    receivers: list[Receiver] 
        List of receivers
    dist_dif: np.matrix
        Matrix of distance differences between receivers
    reference_receiver: default=0
        The receiver that is used as reference in the LS algorithm

    Returns
    -------
    x: np.array
        Coordinates of emitter
    """
    receivers[0], receivers[reference_receiver] = receivers[reference_receiver], receivers[0]

    r = np.zeros((len(receivers) - 1, 1))
    A = np.zeros((len(receivers) - 1, len(receivers[0].coords) + 1))

    for i in range(len(r)):
        d0i = dist_dif[0, i+1]
        r[i] = np.linalg.norm(receivers[0].coords) ** 2 - np.linalg.norm(receivers[i+1].coords) ** 2+ d0i**2
        A[i, 0:3] = receivers[0].coords - receivers[i+1].coords
        A[i, 3] = d0i

    r *= .5

    Ar = np.concatenate((A, r), axis=1)

    s = sp.linalg.svd(Ar)
    s = np.min(s[1])
    
    x = np.linalg.inv(A.transpose() @ A - s**2 * np.eye(len(A.transpose() @ A))) @ A.transpose() @ r

    return x[:3]