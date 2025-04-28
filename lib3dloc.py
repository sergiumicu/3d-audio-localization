import numpy as np, scipy as sp
import matplotlib.pyplot as plt

class Receiver:
    def __init__(self, x: float = 0, y: float = 0, z: float  = 0):
        self.coords = np.array([x, y, z])
        self.x = x
        self.y = y
        self.z = z
    def load(self, signal: np.array):
        self.signal = signal

class Emitter:
    def __init__(self, x: float = 0, y: float = 0, z: float  = 0):
        self.coords = np.array([x, y, z])
        self.x = x
        self.y = y
        self.z = z
    def load(self, signal: np.array):
        self.signal = signal

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
        Axes object to plot on. A new one will be created if none is specified
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