import lib3dloc, numpy as np
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt

sample_rate, audio = wavfile.read(os.path.join('demo_files', 'chill1.wav'))

if np.shape(audio)[1] > 1: # make mono
    audio = np.mean(audio, axis=1)

audio = audio[:sample_rate] # get first second of sound

receivers = [lib3dloc.Receiver(1,   0.0, 0),
             lib3dloc.Receiver(1  ,  .1, 0),
             lib3dloc.Receiver(1  , -.1, 0),
             lib3dloc.Receiver(1.1, 0.0, 0),
             lib3dloc.Receiver(0.9, 0.0, 0)]

emmiter = lib3dloc.Emitter(0, -.5, 1)
emmiter.load(audio)

for i in len(receivers):
    d = lib3dloc.compute_delay(receivers[i], emmiter)

    receivers[i].load(lib3dloc.simulate_delay(emmiter.signal, d, sample_rate))

ax = lib3dloc.plot_devices(emmiter, color='r', marker='x')
lib3dloc.plot_devices(receivers, ax=ax)
ax.axis('equal')
plt.show()


