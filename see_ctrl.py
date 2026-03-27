import numpy as np
import matplotlib.pyplot as plt

ctrls = np.load("control.npy")
plt.plot(ctrls)
plt.legend(["rot1", "pitch", "rot2", "tail"])
plt.show()