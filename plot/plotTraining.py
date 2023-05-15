#!/bin/python3
import numpy as np
import matplotlib.pylab as plt
import os
import h5py

from plotUtils import  makeCustomLegend, color

plt.style.use("./presentation.mplstyle")
# plt.style.use("./paper.mplstyle")


dataDir = "../../logs/test"

fig, ax = plt.subplots()
ax2=ax.twinx()
ax.set_zorder(-1)  # default zorder is 0 for ax1 and ax2
ax.patch.set_visible(False)  # prevents ax1 from hiding ax2

with h5py.File(os.path.join(dataDir, "data.hdf5"), "r", locking = False) as f:
    ax.plot(np.arange(len(f["test"]["loss"][:])), f["test"]["loss"][:], "-",  color = color(0,1), label = f"$\lambda = data $")
    ax.plot(np.arange(len(f["train"]["loss"][:])), f["train"]["loss"][:], "--",  color = color(0,1))
    ax2.plot(np.arange(len(f["test"]["accuracy"][:])), f["test"]["accuracy"][:], ":",  color = color(0,1))

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax2.set_ylabel("Accuracy", rotation = -90, labelpad = 10)
ax.set_xlim(0,None)
ax.set_ylim(0,None)
ax2.set_ylim(0,1)
makeCustomLegend(ax,
                [
                    ({"c":"k", "linestyle": "-"}, r"$L^{test}$"),
                    ({"c":"k", "linestyle": "--"}, r"$L^{train}$"),
                    ({"c":"k", "linestyle": ":"}, r"Accuracy"),
                ],
                bbox_to_anchor=(0,1.02,1,0.2),
                mode="expand",
                ncol = 3,
                loc = "lower left",
                borderaxespad=0)
ax.legend(loc = "upper left", bbox_to_anchor=(1.1,1), borderaxespad=0)
# plt.savefig(os.path.join(dataDir, f"loss.pdf")) 
plt.show()
plt.close()