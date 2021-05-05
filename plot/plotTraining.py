#!/bin/python3
import numpy as np
import matplotlib.pylab as plt
import os


from plotUtils import  makeCustomLegend, StdMean, color, readTensorBoardLog


dataDir = "../../logs"

fig, ax = plt.subplots()
ax2=ax.twinx()
ax.set_zorder(-1)  # default zorder is 0 for ax1 and ax2
ax.patch.set_visible(False)  # prevents ax1 from hiding ax2

data = readTensorBoardLog(os.path.join(dataDir, "test"))

ax.plot(np.arange(len(data["loss/test"]))    ,data["loss/test"], "-",  color = color(0,1), label = f"$\lambda = data $")
ax.plot(np.arange(len(data["loss/train"]))    ,data["loss/train"], "--",  color = color(0,1))
ax2.plot(np.arange(len(data["accuracy/test"]))    ,data["accuracy/test"], ":",  color = color(0,1))


ax.set_xscale('log')
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
# ax.set_xlim(-1,50)
ax2.set_ylim(0,1)
makeCustomLegend(ax,
                [
                    ({"c":"k", "linestyle": "-"}, r"$L^{test}$"),
                    ({"c":"k", "linestyle": "--"}, r"$L^{train}$"),
                    ({"c":"k", "linestyle": ":"}, r"accur"),
                ],
                bbox_to_anchor=(0,1.02,1,0.2),
                mode="expand",
                ncol = 3,
                loc = "lower left",
                borderaxespad=0)
ax.legend(loc = "upper left", bbox_to_anchor=(1.1,1), borderaxespad=0)
plt.subplots_adjust(right=0.75)
# plt.savefig(os.path.join(dataDir, f"loss.png"), bb_inches = "tight", dpi =300) 
plt.show()
plt.close()