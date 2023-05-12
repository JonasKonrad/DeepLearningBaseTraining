#!/bin/python3
import numpy as np
import matplotlib.pylab as plt
import os
import h5py

from plotUtils import  makeCustomLegend, color

plt.rcParams['legend.framealpha'] = 1
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

plt.rcParams["text.usetex"] = True
plt.rc('text.latex', preamble=r"""
\usepackage[utf8]{inputenc} % allow utf-8 input
\usepackage[T1]{fontenc}    % use 8-bit T1 fonts
\usepackage{amsfonts}       % blackboard math symbols
\usepackage{nicefrac}       % compact symbols for 1/2, etc.
\usepackage{microtype}      % microtypography
\usepackage{amsmath}
\usepackage{bm}
\usepackage{mathtools}
""")
plt.rc('font', **{'family': 'serif'})

plt.rcParams["figure.subplot.top"] = 0.92
plt.rcParams["figure.subplot.bottom"] = 0.09
plt.rcParams["figure.subplot.right"] = 0.75
plt.rcParams["figure.subplot.left"] = 0.08
plt.rcParams["figure.figsize"] = (7,5)



dataDir = "../../logs/run2"
# dataDir = "../../logs/test"

fig, ax = plt.subplots()
ax2=ax.twinx()
ax.set_zorder(-1)  # default zorder is 0 for ax1 and ax2
ax.patch.set_visible(False)  # prevents ax1 from hiding ax2

with h5py.File(os.path.join(dataDir, "logs.hdf5"), "r", locking = False) as f:
    # ax.plot(np.arange(len(f["test"]["loss"][:])), f["test"]["loss"][:], "-",  color = color(0,1), label = f"$\lambda = data $")
    ax.plot(np.linspace(0, len(f["train"]["loss"][:]), len(f["train"]["lossPS"][:])), f["train"]["lossPS"][:], ".",  color = color(0,1), alpha = 0.5, markeredgecolor = "none", ms = 2)
    end = len(f["train"]["loss"][:])
    
    N = 256
    ax.plot(np.linspace(0, len(f["train"]["loss"][:]), len(f["train"]["lossPS"][:]))[(N-1)//2:-(N-1)//2], np.convolve(f["train"]["lossPS"][:], np.ones((N,))/N, mode='valid'), "-",  color = "darkorange")

    N = 2560
    ax.plot(np.linspace(0, len(f["train"]["loss"][:]), len(f["train"]["lossPS"][:]))[(N-1)//2:-(N-1)//2], np.convolve(f["train"]["lossPS"][:], np.ones((N,))/N, mode='valid'), "-",  color = "gold")

    ax.plot(np.arange(1, 1+len(f["train"]["loss"][:])), f["train"]["loss"][:], "-",  color = "red")
    
    ax.plot(np.arange(1, 1+len(f["test"]["loss"][:])), f["test"]["loss"][:], "-",  color = "lightblue")
    
    
    ax2.plot(np.arange(len(f["test"]["accuracy"][:])), f["test"]["accuracy"][:], ":",  color = color(0,1))


# with h5py.File(os.path.join(dataDir, "logs.hdf5"), "r", locking = False) as f:
#     # ax.plot(np.arange(len(f["test"]["loss"][:])), f["test"]["loss"][:], "-",  color = color(0,1), label = f"$\lambda = data $")
#     ax.plot(np.linspace(0, len(f["test"]["loss"][:]), len(f["test"]["lossPS"][:])), f["test"]["lossPS"][:], ".",  color = color(0,1), alpha = 0.5, markeredgecolor = "none", ms = 2)
#     end = len(f["test"]["loss"][:])
    
#     N = 256
#     ax.plot(np.linspace(0, len(f["test"]["loss"][:]), len(f["test"]["lossPS"][:]))[(N-1)//2:-(N-1)//2], np.convolve(f["test"]["lossPS"][:], np.ones((N,))/N, mode='valid'), "-",  color = "darkorange")

#     N = 2560
#     ax.plot(np.linspace(0, len(f["test"]["loss"][:]), len(f["test"]["lossPS"][:]))[(N-1)//2:-(N-1)//2], np.convolve(f["test"]["lossPS"][:], np.ones((N,))/N, mode='valid'), "-",  color = "gold")

#     ax.plot(np.arange(1, 1+len(f["test"]["loss"][:])), f["test"]["loss"][:], "-",  color = "red")
    
#     # ax2.plot(np.arange(len(f["test"]["accuracy"][:])), f["test"]["accuracy"][:], ":",  color = color(0,1))



# for i in range(200):
#     ax.axvline(i, color = "k")

# ax.set_xscale('log')
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax2.set_ylabel("Accuracy", rotation = -90, labelpad = 10)
ax.set_xlim(0,end)
ax.set_ylim(0,6)
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
# plt.savefig(os.path.join(dataDir, f"loss.png"), dpi =300) 
plt.show()
plt.close()