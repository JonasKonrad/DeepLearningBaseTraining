#!/bin/python3
import numpy as np
import matplotlib.pylab as plt
import os
import h5py


from plotUtils import  StdMean


dataDir = "../../logs"
endEpoch = 5
repititions = 2
paramList = ["1", "2", "3"]

fig, ax = plt.subplots()

accuracy = {}
for postfix in [""] + [f"_rep_{i}" for i in range(2,repititions+1)]:
    for i_param, param in enumerate(paramList):
        try:
            folder = f"test_{param}{postfix}"
            with h5py.File(os.path.join(dataDir, folder, "logs.hdf5"), "r") as f:
                data = f["test"]["accuracy"][:]
        except OSError:
            print(f"No Logfile found in folder {folder}.")
            continue

        if data.shape[0] == endEpoch:
            lastAccur = data[-1]
            if param in accuracy.keys():
                accuracy[param].append(lastAccur)
            else:
                accuracy[param] = [lastAccur]
        else:
            print(f"Runs {folder} not finished. Epoch: {data.shape[0]}/{endEpoch}.")

paramList_cleared = []
meanAcc = []
stdAcc = []
bestAcc = {"accur" : 0}
for param in paramList:
    if param in accuracy.keys():
        paramList_cleared.append(param)
        meanAcc.append(np.mean(accuracy[param]))
        stdAcc.append(StdMean(accuracy[param]))

        if meanAcc[-1] > bestAcc["accur"]:
            bestAcc={"accur": meanAcc[-1], "std": stdAcc[-1], "param": param}

meanAcc = np.array(meanAcc)
stdAcc = np.array(stdAcc)


ax.plot([float(param) for param in paramList_cleared], meanAcc, "-", color= "k")
ax.fill_between([float(param) for param in paramList_cleared], meanAcc-stdAcc, meanAcc+stdAcc, facecolor= "k", alpha = 0.4, edgecolor = None )
ax.scatter([float(param) for param in paramList_cleared for _ in range(len(accuracy[param]))], [accuracy[param][i] for param in paramList_cleared for i in range(len(accuracy[param]))], marker= ".", color= "k")


##baseline
blAcc = 0.8167500027588436
blStd = 0.00033224558418281904

print(f"Baseline: {blAcc} +- {blStd}")
print(f"Best    : {bestAcc['accur']} +- {bestAcc['std']}")
print(f"param: {bestAcc['param']}, gain: {bestAcc['accur']-blAcc} +- {np.sqrt((bestAcc['std'])**2 + (blStd)**2)}" )

ax.axhline(blAcc, color = "k")
ax.axhspan(blAcc-blStd, blAcc+blStd, alpha=0.4, color='grey', ec = None)

# ax.set_xscale('log')
# ax.set_xscale('symlog')
ax.set_xlabel(fr"parameter")
ax.set_ylabel("accuracy")
# ax.legend()
# plt.savefig(os.path.join(dataDir, f"accur.png"), bbox_inches = "tight", dpi =300) 
plt.show()
plt.close()
