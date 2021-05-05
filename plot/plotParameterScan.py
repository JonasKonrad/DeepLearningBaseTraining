#!/bin/python3
import numpy as np
import matplotlib.pylab as plt
import os


from plotUtils import  makeCustomLegend, StdMean, color, readTensorBoardLog


dataDir = "../../logs"
endEpoch = 300
repititions = 4
paramList = ["1", "2", "3"]

fig, ax = plt.subplots()

bestAcc = {"acc" : 0}
accuracy = {}
for postfix in [""] + [f"_rep_{i}" for i in range(2,repititions+1)]:
    for i_param, param in enumerate(paramList):
        try:
            folder = f"name_{param}{postfix}"
            data = readTensorBoardLog(os.path.join(dataDir, folder))
        except FileNotFoundError:
            print(f"File: {folder} not found.")
            continue
        
        dat = data[f"accuracy/test"]

        if dat.shape[0] == endEpoch:
            maxAc = dat[-1]
            if param in accuracy.keys():
                accuracy[param].append(maxAc)
            else:
                accuracy[param] = [maxAc]
        else:
            print(f"Runs {folder} not finished. Epoch: {dat.shape[0]}/{endEpoch}.")

paramList_cleared = []
meanAcc = []
stdAcc = []
for param in paramList:
    if param in accuracy.keys():
        paramList_cleared.append(param)
        meanAcc.append(np.mean(accuracy[param]))
        stdAcc.append(StdMean(accuracy[param]))

        if meanAcc[-1] > bestAcc["acc"] and np.isfinite(stdAcc[-1]):
            bestAcc={"acc": meanAcc[-1], "std": stdAcc[-1], "param": param}

meanAcc = np.array(meanAcc)
stdAcc = np.array(stdAcc)


ax.plot([float(param) for param in paramList_cleared], meanAcc, "-", color= "k")
ax.fill_between([float(param) for param in paramList_cleared], meanAcc-stdAcc, meanAcc+stdAcc, facecolor= "k", alpha = 0.4, edgecolor = None )
ax.scatter([float(param) for param in paramList_cleared for _ in range(len(accuracy[param]))], [accuracy[param][i] for param in paramList_cleared for i in range(len(accuracy[param]))], marker= ".", color= "k")


##baseline
blAcc = 0.8167500027588436
blStd = 0.00033224558418281904

print(f"Baseline: {blAcc} +- {blStd}")
print(f"Best    : {bestAcc['acc']} +- {bestAcc['std']}")
print(f"param: {bestAcc['param']}, gain: {bestAcc['acc']-blAcc} +- {np.sqrt((bestAcc['std'])**2 + (blStd)**2)}" )

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
