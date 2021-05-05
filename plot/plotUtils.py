#!/bin/python3
import numpy as np
import matplotlib.pylab as plt
import os
from tensorboard.backend.event_processing import event_accumulator as ea


def StdMean(data,ddof=1,t_verteilung=True):
    """ gibt standardunsicherheit des mittelwertes einer liste unter beruecksichtigung der studentschen t verteilung aus, basiert auf np.std
    """
    t_faktor=[0,1.84,1.32,1.20,1.14,1.11,1.09,1.08,1.07,1.06,1.05,1.05,1.04,1.04,1.04]
    n=len(data)
    if t_verteilung:
        if n <15:
            t_fak=t_faktor[n]
        elif n<25:
            t_fak=1.03
        elif n<35:
            t_fak=1.02
        elif n<100:
            t_fak=1.01
        else:
            t_fak=1
    else:
        t_fak=1
    return np.std(data,ddof=ddof)/np.sqrt(n)*t_fak

def color(i,N):
    if N==1: return  ["k"][i]
    if N==2: return  ["darkblue","darkred"][i]
    if N==3: return  ["darkgreen","darkblue","darkred"][i]
    if N==4: return  ["darkgreen","darkblue","darkred","darkorange"][i]
    if N==5: return  ["darkgreen","darkcyan","darkblue","darkred","darkorange"][i]
    if N==6: return  ["darkgreen","darkcyan","darkblue","darkred","darkorange","gold"][i]
    if N==7: return  ["darkgreen","darkcyan","darkblue","purple","darkred","darkorange","gold"][i]
    if N==8: return  ["darkgreen","darkcyan","blue","darkblue","purple","darkred","darkorange","gold"][i]
    if N==9: return  ["darkgreen","darkcyan","blue","darkblue","purple","darkred","orangered","darkorange","gold"][i]
    if N==10:return  ["darkgreen","limegreen","darkcyan","blue","darkblue","purple","darkred","orangered","darkorange","gold"][i]

    Nmax=10
    from matplotlib import colors
    from scipy.interpolate import interp1d

    rgbaData=[]
    for j in range(Nmax):
        rgbaData.append(colors.to_rgba(color(j,Nmax)))
    rgbaData=np.array(rgbaData)

    x=np.linspace(0,1,Nmax)
    fr=interp1d(x,rgbaData[:,0], kind="cubic")
    fg=interp1d(x,rgbaData[:,1], kind="cubic")
    fb=interp1d(x,rgbaData[:,2], kind="cubic")
    RGBA=np.array([fr(i/(N-1)),fg(i/(N-1)),fb(i/(N-1)),1])
    RGBA[np.where(RGBA>1)]=1
    RGBA[np.where(RGBA<0)]=0
    return RGBA


def colorInterpol(colorList, i, N):
    from matplotlib import colors
    from scipy.interpolate import interp1d

    rgbaData=[]
    for c in colorList:
        rgbaData.append(colors.to_rgba(c))
    rgbaData=np.array(rgbaData)

    x=np.linspace(0,1,len(colorList))
    fr=interp1d(x,rgbaData[:,0], kind=3)
    fg=interp1d(x,rgbaData[:,1], kind=3)
    fb=interp1d(x,rgbaData[:,2], kind=3)
    RGBA=np.array([fr(i/(N-1)),fg(i/(N-1)),fb(i/(N-1)),1])
    RGBA[np.where(RGBA>1)]=1
    RGBA[np.where(RGBA<0)]=0
    return RGBA

def makeCustomLegend(axis, styleNameTuples, **kwargs):
    from matplotlib.lines import Line2D
    lines = [Line2D([], [], **style) for style, _ in styleNameTuples]
    plt.sca(axis)
    legend = plt.legend(lines, list(zip(*styleNameTuples))[1], **kwargs)
    axis.add_artist(legend)



def getTensorBoardFileName(path):
    fileName = None
    for file in os.listdir(path):
        if file.startswith("events.out"):
            if fileName is None:
                fileName = os.path.join(path, file)
            else:
                raise ValueError("found more than one tensorboard log file")
    return fileName

def convertTensorBoardEvent_to_npArray(event):
    return np.array([event[i][2] for i in range(len(event))])


def readTensorBoardLog(path):
    file  = getTensorBoardFileName(path)

    acc = ea.EventAccumulator(file, size_guidance={'tensors': 0,})
    acc.Reload()

    return {tensorName : convertTensorBoardEvent_to_npArray(acc.Scalars(tensorName)) for tensorName in acc.scalars.Keys()}
