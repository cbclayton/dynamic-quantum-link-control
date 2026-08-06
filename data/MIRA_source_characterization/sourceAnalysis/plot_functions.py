from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import warnings

"""
Copyright 2020 University of Illinois Board of Trustees.
Licensed under the terms of an MIT license
"""

"""
Modified by cmn - Sept 2025


"""


"""CHECK OUT THE REFERENCE PAGE ON OUR WEBSITE :
https://quantumtomo.web.illinois.edu/Doc/"""


"""
makeRhoImages(p, plt_given, customColor)
Desc: Creates matlab plots of the density matrix.

Parameters
----------
p : ndarray with shape = (n, 2^numQubits, 2^numQubits)
    The density matrix you want to create plots of.
plt_given : matplotlib.pyplot
    Input pyplot for which the figures will be saved on to.
customColor : boolean
    Specify if you want our custom colorMap. Default is true

See Also
 ------ 
saveRhoImages
"""
warnings.filterwarnings("ignore")

def makeRhoImages(p, plt_given, outname, customColor = True):
    # Set up
    numQubits = int(np.log2(p.shape[0]))
    xpos = np.zeros_like(p.flatten(), dtype = float)
    ypos = np.zeros_like(p.flatten(), dtype = float)
    for i in range(0, 2**numQubits):
        xpos[i*2**numQubits:(1+i)*2**numQubits] = .5+i
    for i in range(0, 2**numQubits):
        ypos[i::2**numQubits] = .5+i
    zpos = np.zeros_like(p.flatten(), dtype = float)
    # width of cols
    dx = .9*np.ones_like(xpos)
    dy = .9*np.ones_like(ypos)
    # custom color map
    n_bin = 100
    if(customColor):
        from matplotlib.colors import LinearSegmentedColormap
        cmap_name = 'my_list'
        colors = [(1 / 255.0, 221 / 255.0, 137 / 255.0),
                  (32 / 255.0, 151 / 255.0, 138 / 255.0),
                  (53 / 255.0, 106 / 255.0, 138 / 255.0),
                  (86 / 255.0, 33 / 255.0, 139 / 255.0),
                  (131 / 255.0, 75 / 255.0, 114 / 255.0),
                  (173 / 255.0, 114 / 255.0, 90 / 255.0),
                  (253 / 255.0, 187 / 255.0, 45 / 255.0)]
        colorMap = LinearSegmentedColormap.from_list(cmap_name, colors, N = n_bin)
    else:
        colorMap = plt.cm.jet
    norm = mpl.colors.Normalize(vmin = -1, vmax = 1)

    tickBase = ["H", "V"]
    tick = [""]
    for x in range(numQubits):
        newTick = np.zeros(len(tick)*2, dtype = "O")
        for i in range(len(tick)):
            for j in range(len(tickBase)):
                newTick[len(tick)*i +j] = tick[i] + tickBase[j]
        tick = newTick
    xTicks = ["|"+x+">" for x in tick]
    yTicks = ["|"+x+">" for x in tick]


    # Real Graph
    fig = plt_given.figure()
    ax1 = fig.add_subplot(111, projection = '3d')
    dz = p.flatten().real.astype(float)
    img = ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color = colorMap((dz + 1) / 2), edgecolor = "black", alpha = .8)



    ax1.axes.set_xticks(range(1, 2**numQubits+1))
    ax1.axes.set_yticks(range(1, 2**numQubits+1))
    ax1.axes.set_zticks(np.arange(-1, 1.1, .2))
    ax1.axes.set_xticklabels(xTicks)
    ax1.axes.set_yticklabels(yTicks)
    ax1.axes.set_zlim3d(-1, 1)
    plt_given.title("Rho Real")
    fig.subplots_adjust(bottom = 0.2)
    ax1 = fig.add_axes([0.2, 0.10, 0.7, 0.065])
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap = colorMap,
                                    norm = norm,
                                    orientation = 'horizontal')

    # Imaginary graph
    fig = plt_given.figure()
    ax1 = fig.add_subplot(111, projection = '3d')
    dz = p.flatten().imag.astype(float)
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color = colorMap((dz + 1) / 2), edgecolor = "black", alpha = .8)

    ax1.axes.set_xticks(range(1, 2 ** numQubits + 1))
    ax1.axes.set_yticks(range(1, 2 ** numQubits + 1))
    ax1.axes.set_zticks(np.arange(-1, 1.1, .2))
    ax1.axes.set_xticklabels(xTicks)
    ax1.axes.set_yticklabels(yTicks)
    ax1.axes.set_zlim3d(-1, 1)
    plt_given.title("Rho Imaginary")

    fig.subplots_adjust(bottom = 0.2)
    ax1 = fig.add_axes([0.2, 0.10, 0.7, 0.065])
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap = colorMap,
                                    norm = norm,
                                    orientation = 'horizontal')

"""
saveRhoImages(p, pathToDirectory, customColor)
Desc: Creates and saves matlab plots of the density matrix.

Parameters
----------
p : ndarray with shape = (n, 2^numQubits, 2^numQubits)
    The density matrix you want to create plots of.
pathToDirectory : string
    Path to where you want your images to be saved.

See Also
 ------ 
makeRhoImages
"""
def saveRhoImages(p, outname, pathToDirectory):
    # Set up
    numQubits = int(np.log2(p.shape[0]))
    xpos = np.zeros_like(p.flatten(), dtype = float)
    ypos = np.zeros_like(p.flatten(), dtype = float)
    for i in range(0, 2 ** numQubits):
        xpos[i * 2 ** numQubits:(1 + i) * 2 ** numQubits] = .5 + i
    for i in range(0, 2 ** numQubits):
        ypos[i::2 ** numQubits] = .5 + i
    zpos = np.zeros_like(p.flatten(), dtype = float)
    # width of cols
    dx = .9 * np.ones_like(xpos)
    dy = .9 * np.ones_like(ypos)
    # custom color map
    n_bin = 100
    cmap_name = 'my_list'
    colors = [(1 / 255.0, 221 / 255.0, 137 / 255.0),
              (32 / 255.0, 151 / 255.0, 138 / 255.0),
              (53 / 255.0, 106 / 255.0, 138 / 255.0),
              (86 / 255.0, 33 / 255.0, 139 / 255.0),
              (131 / 255.0, 75 / 255.0, 114 / 255.0),
              (173 / 255.0, 114 / 255.0, 90 / 255.0),
              (253 / 255.0, 187 / 255.0, 45 / 255.0)]
    colorMap = LinearSegmentedColormap.from_list(cmap_name, colors, N = n_bin)

    norm = mpl.colors.Normalize(vmin = -1, vmax = 1)

    tickBase = ["H", "V"]
    tick = [""]
    for x in range(numQubits):
        newTick = np.zeros(len(tick) * 2, dtype = "O")
        for i in range(len(tick)):
            for j in range(len(tickBase)):
                newTick[len(tick) * i + j] = tick[i] + tickBase[j]
        tick = newTick
    xTicks = ["|" + x + ">" for x in tick]
    yTicks = ["|" + x + ">" for x in tick]

    # Real Graph
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection = '3d')
    dz = p.flatten().astype(float)
    img = ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color = colorMap((dz + 1) / 2), edgecolor = "black", alpha = .8)

    ax1.axes.set_xticklabels(xTicks)
    ax1.axes.set_yticklabels(yTicks)
    ax1.axes.set_xticks(range(1, 2 ** numQubits + 1))
    ax1.axes.set_yticks(range(1, 2 ** numQubits + 1))
    ax1.axes.set_zticks(np.arange(-1, 1.1, .2))
    ax1.axes.set_zlim3d(-1, 1)
    plt.title("Rho Real")
    fig.subplots_adjust(bottom = 0.2)
    ax1 = fig.add_axes([0.2, 0.10, 0.7, 0.065])
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap = colorMap,
                                    norm = norm,
                                    orientation = 'horizontal')
    plt.savefig(pathToDirectory + '/' + outname + "-RE.png", bbox_inches = 'tight', pad_inches = 0)

    # Imaginary graph
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection = '3d')
    dz = p.flatten().imag.astype(float)
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color = colorMap((dz + 1) / 2), edgecolor = "black", alpha = .8)

    ax1.axes.set_xticklabels(xTicks)
    ax1.axes.set_yticklabels(yTicks)
    ax1.axes.set_xticks(range(1, 2 ** numQubits + 1))
    ax1.axes.set_yticks(range(1, 2 ** numQubits + 1))
    ax1.axes.set_zticks(np.arange(-1, 1.1, .2))
    ax1.axes.set_zlim3d(-1, 1)
    plt.title("Rho Imaginary")
    fig.subplots_adjust(bottom = 0.2)
    ax1 = fig.add_axes([0.2, 0.10, 0.7, 0.065])
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap = colorMap,
                                    norm = norm,
                                    orientation = 'horizontal')

    plt.savefig(pathToDirectory + '/' + outname + "-IM.png", bbox_inches = 'tight', pad_inches = 0)
    
def saveRhoSubplots(p, outname, pathToDirectory, title):
    # Set up
    numQubits = int(np.log2(p.shape[0]))
    xpos = np.zeros_like(p.flatten(), dtype = float)
    ypos = np.zeros_like(p.flatten(), dtype = float)
    for i in range(0, 2 ** numQubits):
        xpos[i * 2 ** numQubits:(1 + i) * 2 ** numQubits] = .5 + i
    for i in range(0, 2 ** numQubits):
        ypos[i::2 ** numQubits] = .5 + i
    zpos = np.zeros_like(p.flatten(), dtype = float)
    # width of cols
    dx = .9 * np.ones_like(xpos)
    dy = .9 * np.ones_like(ypos)
    # custom color map
    n_bin = 100
    cmap_name = 'my_list'
    colors = [(1 / 255.0, 221 / 255.0, 137 / 255.0),
              (32 / 255.0, 151 / 255.0, 138 / 255.0),
              (53 / 255.0, 106 / 255.0, 138 / 255.0),
              (86 / 255.0, 33 / 255.0, 139 / 255.0),
              (131 / 255.0, 75 / 255.0, 114 / 255.0),
              (173 / 255.0, 114 / 255.0, 90 / 255.0),
              (253 / 255.0, 187 / 255.0, 45 / 255.0)]
    colorMap = LinearSegmentedColormap.from_list(cmap_name, colors, N = n_bin)

    norm = mpl.colors.Normalize(vmin = -1, vmax = 1)

    tickBase = ["H", "V"]
    tick = [""]
    for x in range(numQubits):
        newTick = np.zeros(len(tick) * 2, dtype = "O")
        for i in range(len(tick)):
            for j in range(len(tickBase)):
                newTick[len(tick) * i + j] = tick[i] + tickBase[j]
        tick = newTick
    xTicks = ["|" + x + ">" for x in tick]
    yTicks = ["|" + x + ">" for x in tick]

    # Real Graph
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection = '3d')
    dz = p.flatten().astype(float)
    img = ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color = colorMap((dz + 1) / 2), edgecolor = "black", alpha = .8)

    plt.suptitle(title)
    ax1.axes.set_xticklabels(xTicks)
    ax1.axes.set_yticklabels(yTicks)
    ax1.axes.set_xticks(range(1, 2 ** numQubits + 1))
    ax1.axes.set_yticks(range(1, 2 ** numQubits + 1))
    ax1.axes.set_zticks(np.arange(-1, 1.1, .25))
    plt.setp(ax1.get_zticklabels()[1::2], visible=False)
    ax1.axes.set_zlim3d(-1, 1)
    # ax1.tick_params(axis='z',labelsize=6)
    # plt.title("Rho Real")
    # fig.subplots_adjust(bottom = 0.2)
    ax1 = fig.add_axes([0.2, 0.10, 0.7, 0.065])
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap = colorMap,
                                    norm = norm,
                                    orientation = 'horizontal')
    # plt.savefig(pathToDirectory + '/' + outname + "-RE.png", bbox_inches = 'tight', pad_inches = 0)

    # Imaginary graph
    # fig = plt.figure()
    ax2 = fig.add_subplot(122, projection = '3d')
    dz = p.flatten().imag.astype(float)
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, color = colorMap((dz + 1) / 2), edgecolor = "black", alpha = .8)

    ax2.axes.set_xticklabels(xTicks)
    ax2.axes.set_yticklabels(yTicks)
    ax2.axes.set_xticks(range(1, 2 ** numQubits + 1))
    ax2.axes.set_yticks(range(1, 2 ** numQubits + 1))
    ax2.axes.set_zticks(np.arange(-1, 1.1, .25))
    plt.setp(ax2.get_zticklabels()[1::2], visible=False)
    ax2.axes.set_zlim3d(-1, 1)
    # plt.title("Rho Imaginary")
    # fig.subplots_adjust(bottom = 0.2)
    ax2 = fig.add_axes([0.2, 0.10, 0.7, 0.065])
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap = colorMap,
                                    norm = norm,
                                    orientation = 'horizontal')

    plt.savefig(pathToDirectory + '/' + outname + '.png', bbox_inches = 'tight', pad_inches = 0)
    
    # fig.table()
    
    plt.show()
    
    # mpl.rc('text', usetex=True)
    
    
    
    
    
    
    
    
    
    
def saveAllSubplots(p, pc, outname, pathToDirectory, title):
    # Set up
    numQubits = int(np.log2(p.shape[0]))
    xpos = np.zeros_like(p.flatten(), dtype = float)
    ypos = np.zeros_like(p.flatten(), dtype = float)
    for i in range(0, 2 ** numQubits):
        xpos[i * 2 ** numQubits:(1 + i) * 2 ** numQubits] = .5 + i
    for i in range(0, 2 ** numQubits):
        ypos[i::2 ** numQubits] = .5 + i
    zpos = np.zeros_like(p.flatten(), dtype = float)
    # width of cols
    dx = .9 * np.ones_like(xpos)
    dy = .9 * np.ones_like(ypos)
    # custom color map
    n_bin = 100
    cmap_name = 'my_list'
    colors = [(1 / 255.0, 221 / 255.0, 137 / 255.0),
              (32 / 255.0, 151 / 255.0, 138 / 255.0),
              (53 / 255.0, 106 / 255.0, 138 / 255.0),
              (86 / 255.0, 33 / 255.0, 139 / 255.0),
              (131 / 255.0, 75 / 255.0, 114 / 255.0),
              (173 / 255.0, 114 / 255.0, 90 / 255.0),
              (253 / 255.0, 187 / 255.0, 45 / 255.0)]
    colorMap = LinearSegmentedColormap.from_list(cmap_name, colors, N = n_bin)

    norm = mpl.colors.Normalize(vmin = -1, vmax = 1)

    tickBase = ["H", "V"]
    tick = [""]
    for x in range(numQubits):
        newTick = np.zeros(len(tick) * 2, dtype = "O")
        for i in range(len(tick)):
            for j in range(len(tickBase)):
                newTick[len(tick) * i + j] = tick[i] + tickBase[j]
        tick = newTick
    xTicks = ["|" + x + ">" for x in tick]
    yTicks = ["|" + x + ">" for x in tick]
    
    fig = plt.figure()
    
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.subplots_adjust(top=1.1,bottom=0.2)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(title,y=0)

    # Real Graph    
    ax1 = fig.add_subplot(221, projection = '3d')
    dz = p.flatten().astype(float)
    img = ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color = colorMap((dz + 1) / 2), edgecolor = "black", alpha = .8)
    
    ax1.axes.set_xticklabels(xTicks)
    ax1.axes.set_yticklabels(yTicks)
    ax1.axes.set_xticks(range(1, 2 ** numQubits + 1))
    ax1.axes.set_yticks(range(1, 2 ** numQubits + 1))
    ax1.axes.set_zticks(np.arange(-1, 1.1, .25))
    plt.setp(ax1.get_zticklabels()[1::2], visible=False)
    ax1.axes.set_zlim3d(-1, 1)
    # ax1.tick_params(axis='z',labelsize=6)
    plt.title("Rho Real")
    # fig.subplots_adjust(top=1.1,bottom=-0.5)
    ax1 = fig.add_axes([0.2, 0.10, 0.7, 0.065])
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap = colorMap,
                                    norm = norm,
                                    orientation = 'horizontal')
    
    # Imaginary graph
    ax2 = fig.add_subplot(222, projection = '3d')
    dz = p.flatten().imag.astype(float)
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, color = colorMap((dz + 1) / 2), edgecolor = "black", alpha = .8)

    ax2.axes.set_xticklabels(xTicks)
    ax2.axes.set_yticklabels(yTicks)
    ax2.axes.set_xticks(range(1, 2 ** numQubits + 1))
    ax2.axes.set_yticks(range(1, 2 ** numQubits + 1))
    ax2.axes.set_zticks(np.arange(-1, 1.1, .25))
    plt.setp(ax2.get_zticklabels()[1::2], visible=False)
    ax2.axes.set_zlim3d(-1, 1)
    plt.title("Rho Imaginary")
    # fig.subplots_adjust(top=1.1,bottom=0)
    ax2 = fig.add_axes([0.2, 0.10, 0.7, 0.065])
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap = colorMap,
                                    norm = norm,
                                    orientation = 'horizontal')

    # Real Graph (corrected)
    ax3 = fig.add_subplot(223, projection = '3d')
    dz = pc.flatten().astype(float)
    img = ax3.bar3d(xpos, ypos, zpos, dx, dy, dz, color = colorMap((dz + 1) / 2), edgecolor = "black", alpha = .8)
    
    ax3.axes.set_xticklabels(xTicks)
    ax3.axes.set_yticklabels(yTicks)
    ax3.axes.set_xticks(range(1, 2 ** numQubits + 1))
    ax3.axes.set_yticks(range(1, 2 ** numQubits + 1))
    ax3.axes.set_zticks(np.arange(-1, 1.1, .25))
    plt.setp(ax3.get_zticklabels()[1::2], visible=False)
    ax3.axes.set_zlim3d(-1, 1)
    # ax1.tick_params(axis='z',labelsize=6)
    plt.title("Rho Real (Corrected)")
    # fig.subplots_adjust(top=1.1,bottom = 0.2)
    ax3 = fig.add_axes([0.2, 0.10, 0.7, 0.065])
    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap = colorMap,
                                    norm = norm,
                                    orientation = 'horizontal')
    
    # Imaginary graph (corrected)
    ax4 = fig.add_subplot(224, projection = '3d')
    dz = pc.flatten().imag.astype(float)
    ax4.bar3d(xpos, ypos, zpos, dx, dy, dz, color = colorMap((dz + 1) / 2), edgecolor = "black", alpha = .8)

    ax4.axes.set_xticklabels(xTicks)
    ax4.axes.set_yticklabels(yTicks)
    ax4.axes.set_xticks(range(1, 2 ** numQubits + 1))
    ax4.axes.set_yticks(range(1, 2 ** numQubits + 1))
    ax4.axes.set_zticks(np.arange(-1, 1.1, .25))
    plt.setp(ax4.get_zticklabels()[1::2], visible=False)
    ax4.axes.set_zlim3d(-1, 1)
    plt.title("Rho Imaginary (Corrected)")
    
    ax4 = fig.add_axes([0.2, 0.10, 0.7, 0.065])
    cb1 = mpl.colorbar.ColorbarBase(ax4, cmap = colorMap,
                                    norm = norm,
                                    orientation = 'horizontal')
    
    plt.savefig(pathToDirectory + '/' + outname + '.png', bbox_inches = 'tight', pad_inches = 0)
    
    # fig.table()
    
    plt.show()
    
    # mpl.rc('text', usetex=True)
    
def plot_rawdata(basis_names,coinc,outfile=''):
    l = len(basis_names)

    plt.figure()
    plt.imshow(np.reshape(coinc,(l,l)),origin='lower')
    plt.xticks(range(l),basis_names)
    plt.yticks(range(l),basis_names)
    plt.title(outfile+'\n'+'Coincident counts')
    plt.colorbar()
    plt.show()
    
    if len(outfile) > 0: #save figure if output file is specified
        plt.savefig(outfile)