#!/usr/local/bin/python3

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage

def main():
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    
    plotStreamFunctionContour()
    plotVorticityContour()
    plotPressureContour()

def plotStreamFunctionContour():
    streamFunction = np.loadtxt('streamFunction.dat')
    X = np.loadtxt('streamFunctionX.dat')
    Y = np.loadtxt('streamFunctionY.dat')
    
    streamFunction = scipy.ndimage.zoom(streamFunction, 5)
    X = scipy.ndimage.zoom(X, 5)
    Y = scipy.ndimage.zoom(Y, 5)
    
    levels = [-1.5e-3, -1e-3, -5e-4, -2.5e-4, -1e-4, -5e-5, -1e-5, -1e-6, 0,
              1e-10, 1e-5, 1e-4, 1e-2, 3e-2, 5e-2, 7e-2, 9e-2, 0.1, 0.11,
              0.115, 0.1175]
    
    plt.figure(num='Stream function', figsize=(10, 10))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.contour(X, Y, streamFunction, levels=levels, colors='k')
    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])
    frame.axes.yaxis.set_ticklabels([])
    frame.xaxis.set_major_locator(plt.NullLocator())
    frame.yaxis.set_major_locator(plt.NullLocator())
    plt.savefig('streamFunction.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def plotVorticityContour():
    vorticity = np.loadtxt('vorticity.dat')
    X = np.loadtxt('vorticityX.dat')
    Y = np.loadtxt('vorticityY.dat')
    
    vorticity = scipy.ndimage.zoom(vorticity, 5)
    X = scipy.ndimage.zoom(X, 5)
    Y = scipy.ndimage.zoom(Y, 5)
    
    levels = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    
    plt.figure(num='Vorticity', figsize=(10, 10))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.contour(X, Y, vorticity, levels=levels, colors='k')
    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])
    frame.axes.yaxis.set_ticklabels([])
    frame.xaxis.set_major_locator(plt.NullLocator())
    frame.yaxis.set_major_locator(plt.NullLocator())
    plt.savefig('vorticity.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def plotPressureContour():
    pressure = np.loadtxt('pressure.dat')
    X = np.loadtxt('pressureX.dat')
    Y = np.loadtxt('pressureY.dat')
    
    pressure = scipy.ndimage.zoom(pressure, 5)
    X = scipy.ndimage.zoom(X, 5)
    Y = scipy.ndimage.zoom(Y, 5)
    
    levels = [-0.002, 0, 0.02, 0.05, 0.07, 0.09, 0.11, 0.12, 0.17, 0.3]
    
    plt.figure(num='Pressure', figsize=(10, 10))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.contour(X, Y, pressure, levels=levels, colors='k')
    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])
    frame.axes.yaxis.set_ticklabels([])
    frame.xaxis.set_major_locator(plt.NullLocator())
    frame.yaxis.set_major_locator(plt.NullLocator())
    plt.savefig('pressure.pdf', format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
