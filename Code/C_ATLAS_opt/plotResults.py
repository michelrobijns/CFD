#!/usr/local/bin/python3

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def main():
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    
    plotStreamFunctionContour()
    plotVorticityContour()
    plotPressureContour()

def plotStreamFunctionContour():
    streamFunction = np.loadtxt('streamFunction.dat')
    X = np.loadtxt('streamFunctionX.dat')
    Y = np.loadtxt('streamFunctionY.dat')
    
    levels = [-1.5e-3, -1e-3, -5e-4, -2.5e-4, -1e-4, -5e-5, -1e-5, -1e-6, 0,
              1e-10, 1e-5, 1e-4, 1e-2, 3e-2, 5e-2, 7e-2, 9e-2, 0.1, 0.11,
              0.115, 0.1175]
    
    plt.contour(X, Y, streamFunction, levels=levels, colors='k')
    plt.show()

def plotVorticityContour():
    vorticity = np.loadtxt('vorticity.dat')
    X = np.loadtxt('vorticityX.dat')
    Y = np.loadtxt('vorticityY.dat')
    
    levels = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    
    plt.contour(X, Y, vorticity, levels=levels, colors='k')
    plt.show()

def plotPressureContour():
    pressure = np.loadtxt('pressure.dat')
    X = np.loadtxt('pressureX.dat')
    Y = np.loadtxt('pressureY.dat')
    
    levels = [-0.002, 0, 0.02, 0.05, 0.07, 0.09, 0.11, 0.12, 0.17, 0.3]
    
    plt.contour(X, Y, pressure, levels=levels, colors='k')
    plt.show()

if __name__ == '__main__':
    main()
