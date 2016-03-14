#!/usr/local/bin/python3

import sys
import numpy as np
from math import *
import matplotlib.pyplot as plt
import scipy.linalg

def main():
    if (len(sys.argv) != 3):
        print('Wrong number of arguments\nUsage:', sys.argv[0], 'dt N')
        exit()

    np.set_printoptions(precision=4, linewidth=178)

    tol = 1e-5
    Re = 3200
    dt = float(sys.argv[1])
    N = int(sys.argv[2])

    # Mesh
    x, h, tx, th = generateMesh(N)

    # Matrices
    tE21 = generateTE21(N)
    E10 = -np.transpose(tE21)
    E21K = generateE21K(N)
    E21 = generateE21(N)
    tE10 = np.transpose(E21)
    H1t1 = generateH1t1(N, h)
    Ht11 = np.linalg.inv(H1t1)
    Ht02 = generateHt02(N, th)
    A = tE21.dot(Ht11).dot(E10)
    LU = scipy.linalg.lu_factor(A)

    # Vectors
    u = np.zeros(2 * N * (N - 1))
    uK = generateUK(N, th)
    uPres = H1t1.dot(tE10).dot(Ht02).dot(E21K).dot(uK)

    # Constants
    C0 = Ht02.dot(E21)
    C1 = tE21.dot(Ht11)
    C2 = H1t1.dot(tE10).dot(C0)
    C3 = uPres / Re

    diff = 1
    iteration = 0

    # Loop
    while (diff > tol):
        iteration += 1

        xi = C0.dot(u)

        convective = generateConvective(N, xi, u, uK, th)
        
        C4 = C2.dot(u / Re) + C3 + convective

        rhs = C1.dot(u / dt - C4)

        P = scipy.linalg.lu_solve(LU, rhs)

        uOld = np.copy(u)

        u -= dt * (E10.dot(P) + C4)

        if (iteration % 1000 == 0):
            diff = max(np.abs(u - uOld)) / dt
            print('Iteration', iteration ,'\tdiff =', diff)

    print('Converged after', iteration, 'iterations.')

    plotContour(N, x, Ht11.dot(u))










def generateMesh(N):
    x = np.zeros(N+1)

    for i in range(0, N+1):
        x[i] = 0.5 * (1 - cos(pi * i * 1 / N))

    h = x[1:] - x[:-1]

    tx = np.concatenate(([0], 0.5 * (x[:-1] + x[1:]), [1]))
    th = tx[1:] - tx[:-1]

    return x, h, tx, th

def generateUK(N, th):
    rows = 4 * (N + 1) + 4 * N

    uK = np.zeros(rows)

    for i in range(0, N + 1):
        uK[i+N*2+N+1] = -1 * th[i];

    return uK

def generateTuK(N):
    rows = 4 * (2 * N + 1)

    return np.zeros(rows)

def generateTE21(N):
    rows = N * N;
    columns = 2 * N * (N - 1);

    tE21 = np.zeros((rows, columns))

    i = 0
    k = 0

    for j in range(0, columns // 2):
        tE21[i][j] = 1
        tE21[i+1][j] = -1

        tE21[k][j+columns/2] = 1
        tE21[k+N][j+columns/2] = -1

        i += 2 if ((j + 1) % (N - 1) == 0) else 1
        k += 1

    return tE21

def generateE21K(N):
    rows = (N + 1) * (N + 1)
    columns = 4 * (N + 1) + 4 * N

    E21K = np.zeros((rows, columns))

    i = 0
    k = rows - 1

    for j in range(0, N + 1):
        E21K[i][j] = 1
        E21K[k][columns / 2 - 1 - j] = -1

        E21K[i][j+columns/2] = -1
        E21K[i][j+columns/2 + 1] = 1

        E21K[k][columns-j-1] = 1
        E21K[k][columns-j-2] = -1

        i += 1
        k -= 1

    i = 0

    for j in range(N + 1, columns // 2 - (N + 1)):
        E21K[i][j] = -1
        E21K[i+N+1][j] = 1

        i += N if ((j - N - 1) % 2 == 0) else 1

    i = N + 1

    for j in range(columns // 2 + N + 2, columns - (N + 2), 2):
        E21K[i][j] = -1
        E21K[i+N][j+1] = 1

        i += N + 1

    return E21K

def generateE21(N):
    rows = (N + 1) * (N + 1)
    columns = 2 * N * (N - 1)

    E21 = np.zeros((rows, columns))

    i = 1
    k = N + 1

    for j in range(0, columns // 2):
        E21[i][j] = -1
        E21[i+N+1][j] = 1

        E21[k][j+columns/2] = 1
        E21[k+1][j+columns/2] = -1

        i += 3 if ((j + 1) % (N - 1) == 0) else 1
        k += 2 if ((j + 1) % N == 0) else 1

    return E21

def generateH1t1(N, h):
    rows = 2 * N * (N - 1)
    columns = 2 * N * (N - 1)

    H1t1 = np.zeros((rows, columns))

    for j in range(0, N):
        velocity = 1 / h[j]

        for i in range(0, N - 1):
            length = 0.5 * (h[i] + h[i+1])

            k = i + (N - 1) * j

            H1t1[k][k] = velocity * length

    for j in range(0, N - 1):
        length = 0.5 * (h[j] + h[j+1])

        for i in range(0, N):
            velocity = 1 / h[i]

            k = rows // 2 + i + N * j;

            H1t1[k][k] = velocity * length

    return  H1t1

def generateHt02(N, th):
    rows = (N + 1) * (N + 1)
    columns = (N + 1) * (N + 1)

    Ht02 = np.zeros((rows, columns))

    for i in range(0, N + 1):
        for j in range(0, N + 1):
            k = j + (N + 1) * i

            Ht02[k][k] = 1 / (th[i] * th[j])

    return Ht02

def generateConvective(N, xi, u, uK, h):
    rows = 2 * N * (N - 1)

    convective = np.zeros(rows)

    uTotal = combineUAndUK(N, u, uK)
    
    offset = (N + 1) * (N + 2)

    for i in range(1, N):
        for j in range(0, N):
            k = (i - 1) + (N - 1) * j

            V1 = uTotal[offset+i+j*(N+2)] + uTotal[offset+(i+1)+j*(N+2)]
            V2 = uTotal[offset+i+(j+1)*(N+2)] + \
                 uTotal[offset+(i+1)+(j+1)*(N+2)]

            convective[k] = - h[i] / (4 * h[j]) * V1 * xi[i+j*(N+1)] \
                            - h[i] / (4 * h[j+1]) * V2 * xi[i+(j+1)*(N+1)]

    for i in range(0, N):
        for j in range(1, N):
            k = N * (N - 1) + i + N * (j - 1)

            U1 = uTotal[i+j*(N+1)] + uTotal[i+(j+1)*(N+1)]
            U2 = uTotal[(i+1)+j*(N+1)] + uTotal[(i+1)+(j+1)*(N+1)]

            convective[k] = h[j] / (4 * h[i]) * U1 * xi[i+j*(N+1)] + \
                            h[j] / (4 * h[i+1]) * U2 * xi[(i+1)+j*(N+1)]

    return convective

def combineUAndUK(N, u, uK):
    uTotal = np.zeros(2 * (N + 1) * (N + 2))

    for i in range(0, N + 2):
        uTotal[(N+1)*(N+2)-1-i] = uK[2*(N+1)+N*2-1-i]

    k = N + 2

    for j in range(0, N):
        for i in range(0, N - 1):
            uTotal[k] = u[i+(N-1)*j] = u[i+(N-1)*j]
            k += 1
        k += 2

    k = (N + 1) * (N + 2) + N + 3

    for j in range(0, N - 1):
        for i in range(0, N):
            uTotal[k] = u[i+N*j+N*(N-1)] = u[i+N*j+N*(N-1)]
            k += 1
        k += 2

    return uTotal

def plotContour(N, x, u):
    rows = N + 1
    columns = N + 1

    stream = np.zeros((rows, columns))

    for i in range(1, N):
        for j in range(1, N):
            k = (j - 1) + (i - 1) * (N - 1)

            stream[N-i,j] = u[k] + stream[N-i+1,j]

    X, Y = np.meshgrid(x, 1 - x)

    plt.contour(X, Y, stream)
    plt.show()

if __name__ == '__main__':
    main()
