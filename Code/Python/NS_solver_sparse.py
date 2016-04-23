#!/usr/local/bin/python3

import sys
import numpy as np
from math import cos
from math import pi
from math import sqrt
from math import floor
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg
from scipy import sparse


def main():
    # Set numpy print options
    np.set_printoptions(precision=4, linewidth=178)

    # Make matplotlib plot colorless contour plots
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

    # Verify the correct amount of command line arguments
    if (len(sys.argv) != 5):
        print('Wrong number of arguments\nUsage:', sys.argv[0], 'dt N')
        exit()

    # Fetch command line arguments
    tol = float(sys.argv[1]) # Stopping criterion
    Re = float(sys.argv[2]) # Reynolds number
    dt = float(sys.argv[3]) # Timestep
    N = int(sys.argv[4]) # Number of planes in the x and y-direction

    # Variable name convention:
    #   - 't' or 'T' means 'tilde', an indicator for the outer-oriented grid
    #   - 'K' means 'known'
    #
    # Examples:
    #   - tE21 means $\tilde{E}^{(2,1)}$
    #   - E21K means $E^{(2,1)}_{\text{known}}$
    #   - tE21 means $\tilde{E}^{(2,1)}$
    #   - H1t1 means $H^{(1,\tilde{1})}$

    # Generate mesh
    x, h, tx, th = generateMesh(N)

    # Generate incidence matrices and hodge matrices
    tE21 = generateTE21(N)
    E10 = -np.transpose(tE21)
    E21K = generateE21K(N)
    E21 = generateE21(N)
    tE10 = np.transpose(E21)
    H1t1 = generateH1t1(N, th)
    Ht11 = np.linalg.inv(H1t1)
    Ht02 = generateHt02(N, h)
    
    # Generate velocity vectors

    # The vector u contains the *inner-oriented* circulations as unknowns and
    # boh u and v (that is, the horizontal and vertical circulations) will be
    # stored in u. The vector u only contains the *true* unknowns, not the
    # circulations that are prescribed by the boundary conditions 
    u = np.zeros(2 * N * (N - 1))

    # The vector uK contains the circulations that are prescribed by the
    # boundary conditionss
    uK = generateUK(N, h)
    uPres = H1t1.dot(tE10).dot(Ht02).dot(E21K).dot(uK)
    
    # Generate constants
    C0 = Ht02.dot(E21)
    C1 = tE21.dot(Ht11)
    C2 = H1t1.dot(tE10).dot(C0)
    C3 = uPres / Re

    # Generate the pressure matrix
    A = C1.dot(E10)

    # Compute the LU factorization of the pressure matrix A
    LU = scipy.linalg.lu_factor(A)
    
    diff = 1
    iteration = 0

    # Convert matrices to a sparse storage scheme
    C0 = sparse.csr_matrix(C0)
    C1 = sparse.csr_matrix(C1)
    C2 = sparse.csr_matrix(C2)
    E10 = sparse.csr_matrix(E10)
    
    # Simulation loop
    while (diff > tol):
        iteration += 1

        # Compute vorticity
        xi = C0.dot(u)

        # Generate the convective term
        convective = generateConvective(N, xi, u, uK, h)
        
        # Update constant C4
        C4 = C2.dot(u / Re) + C3 + convective

        # Compute the right-hand side of the system
        rhs = C1.dot(u / dt - C4)

        # Solve the system for the pressure P
        P = scipy.linalg.lu_solve(LU, rhs)

        # Store the previous circulations in uOld
        uOld = np.copy(u)

        # Compute the new circulations u
        u -= dt * (E10.dot(P) + C4)

        # Check for convergence and check the rate of mass creation (must be
        # in the order of machine precision)
        if (iteration % 100 == 0):
            maxdiv = max(C1.dot(u))

            diff = max(np.abs(u - uOld)) / dt
            print('Iteration', iteration, '\tdiff =',
                  '{0:0=6.5f}'.format(diff), '\trate of mass creation =',
                  '{0:0=6.5e}'.format(maxdiv), end="\r")

    print('\nConverged after', iteration, 'iterations.')
    
    # Plot the stream function, vorticity, and pressure
    plotStreamFunctionContour(N, tx, Ht11.dot(u))
    plotVorticityContour(N, tx, xi)
    plotPressureContour(N, x, h, u, uK, P)


def generateMesh(N):
    # Generates the coordinates of the nodel points and the edge lengths of
    # the inner and outer-oriented grid
    #
    # Arguments:
    #   N   - Number of planes in the x and y-direction
    #
    # Returns:
    #   x   - Vector containing the coordinates of the nodal points of the
    #         inner-oriented grid (including the endpoints 0 and 1)
    #   h   - Vector containing the edge lengths of the inner-oriented grid
    #   tx  - Vector containing the coordinates of the nodal points of the
    #         outer-oriented grid (including the endpoints 0 and 1)
    #   th  - Vector containing the edge lenghts of the outer-oriented grid

    tx = np.zeros(N+1)

    for i in range(0, N+1):
        tx[i] = 0.5 * (1 - cos(pi * i * 1 / N))

    th = tx[1:] - tx[:-1]

    x = np.concatenate(([0], 0.5 * (tx[:-1] + tx[1:]), [1]))
    h = x[1:] - x[:-1]

    return x, h, tx, th


def generateUK(N, h):
    # Generates the vorticities on the boundary based on the boundary
    # condition. The boundary condition is hard-coded into this function
    #
    # Arguments:
    #   N   - Number of planes in the x and y-direction
    #   h   - Vector containing the edge lengths of the inner-oriented grid
    #
    # Returns:
    #   uK  - Vector containing the circulations that are prescribed by the
    #         boundary conditionss and zeros elsewhere

    rows = 4 * (N + 1) + 4 * N

    uK = np.zeros(rows)

    for i in range(0, N + 1):
        uK[i+N*2+N+1] = -1 * h[i];

    return uK


def generateTE21(N):
    # Generates the incidence matrix tE21
    #
    # Arguments:
    #   N       - Number of planes in the x and y-direction
    #
    # Returns:
    #   tE21    - Incidence matrix

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
    # Generates the incidence matrix E21K
    #
    # Arguments:
    #   N       - Number of planes in the x and y-direction
    #
    # Returns:
    #   E21K    - Incidence matrix

    rows = (N + 1) * (N + 1)
    columns = 4 * (N + 1) + 4 * N

    E21K = np.zeros((rows, columns))

    # Generate the left-half columns of the matrix
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

    # Generate the right-half columns of the matrix
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
    # Generates the incidence matrix E21
    #
    # Arguments:
    #   N       - Number of planes in the x and y-direction
    #
    # Returns:
    #   E21     - Incidence matrix

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


def generateH1t1(N, th):
    # Generates the hodge matrix H1t1
    #
    # Arguments:
    #   N       - Number of planes in the x and y-direction
    #   th      - Vector containing the edge lenghts of the outer-oriented
    #             grid
    #
    # Returns:
    #   H1t1    - Hodge matrix

    rows = 2 * N * (N - 1)
    columns = 2 * N * (N - 1)

    H1t1 = np.zeros((rows, columns))

    for j in range(0, N):
        velocity = 1 / th[j]

        for i in range(0, N - 1):
            length = 0.5 * (th[i] + th[i+1])

            k = i + (N - 1) * j

            H1t1[k][k] = velocity * length

    for j in range(0, N - 1):
        length = 0.5 * (th[j] + th[j+1])

        for i in range(0, N):
            velocity = 1 / th[i]

            k = rows // 2 + i + N * j;

            H1t1[k][k] = velocity * length

    return H1t1


def generateHt02(N, h):
    # Generates the hodge matrix Ht02
    #
    # Arguments:
    #   N       - Number of planes in the x and y-direction
    #   h       - Vector containing the edge lengths of the inner-oriented
    #             grid
    #
    # Returns:
    #   Ht02    - Hodge matrix

    rows = (N + 1) * (N + 1)
    columns = (N + 1) * (N + 1)

    Ht02 = np.zeros((rows, columns))

    for i in range(0, N + 1):
        for j in range(0, N + 1):
            k = j + (N + 1) * i

            Ht02[k][k] = 1 / (h[i] * h[j])

    return Ht02


def generateConvective(N, xi, u, uK, h):
    # Generates the convective term
    #
    # Arguments:
    #   N           - Number of planes in the x and y-direction
    #   xi          - Vector containing vorticity
    #   u           - Vector containing the circulation excluding the boundary
    #   uK          - Vector containing the circulations that are prescribed
    #                 by the boundary conditionss and zeros elsewhere
    #   h           - Vector containing the edge lengths of the inner-oriented
    #                 grid
    #
    # Returns:
    #   convective  - vector containing the convective term

    rows = 2 * N * (N - 1)

    convective = np.zeros(rows)

    uTotal = combineUAndUK(N, u, uK)
    
    # Since u contains first the  horizontal and then the vertical elements,
    # it makes sense to define an offset that separtes the horizontal and
    # vertical elements 
    offset = (N + 1) * (N + 2)

    # Generate the top-half rows which are based on the vertical line segments
    for i in range(1, N):
        for j in range(0, N):
            k = (i - 1) + (N - 1) * j

            V1 = uTotal[offset+i+j*(N+2)] + uTotal[offset+(i+1)+j*(N+2)]
            V2 = uTotal[offset+i+(j+1)*(N+2)] + \
                 uTotal[offset+(i+1)+(j+1)*(N+2)]

            convective[k] = - h[i] / (4 * h[j]) * V1 * xi[i+j*(N+1)] \
                            - h[i] / (4 * h[j+1]) * V2 * xi[i+(j+1)*(N+1)]

    # Generate the bottom-half rows which are based on the horizontal line
    # segments
    for i in range(0, N):
        for j in range(1, N):
            k = N * (N - 1) + i + N * (j - 1)

            U1 = uTotal[i+j*(N+1)] + uTotal[i+(j+1)*(N+1)]
            U2 = uTotal[(i+1)+j*(N+1)] + uTotal[(i+1)+(j+1)*(N+1)]

            convective[k] = h[j] / (4 * h[i]) * U1 * xi[i+j*(N+1)] + \
                            h[j] / (4 * h[i+1]) * U2 * xi[(i+1)+j*(N+1)]

    return convective


def combineUAndUK(N, u, uK):
    # Combines the known circulation on the boundary with the computed
    # circulation in the center of the domain
    #
    # Arguments:
    #   N       - Number of planes in the x and y-direction
    #   u       - Vector containing the circulation excluding the boundary
    #   uK      - Vector containing the circulations that are prescribed by
    #             the boundary conditionss and zeros elsewhere
    #
    # Returns:
    #   uTotal  - vector containing vorticity in the domain and on its
    #             boundary

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


def plotStreamFunctionContour(N, tx, u):
    # Generates a contour plot of the stream function
    #
    # Arguments:
    #   N   - Number of planes in the x and y-direction
    #   tx  - Vector containing the coordinates of the nodal points of the
    #         outer-oriented grid (including the endpoints 0 and 1)
    #   u   - Vector containing the circulation excluding the boundary

    rows = N + 1
    columns = N + 1

    stream = np.zeros((rows, columns))

    for i in range(1, N):
        for j in range(1, N):
            k = (j - 1) + (i - 1) * (N - 1)

            stream[N-i,j] = u[k] + stream[N-i+1,j]

    X, Y = np.meshgrid(tx, 1 - tx)
    
    levels = [-1.5e-3, -1e-3, -5e-4, -2.5e-4, -1e-4, -5e-5, -1e-5, -1e-6, 0,
              1e-10, 1e-5, 1e-4, 1e-2, 3e-2, 5e-2, 7e-2, 9e-2, 0.1, 0.11,
              0.115, 0.1175]
    
    plt.contour(X, Y, stream, levels=levels, colors='k')
    plt.show()


def plotVorticityContour(N, tx, xi):
    # Generates a contour plot of the vorticity
    #
    # Arguments:
    #   N   - Number of planes in the x and y-direction
    #   tx  - Vector containing the coordinates of the nodal points of the
    #         outer-oriented grid (including the endpoints 0 and 1)
    #   xi  - Vector containing vorticity

    rows = N + 1
    columns = N + 1
    
    vorticity = np.zeros((rows, columns))
    
    for i in range(0, N + 1):
        for j in range(0, N + 1):
            k = j + i * (N + 1)
            
            vorticity[N-i,j] = xi[k]
    
    X, Y = np.meshgrid(tx, 1 - tx)
    
    levels = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    
    plt.contour(X, Y, vorticity, levels=levels, colors='k')
    plt.show()


def plotPressureContour(N, x, h, u, uK, P):
    # Generates a contour plot of the static pressure
    #
    # Arguments:
    #   N   - Number of planes in the x and y-direction
    #   x   - Vector containing the coordinates of the nodal points of the
    #         inner-oriented grid (including the endpoints 0 and 1)
    #   h   - Vector containing the edge lengths of the inner-oriented grid
    #   u   - Vector containing the circulation excluding the boundary
    #   uK  - Vector containing the circulations that are prescribed by the
    #         boundary conditionss and zeros elsewhere
    #   P   - Vector containing the total pressure

    rows = N
    columns = N
        
    uTotal = combineUAndUK(N, u, uK)
    
    pressure = np.zeros((rows, columns))
    
    # Subtract half the magnitude of the velocity squared from the total
    # pressure term to obtain the static pressure
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            k = (j - 1) + (i - 1) * N
            l = j + i * (N + 1)
            m = j + i * (N + 2) + (N + 1) * (N + 2)
            
            uAverage = 0.5 * (uTotal[l] / h[j] + uTotal[l-1] / h[j-1])
            vAverage = 0.5 * (uTotal[m] / h[i] + uTotal[m-(N+2)] / h[i-1])
            
            velocity = uAverage ** 2 + vAverage ** 2
            
            pressure[N-i,j-1] = P[k] - 0.5 * sqrt(velocity) ** 2
    
    X, Y = np.meshgrid(x[1:(N+1)], 1 - x[1:(N+1)])
    
    # Scale the static pressure such that the static pressure is zero in the
    # center of the domain
    levels = [-0.002, 0, 0.02, 0.05, 0.07, 0.09, 0.11, 0.12, 0.17, 0.3]
        
    pressure = pressure - pressure[floor(N/2),floor(N/2)]
    
    plt.contour(X, Y, pressure, levels=levels, colors='k')
    plt.show()


if __name__ == '__main__':
    main()
