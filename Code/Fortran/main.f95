program main

double precision, dimension(N) :: A,B,C,D
double precision :: S,E,MFLOPS

! Initialize arrays
do i=1,N
    A(i) = 0.d0
    B(i) = 1.d0
    C(i) = 2.d0
    D(i) = 3.d0
enddo

! Get time stamp
call get_walltime(S)

do j=1,R
    do i=1,N
        ! 3 loads, 1 store
        A(i) = B(i) + C(i) * D(i)
    enddo
    ! Prevent loop interchange
    if (A(2).lt.0) call dummy(A,B,C,D)
enddo

! Get time stamp
call get_walltime(E)

! Compute MFlop/sec rate
MFLOPS = R*N*2.d0/((E-S)*1.d6)

end
