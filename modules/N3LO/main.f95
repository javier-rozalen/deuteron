!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!     DEUTERON FROM GENERAL NN POTENTIALS
! ... SURREY JANUARY 2012
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      include 'modules.f95'

      PROGRAM DEUT

        USE precision_definition
        USE physical_constants
        USE NN_interaction
        USE mesh
        USE deuteron_wf

        IMPLICIT NONE
        REAL(long) :: xmassn,xmassp
        INTEGER(ilong) :: Ns,ik,jk,ll,iju,ija,jj

! ... CONSTANTS
        hbc=197.32968d0
        hbc2=hbc*hbc
        hbc3=hbc*hbc*hbc
        pi=4.d0*datan(1.d0)
        pi2=pi*pi

        xmassn=939.5653d0
        xmassp=938.2720d0


! ... FIX ISOSPIN OF PARTICLES
        iz1=1
        iz2=-1
        xmass=2.d0*xmassp*xmassn/(xmassp+xmassn)
        
! ... FIX MAXIMUM J=1
        Jmax=1

! ... READ MAXIMUM Nmsh
! ... DEFINE MESH PRECISION
        write(*,*) 'Mesh point length and maximum?'
        read(*,*) Nmsh,cmaxk

! ... cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! ... ALLOCATE AND CALL NN INTERACTION        
        Nch=8
        ALLOCATE( vNN(Nmsh,Nmsh,0:Jmax,Nch) )
        call NNPOT

! ... WITH THIS FACTOR, V IS IN fm
! ... TMATRIX IN THIS PRESCRIPTION: T= V + 2/pi*V*G*T
        fact=1d0!4.d0*pi/xmass/(2.d0*pi**2)/hbc
        
        call WRITEVNN

        call DEUTERON

        call WFREAL
        
      END PROGRAM DEUT
