! ... DEFINES PRECISIONS
      MODULE precision_definition
        IMPLICIT NONE
        INTEGER, parameter :: ilong=selected_int_kind(10)
        INTEGER, parameter :: long=selected_real_kind(15,307)
        !INTEGER, parameter :: long=selected_real_kind(15,310)
        !	     INTEGER (ilong), parameter :: Nx=100
        LOGICAL :: rkspace ! -> TRUE IF R SPACE, FALSE IF K SPACE
      END MODULE precision_definition 

! ... PHYSICAL CONSTANTS TO BE USED
      MODULE physical_constants
        USE precision_definition
        REAL (long) :: pi,pi2,hbc,hbc2,hbc3,xmass,htm,deg,fact
        COMPLEX (long) :: zi
      END MODULE physical_constants

! ... MESH IN X AND K SPACE
      MODULE mesh
        USE precision_definition
        INTEGER (ilong) :: Nmsh
        REAL (long), ALLOCATABLE, DIMENSION(:) :: xk,xkw
      END MODULE mesh

! ... NN interaction passed around
      MODULE NN_INTERACTION
        USE precision_definition
        INTEGER (ilong) :: Jmax,Nch,iz1,iz2
        REAL (long) :: cmaxk
        REAL (long), ALLOCATABLE, DIMENSION(:,:,:,:) :: vNN
      END MODULE NN_INTERACTION


      MODULE BESSEL_FUNCTIONS
        INTERFACE
           SUBROUTINE BESS(X,LMAX,IFAIL,ACCUR,XJ)
             USE precision_definition
             IMPLICIT NONE
             INTEGER(ilong), INTENT(IN) :: LMAX
             INTEGER(ilong), INTENT(OUT) :: IFAIL
             REAL(long), INTENT(IN) :: ACCUR,X
             REAL(long), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: XJ(:)
           END SUBROUTINE BESS
        END INTERFACE
      END MODULE BESSEL_FUNCTIONS
