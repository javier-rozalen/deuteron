! ... MODULE THAT INTERFACES ALL NEEDED DATA FOR INTERPOLATION
      MODULE gausspoints
        INTERFACE 

        SUBROUTINE GAUSS(x1,x2,n,x,w)
          USE precision_definition
          IMPLICIT NONE
          REAL (long), INTENT(IN) :: x1,x2
          INTEGER (ilong), INTENT (IN) :: n
          REAL (long), INTENT(OUT), DIMENSION(:) :: x,w
        END SUBROUTINE GAUSS

        SUBROUTINE LINEAR(x1,x2,n,x,w)
          USE precision_definition
          IMPLICIT NONE
          REAL (long), INTENT(IN) :: x1,x2
          INTEGER (ilong), INTENT (IN) :: n
          REAL (long), INTENT(OUT), DIMENSION(:) :: x,w
        END SUBROUTINE LINEAR

        END INTERFACE
      END MODULE gausspoints
