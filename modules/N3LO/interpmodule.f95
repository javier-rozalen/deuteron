! ... MODULE THAT INTERFACES ALL NEEDED DATA FOR INTERPOLATION
      MODULE interpolation
        INTERFACE 

        SUBROUTINE LIN_INT(xa,ya,x,y)
          USE precision_definition
          IMPLICIT NONE
          REAL (long), INTENT(IN), DIMENSION(:) :: xa,ya
          REAL (long), INTENT(IN), DIMENSION(:) :: x
          REAL (long), INTENT(OUT), DIMENSION(size(x)) :: y
        END SUBROUTINE LIN_INT

        SUBROUTINE LIN_INT2D(xa,ya,za,x,y,z)
          USE precision_definition
          IMPLICIT NONE
          REAL (long), INTENT(IN), DIMENSION(:) :: xa,ya
          REAL (long), INTENT(IN), DIMENSION(:,:)  :: za
          REAL (long), INTENT(IN), DIMENSION(:) :: x,y
          REAL (long), INTENT(OUT), DIMENSION(size(x),size(y)) :: z
        END SUBROUTINE LIN_INT2D

        INTEGER(ilong) PURE FUNCTION LOCATE(xx,x)
          use precision_definition
          IMPLICIT NONE
          REAL (long), INTENT(IN), DIMENSION(:) :: xx
          REAL (long), INTENT(IN) :: x
        END FUNCTION LOCATE
        
        END INTERFACE
      END MODULE interpolation
