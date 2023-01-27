! ... MODULE THAT INTERFACES ALL NEEDED DATA FOR INTERPOLATION
      MODULE interpolation
        CONTAINS
! ... ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! ... !D LINEAR INTERPOLATION
      PURE SUBROUTINE LIN_INT(xa,ya,x,y)
        USE precision_definition
!        USE interpolation

        IMPLICIT NONE
        REAL (long), INTENT(IN), DIMENSION(:) :: xa,ya
        REAL (long), INTENT(IN), DIMENSION(:) :: x
        REAL (long), INTENT(OUT), DIMENSION(size(x)) :: y
        INTEGER (ilong) :: ii,nf
        INTEGER (ilong), DIMENSION(size(x)) :: klo,khi

        nf=size(x)
        FORALL( ii=1:nf )
           klo(ii)=locate(xa,x(ii))
           khi(ii)=klo(ii)+1
           y(ii) = ya(klo(ii)) + ( ya(khi(ii))-ya(klo(ii)) )/( xa(khi(ii))-xa(klo(ii)) )*( x(ii)-xa(klo(ii)) )
        END FORALL        
      END SUBROUTINE LIN_INT

! ... ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! ... 2D LINEAR INTERPOLATION
      PURE SUBROUTINE LIN_INT2D(xa,ya,za,x,y,z)

        USE precision_definition
!        USE interpolation

        IMPLICIT NONE
        REAL (long), INTENT(IN), DIMENSION(:) :: xa,ya
        REAL (long), INTENT(IN), DIMENSION(:,:)  :: za
        REAL (long), INTENT(IN), DIMENSION(:) :: x,y
        REAL (long), INTENT(OUT), DIMENSION(size(x),size(y)) :: z

        INTEGER (ilong) :: ix,nsx,iy,nsy
        INTEGER (ilong), DIMENSION(size(x),size(y)) :: kxlo,kxhi,kylo,kyhi
        
        REAL (long), DIMENSION(size(x),size(y)) :: z1,z2,z3,z4,t,u

        nsx=size(x)
        nsy=size(y)

        FORALL( iy=1:nsy, ix=1:nsx) 
           kylo(ix,iy)=locate(ya,y(iy))
           kyhi(ix,iy)=kylo(ix,iy)+1
           u(ix,iy)=(y(iy)-ya(kylo(ix,iy)))/(ya(kyhi(ix,iy))-ya(kylo(ix,iy)))
           
           kxlo(ix,iy)=locate(xa,x(ix))
           kxhi(ix,iy)=kxlo(ix,iy)+1
           t(ix,iy)=(x(ix)-xa(kxlo(ix,iy)))/(xa(kxhi(ix,iy))-xa(kxlo(ix,iy)))

           z1(ix,iy)=za(kxlo(ix,iy),kylo(ix,iy))
           z2(ix,iy)=za(kxhi(ix,iy),kylo(ix,iy))
           z3(ix,iy)=za(kxhi(ix,iy),kyhi(ix,iy))
           z4(ix,iy)=za(kxlo(ix,iy),kyhi(ix,iy))

           z(ix,iy)=(1.d0-t(ix,iy))*(1.d0-u(ix,iy))*z1(ix,iy) &
                + t(ix,iy)*(1.d0-u(ix,iy))*z2(ix,iy) + t(ix,iy)*u(ix,iy)*z3(ix,iy) &
                + (1.d0-t(ix,iy))*u(ix,iy)*z4(ix,iy)
        END FORALL

      END SUBROUTINE LIN_INT2D

! cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! ... Given an array xx(1:N), and given a value x, returns a value j such that x is between
! ... xx(j) and xx(j + 1). xx must be monotonic, either increasing or decreasing. j = 0 or
! ... j = N-1 is returned to indicate that x is out of range.
! cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      INTEGER(ilong) PURE FUNCTION LOCATE(xx,x)
        use precision_definition
        IMPLICIT NONE
        REAL (long), INTENT(IN), DIMENSION(:) :: xx
        REAL (long), INTENT(IN) :: x
        INTEGER (ilong) :: n,jl,jm,ju
        LOGICAL :: ascnd

        n=size(xx)

        if( x < xx(1) ) then
           locate=1
        elseif( x > xx(n) ) then
           locate=n-1
        else
           ascnd = (xx(n) >= xx(1) ) ! True if ascending, false if descending
           jl=0
           ju=n+1
           do while (ju - jl > 1) 
              jm=(ju+jl)/2
              if (ascnd .eqv. (x >= xx(jm))) then
                 jl=jm
              else
                 ju=jm
              end if
              locate=jl
           end do
        end if

      END FUNCTION LOCATE
      END MODULE interpolation
