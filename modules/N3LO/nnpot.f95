! ... cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! ... Reads the NN interaction as needed in the program     c
! ... cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      SUBROUTINE NNPOT

        USE precision_definition
        USE physical_constants
        USE NN_interaction
        USE mesh
        USE gausspoints

        IMPLICIT NONE

        REAL(long) :: c,cmax,xxw,c0,c1
        INTEGER :: jc,i
        INTEGER(ilong) :: ix,ik,jk,inn,iju,jj,ll
        REAL(long), ALLOCATABLE, DIMENSION(:) :: xaux,waux

! ... POTENTIAL DEPENDENT SECTION
! ... N3LO POTENTIAL COMMONS
        DOUBLE PRECISION  :: v,xmev,ymev
        LOGICAL :: heform,sing,trip,coup,endep
        CHARACTER*4 :: label
        INTEGER :: kda,kwrite,kread,kpunch
        common /cpot/v(6),xmev,ymev
        common /cstate/jc,heform,sing,trip,coup,endep,label
        common /crdwrtNN/ kread,kwrite,kpunch,kda(9)
        common /cnn/ inn
! ... END OF POTENTIAL DEPENDENT SECTION


! ... START OF SUBROUTINE
        ALLOCATE( xk(Nmsh),xkw(Nmsh) )
        ALLOCATE( xaux(Nmsh),waux(Nmsh) )
        
        write(*,*) 
        write(*,*) 'Loading the NN potential...'

! ... DATA OF THE CDBONN POTENTIAL 
        heform=.false.
        sing=.true.
        trip=.true.
        coup=.true.
        kread=5
        kwrite=6

! ... Mesh of momenta 
! ... Linear set of points
        c0=0.00001d0
        c1=cmaxk
        call linear(c0,c1,Nmsh,xaux,waux)
        xk=xaux
        xkw=waux

        open(11,file="mesh.dat")
        do i=1,Nmsh
            write(11,*) xk(i)/197.3269804d0
        enddo

      


! ... CCCCCCCCCCCCCC  HERE WE CALL THE NN POTENTIAL CCCCCCCCCCCCCCCCCC

!     inn=1  means pp potential,
!     inn=2  means np potential, and
!     inn=3  means nn potential.
      if (iz1.eq.-1 .and. iz2.eq.-1) then 
         inn=3
      elseif(iz1.eq.1 .and. iz2.eq.1) then 
         inn=1
      else
         inn=2
      endif

! ... NN potential 
      do jc=jmax,jmax

         do ik=1,Nmsh

            xmev=xk(ik)
            
            do jk=1,Nmsh

               ymev=xk(jk)
               
               call N3LO

! ... UNCOUPLED STATES
! ... Singlet
               vNN(ik,jk,jc,1)=v(1)
! ... Uncoupled triplet
               vNN(ik,jk,jc,2)=v(2)

! ... COUPLED STATES
! ...  Coupled triplet V--
               vNN(ik,jk,jc,3)=v(4)
! ...  Coupled triplet V++
               vNN(ik,jk,jc,4)=v(3)
! ... 3 four states are the diagonal waves

! ...   Coupled triplet V-+
               vNN(ik,jk,jc,5)=v(6)
! ...   Coupled triplet V+-
               vNN(ik,jk,jc,6)=v(5)

! ... Empty sub boxes
               vNN(ik,jk,jc,7)=0.d0
               vNN(ik,jk,jc,8)=0.d0

               if(jc == 0) then
                  vNN(ik,jk,jc,2)=v(3)
                  vNN(ik,jk,jc,4)=0d0
               endif

            enddo               ! End of loop over jk      
         enddo                  ! End of loop over ik
      enddo                     ! End of loop over jc

      write(*,*) 'Finished loading the NN potential!'

      END SUBROUTINE NNPOT
