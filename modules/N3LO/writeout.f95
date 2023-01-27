! ... cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! ... WRITES OUT AN NN INTERACTION IN AND OUT OF THE DIAGONAL
! ... cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      SUBROUTINE WRITEVNN
        USE physical_constants, ONLY : hbc,fact
        USE NN_interaction
        USE mesh

        IMPLICIT NONE

        INTEGER(ilong) :: jj,iju,ija,ik,jk,ll

        open(unit=31,file='diagvNN_J1.dat')
        open(unit=41,file='2d_vNN_J1.dat')
        
129     format('#',3x,'k[fm]',10x,'V_NN(ich=1,8) [fm]')

! ... WRITE OUT NN INTERACTION     
        iju=29
        ija=39
        do jj=Jmax,Jmax
           iju=31
           ija=41
           write(iju,129)
           do ik=1,Nmsh
              write(iju,'(10e15.5)') xk(ik)/hbc,(vNN(ik,ik,jj,ll)/fact,ll=1,8)
              do jk=1,Nmsh
!                 write(*,'(10e15.5)') (xk(ik)/hbc)**2,(xk(jk)/hbc)**2,vNN(ik,jk,1,2)
                 write(ija,'(12e15.5)') xk(ik)/hbc,xk(jk)/hbc &
!                      ,(xk(ik)/hbc)**2,(xk(jk)/hbc)**2 &
                      ,(vNN(ik,jk,jj,ll)/fact,ll=1,6)
              enddo
              write(ija,*)
           enddo
           write(ija,'(/,/)')
           write(iju,'(/,/)')
        enddo
        
      END SUBROUTINE WRITEVNN
