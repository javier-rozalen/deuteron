! ... DEUTERON WAVE FUNCTION IN R AND K SPACE
      MODULE DEUTERON_WF

        USE precision_definition
        REAL (long), ALLOCATABLE, DIMENSION(:,:) :: wfk,wfr

        CONTAINS

! ... cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! ... SOLVES THE DEUTERON IN K-SPACE
! ... cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      SUBROUTINE DEUTERON

        USE precision_definition
        USE physical_constants
        USE NN_interaction
        USE mesh

        IMPLICIT NONE

! ... cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! STUFF FOR MATRIX INVERSION ROUTINE
        REAL(long), ALLOCATABLE, DIMENSION(:) :: wi,wr,xwrk
        REAL(long), ALLOCATABLE, DIMENSION(:,:) :: xvec
        CHARACTER*1 :: job,sort
        INTEGER :: ld,sdim,ldvs,lwork,info
        LOGICAL :: ext
        LOGICAL, ALLOCATABLE, DIMENSION(:) :: bwrk
        EXTERNAL :: sel
! ... cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       
        INTEGER(ilong) :: Nq,jj,ich,ik,jk,i1,i2,i3,i4,iq,ii
        INTEGER(ilong) :: iflag,ili,ix,jij
        REAL(long) :: xnorm1,xnorm2,xnorm,ffac,xni,xtst,xnj,xmom
        REAL(long) :: xekin1,xekin2,ekin,epot
        REAL(long), ALLOCATABLE, DIMENSION(:,:) :: vaux,wei,fac

        open(unit=20,file='wfk.dat')
        open(unit=19,file='momdis.dat')
        open(unit=18,file='energy.dat')

        if(Jmax < 1) then
           write(*,*) 'Increase Jmax!',Jmax
           stop
        endif

        if(iz1*iz2 /= -1) then
           write(*,*) 'Wrong isospin for the deuteron!',iz1,iz2
           stop
        endif

        write(*,*) 'MESH DONE'

        Nq=2*Nmsh
        ALLOCATE( vaux(Nq,Nq) )

        jj=1
        i1=3
        i2=5
        i3=6
        i4=4          

! ... BUILD MATRIX
        ALLOCATE( wei(Nmsh,Nmsh),fac(Nmsh,Nmsh) )
        wei=0d0

        FORALL(ik=1:Nmsh, jk=1:Nmsh)
           wei(ik,jk)=sqrt( xkw(ik)*xkw(jk) )
           fac(ik,jk)=wei(ik,jk)*xk(ik)*xk(jk)
           vaux(ik,jk)=vNN(ik,jk,jj,i1)*fac(ik,jk)
           vaux(ik,Nmsh+jk)=vNN(ik,jk,jj,i2)*fac(ik,jk)
           vaux(Nmsh+ik,jk)=vNN(ik,jk,jj,i3)*fac(ik,jk)
           vaux(Nmsh+ik,Nmsh+jk)=vNN(ik,jk,jj,i4)*fac(ik,jk)
        END FORALL

        FORALL(ik=1:Nmsh)
           vaux(ik,ik)=xk(ik)**2/xmass+vaux(ik,ik)
           vaux(ik+Nmsh,ik+Nmsh)=xk(ik)**2/xmass+vaux(ik+Nmsh,ik+Nmsh)
        END FORALL

        write(*,*) 'MATRIX DONE'     

! ... DIAGONALIZATION ROUTINE
        lwork=10000
        ALLOCATE( wr(Nq),wi(Nq) )
        ALLOCATE( xwrk(lwork) )
        ALLOCATE( xvec(Nq,Nq) )
        ALLOCATE( bwrk(Nq) )
        job='V'
        sort='S'
        ld=Nq
        ldvs=ld

        CALL DGEES(job,sort,sel,Nq,vaux,ld,sdim,wr,wi &
             ,xvec,ldvs,xwrk,lwork,bwrk,info)

        write(*,*) 'Binding energy [Mev]:'
100     format(i4,100(e20.8))
        do ii=1,Nq
! THIS WILL SHOW ALL POTENTIAL EIGENVALUES
           if(wr(ii) < 0) write(*,100) ii,wr(ii)
        enddo

        ALLOCATE( wfk(Nmsh,2) )

! CORRECT OVERALL SIGN
        xnorm1=1d0
        xnorm2=1d0
        if(xvec(1,1) < 0) xnorm1=-1d0
        if(xvec(Nmsh+1,1) < 0) xnorm2=-1d0
        FORALL( ik=1:Nmsh )
           wfk(ik,1)=xvec(ik,1)/sqrt(xkw(ik))*xnorm1
           wfk(ik,2)=xvec(ik+Nmsh,1)/sqrt(xkw(ik))*xnorm2
        END FORALL

! ... EIGENVALUES 
        xnorm1=0.d0
        xnorm2=0.d0
        
        xekin1=0.d0
        xekin2=0.d0

120     format('# wf=k*wf',/,'#',4x,'k [fm]',10x,'3s1 wf',10x,'3d1 wf')
        write(20,120)
119     format('#',4x,'k [fm]',10x,'n(k)')
        write(19,119)       

        do iq=1,Nmsh
           ffac=4.d0*pi*hbc/xk(iq)
           
           write(20,'(3e16.8)') xk(iq)/hbc,wfk(iq,1)*ffac &
                ,wfk(iq,2)*ffac

           xmom=( wfk(iq,1)**2+wfk(iq,2)**2) /xk(iq)**2*hbc**3
           write(19,'(3e16.8)') xk(iq)/hbc,xmom

           xnorm1=xnorm1+wfk(iq,1)**2*xkw(iq)
           xnorm2=xnorm2+wfk(iq,2)**2*xkw(iq)

           xekin1=xekin1+wfk(iq,1)**2*xkw(iq)*(xk(iq)**2/xmass)
           xekin2=xekin2+wfk(iq,2)**2*xkw(iq)*(xk(iq)**2/xmass)
        enddo
      
111     format(/,'Norm: total',11x,'s-state,',8x,'d-state',/,4x,3e16.8)
        xnorm=xnorm1+xnorm2
        write(*,111) xnorm,xnorm1,xnorm2
      
112     format(/,'KE:  total',11x,'s-state,',8x,'d-state',/,4x,3e16.8)
        ekin=xekin1+xekin2
        write(*,112) ekin,xekin1,xekin2

        epot=wr(1)-ekin
102     format('#',4x,'Binding E [MeV]',6x,'Kinetic E [MeV]',5x,'Potential E [MeV]' &
             ,4x,'S state prob',7x,'B state prob',8x,'Kinetic S [MeV]',6x,'Kinetic D [Mev]')
        write(18,102) 
101     format(100(e20.8))
        write(18,101) wr(1),ekin,epot,xnorm1,xnorm2,xekin1,xekin2

        close(20)
        close(19)
        close(18)

        DEALLOCATE( wi,wr,xwrk,xvec,bwrk )
        DEALLOCATE( vaux,wei,fac )

      END SUBROUTINE DEUTERON
      

!ccccccccccccccccccccccccccccccccccccccccccccccccc
! ... FOURIER-BESSEL TRANSFORM OF WAVE FUNCTIONS 
! ... TO REAL SPACE
!ccccccccccccccccccccccccccccccccccccccccccccccccc
      SUBROUTINE WFREAL
        USE precision_definition
        USE physical_constants
        USE NN_interaction
        USE mesh
        USE interpolation
        USE gausspoints
!        USE deuteron_wf
        ! ... THIS IS WHERE BESSEL FUNCTIONS ARE!
        USE BESSEL_FUNCTIONS
        
        IMPLICIT NONE

        INTEGER(ilong) :: Nqr,if1,jj,ir,iq,Nr
        REAL(long) :: acc,ri,rf,xfac,x0,xf,rq,wws,wwd,wfrs,wfrd
        REAL(long) :: xnorms,xnormd,xnorm,rms,quad,rr,xns,xnd,qqu,xmus,xmud,xmu

        REAL(long), ALLOCATABLE, DIMENSION(:) :: xbes
        REAL(long), ALLOCATABLE, DIMENSION(:) :: r,wwr,wfk1,wfk2
        REAL(long), ALLOCATABLE, DIMENSION(:) :: qint,wqint

        open(unit=21,file='wfr.dat')
        open(unit=22,file='r_density.dat')
        open(unit=23,file='obs.dat')

        acc=1e-15

        Nr=200
        ALLOCATE ( r(Nr),wwr(Nr) ) 

        ri=0d0
        rf=40d0
        call GAUSS(ri,rf,Nr,r,wwr)
        
        xfac=sqrt(2d0/pi/hbc3)
        
        Nqr=4000
        ALLOCATE( wfk1(Nqr),wfk2(Nqr),qint(Nqr),wqint(Nqr) )
        x0=0.1d0
        xf=min(3000d0,cmaxk)
        call LINEAR(x0,xf,Nqr,qint,wqint)

        call LIN_INT(xk,wfk(:,1),qint,wfk1)
        call LIN_INT(xk,wfk(:,2),qint,wfk2)

121     format('# wf=wfr',/,'#',4x,'r [fm]',10x,'3s1 wf',10x,'3d1 wf')
        write(21,121)

        ALLOCATE( wfr(Nr,2) )

        xnorms=0.d0
        xnormd=0.d0
        rms=0.d0
        quad=0.d0
        jj=2
! ... LOOP OVER POSITIONS
        do ir=1,Nr
           rr=r(ir)

! ... INTEGRAL OVER MOMENTA
           wfrs=0d0
           wfrd=0d0
           do iq=1,Nqr
! ... CALL BESSEL FUNCTION
              rq=rr*qint(iq)/hbc
              call bess(rq,jj,if1,acc,xbes)
              if(if1 == 1) stop 'error in bessel function routine'

              wws=xbes(1)*qint(iq)*wfk1(iq)
              wwd=xbes(3)*qint(iq)*wfk2(iq)

              wfrs=wfrs+wws*wqint(iq)
              wfrd=wfrd+wwd*wqint(iq)
           enddo ! Loop over momenta

           wfr(ir,1)=wfrs*xfac
           wfr(ir,2)=wfrd*xfac

         write(21,'(3e16.8)') rr,rr*wfr(ir,1),rr*wfr(ir,2)
!         write(21,'(3e16.8)') rr,wfr(ir,1),wfr(ir,2)

! ... RMS FROM WAVE FUNCTIO
         xns=wfr(ir,1)**2*rr**2
         xnd=wfr(ir,2)**2*rr**2
         xnorms=xnorms+xns*wwr(ir)
         xnormd=xnormd+xnd*wwr(ir)

         write(22,'(3e16.8)') rr,xns+xnd

         rms=rms+(xns+xnd)*rr**2*wwr(ir)
         qqu=wfr(ir,2)*(wfr(ir,1)-wfr(ir,2)/sqrt(8.d0))*rr**4
         quad=quad+qqu*wwr(ir)
      enddo ! Loop over position
      xnorm=xnorms+xnormd

 111  format(/,'Norm: total',11x,'s-state,',8x,'d-state',/,4x,4e16.8)
      write(*,111) xnorm,xnorms,xnormd

 211  format(/,'RMS=',2e16.8,' fm')
      write(*,211) dsqrt(rms)/2.d0!,dsqrt(rms)/4d0

 311  format('QUAD=',e16.8,' fm2')
      write(*,311) quad/dsqrt(50.d0)

! ... MAGNETIC MOMENT
      xmus=0.879d0 ! in nuclear magneton
      xmud=0.310d0 ! in nuclear magneton
      xmu=xmus*xnorms + xmud*xnormd

 411  format('MU=',e16.8,' Nuc. Mag.')
      write(*,411) xmu
      
      DEALLOCATE( r,wwr,wfk1,wfk2,qint,wqint,xbes )

402   format('#',6x,'RMS [fm]',10x,'Quadrupole [fm2]',4x,'Magnetic Moment [Nuc Mag]')
      write(23,402)
      write(23,'(3e20.8)') dsqrt(rms)/2d0,quad/dsqrt(50d0),xmu

      close(21)
      close(22)
      close(23)

      END SUBROUTINE WFREAL


 END MODULE DEUTERON_WF

!ccccccccccccccccccccccccccccccccccccccccccccccccc
! ... FUNCTION TO CHOOSE REAL EIGENVALUES
!ccccccccccccccccccccccccccccccccccccccccccccccccc
      LOGICAL FUNCTION SEL(x1,x2)
        double precision x1,x2

        if( x2.eq.0d0 ) then
           if(x1 .lt. 0d0 .and. x1.gt.-10d0) then
              SEL=.TRUE.
           else
              SEL=.FALSE.
           endif
             
        else
           SEL=.FALSE.
        endif

        RETURN
      END FUNCTION SEL


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! ... COMPUTES THE SPHERICAL BESSEL FUNCTION AT X FROM   c
! ... L=0 to L=LMAX AND STORES IT IN XJ(1:LMAX+1)        c
! ... IFAIL SHOULD BE 0, OTHERWISE AN ERROR HAS OCCURRED  c
! ... ACCUR DETERMINES THE ACCURACY AT WHICH THE FUNCTION 
! ... IS COMPUTED
! ... TAKEN FROM:
! ... Computer Physics Communications 21, 297 (1982).
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      SUBROUTINE BESS(X,LMAX,IFAIL,ACCUR,XJ)

        USE precision_definition
        IMPLICIT NONE
        INTEGER(ilong), INTENT(IN) :: LMAX
        INTEGER(ilong), INTENT(OUT) :: IFAIL
        REAL(long), INTENT(IN) :: ACCUR,X
        REAL(long), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: XJ(:)
        
        REAL(long) :: zero,one,acc,xo,w,f,fp,b,d,del,xend,xp2,pl,xi
        INTEGER(ilong) :: l,lp

        ALLOCATE( XJ(20) )
      
        zero=0.d0
        one=1.d0
        ifail=0
        acc=accur
      
        if( abs(x) < accur) then
           ifail=1
           write(*,*) 'x in Bessel is too small:',x
           stop
        endif

        xi=one/x
        w=xi+xi
        f=one
        fp=real(lmax+1,long)*xi
        b=fp+fp+xi

        d=one/b
        del=-d
        fp=fp+del
        xend=b+2000000.d0*w

! ... INITIAL BLOCK
1       b=b+w
        d=one/(b-d)
        del=del*(b*d-one)
        fp=fp+del
     
        if(d < zero) f=-f

        if(b > xend) then
           ifail=1
        endif
        if(abs(del) > abs(fp)*acc) go to 1

        fp=fp*f
        if(lmax==0) go to 3
        xj(lmax+1)=f
        xp2=fp

! ... DOWNWARD RECURSION TO L=0
        pl=dble(lmax)*xi
        l=lmax
        do 2 lp=1,lmax
           xj(l)=pl*xj(l+1) + xp2
           fp=pl*xj(l) - xj(l+1)
           xp2=fp
           pl=pl-xi
           l=l-1
2       enddo
        f=xj(1)

! ... SOLVE FOR L=0
3       w=xi/sqrt(fp*fp + f*f)
        xj(1)=w*f
        if(lmax.ne.0) then
           do 4 l=1,lmax
              xj(l+1)=w*xj(l+1)
4          enddo
        endif

      END SUBROUTINE BESS
