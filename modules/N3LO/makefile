# NAME OF EXECUTABLE
EXEC=deut

# General Files
MAIN=main.o

SUBS=meshgenerator.o interp.o writeout.o nnpot.o deuteron.o

SRC_POT=n3lo.o

# Files which have common.inc included and have to be compilated every time common.inc is changed
COMMONFILES=$(MAIN) $(SUBS)
COMMON=modules.o

###########################################################
# DETERMINE ARCHITECTURE FOR DIFFERENT COMPILERS
ARCH := $(shell arch)

# SHAUN
ifeq ($(ARCH),x86_64) 
COMP	=gfortran
OPTCOMP=-O5
LIBS=-llapack
endif 

# MY MAC
ifeq ($(ARCH),i386) 
COMP	=gfortran
OPTCOMP=-O5 -ftree-vectorize -fbounds-check
LIBS=-llapack -lm -framework Accelerate
endif 

# EUGENE
ifeq ($(ARCH),i86pc) 
COMP	=f95
OPTCOMP=-fpover -O5
LIBS=-lnag -lm
# -L/usr/local/lib -llapack
endif 
#####################################################

$(EXEC).x : $(SUBS) $(SRC_POT) $(MAIN) 
	$(COMP) -o $(EXEC).x $(SUBS) $(SRC_POT) $(LIBS) $(MAIN) 

# This tells how to compile all the files
%.o :	%.f95 
	$(COMP) -c $(OPTCOMP) $<

%.o :	%.f90
	$(COMP) -c $(OPTCOMP) $<

%.o :	%.f
	$(COMP) -c $(OPTCOMP) $<

$(COMMONFILES):	$(COMMON)

clean :
	rm *.o *.mod $(EXEC).x *.dat
