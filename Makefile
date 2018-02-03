CC=gcc
CXX=g++
RM=rm -f
CPPFLAGS=-g
LDFLAGS=-g 
LDLIBS=-lsndfile -lfftw3

SRCS=main.cpp CPUconvIdentity.cpp CPUconvSimpleReverb.cpp CPUconv.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: convreverb

convreverb: $(OBJS)
	$(CXX) $(LDFLAGS) -o convreverb $(OBJS) $(LDLIBS) 
#	g++ -o main main.cpp -lsndfile

depend: .depend

.depend: $(SRCS)
	$(RM) ./.depend
	$(CXX) $(CPPFLAGS) -MM $^>>./.depend;

clean:
	$(RM) $(OBJS)
	$(RM) convreverb

distclean: clean
	$(RM) *~ .depend

include .depend