#EIGEN_DIR := /data/eigen
INCLUDES := -I.. -I../lib
LIBS := -lsfml-graphics -lsfml-window -lsfml-system -lboost_system -lboost_thread
OPTS := -std=c++11 -O2

.DEFAULT_GOAL := all

loader.o: ../lib/loader.h ../lib/loader.cc
	g++ -c ../lib/loader.cc $(INCLUDES) $(LIBS) $(OPTS)

display.o: ../lib/display.h ../lib/display.cc
	g++ -c ../lib/display.cc $(INCLUDES) $(LIBS) $(OPTS)

all: model.cc model.h runner.cc loader.o display.o
	g++ -o app runner.cc model.cc loader.o display.o $(INCLUDES) $(LIBS) $(OPTS) 
