objects = test.o timing.o ThetaCB3.o
CC = g++ -mavx2 -maes -O3

ThetaCB3_Timing : main.cpp ThetaCB3.cpp timing.cpp
	$(CC) -c ThetaCB3.cpp -o ThetaCB3.o
	$(CC) -c timing.cpp ThetaCB3.o -o timing.o
	$(CC) -c test.cpp ThetaCB3.o -o test.o
	$(CC) main.cpp test.o timing.o ThetaCB3.o -o ThetaCB3_Timing
	make clean


clean:
	rm $(objects)