objects = test.o timing.o ZOTR.o
CC = g++ -mavx2 -maes -O3

ZOTR_Timing : main.cpp ZOTR.cpp timing.cpp
	$(CC) -c ZOTR.cpp -o ZOTR.o
	$(CC) -c timing.cpp ZOTR.o -o timing.o
	$(CC) -c test.cpp ZOTR.o -o test.o
	$(CC) main.cpp test.o timing.o ZOTR.o -o ZOTR_Timing
	make clean


clean:
	rm $(objects)