objects = test.o timing.o ZOCB.o
CC = g++ -mavx2 -maes -O3

ZOCB_Timing : main.cpp ZOCB.cpp timing.cpp
	$(CC) -c ZOCB.cpp -o ZOCB.o
	$(CC) -c timing.cpp ZOCB.o -o timing.o
	$(CC) -c test.cpp ZOCB.o -o test.o
	$(CC) main.cpp test.o timing.o ZOCB.o -o ZOCB_Timing
	make clean


clean:
	rm $(objects)