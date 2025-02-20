CC=gcc
CFLAGS=-O2 -march=native -g -funroll-loops -fopenmp -ffast-math -Wall
OPTFLAGS=-O3 -march=native -g -funroll-loops -fopenmp -ffast-math -Wall
OBJS_COMMON=kernel.o rdtsc.o

all:	check calibrate measure

check:	$(OBJS_COMMON) driver_check.o
	$(CC) -o $@ $^ -lm
calibrate: $(OBJS_COMMON) driver_calib.o
	$(CC) -o $@ $^ -lm
measure: $(OBJS_COMMON) driver.o
	$(CC) -o $@ $^ -lm

driver_check.o: driver_check.c
	$(CC) $(CFLAGS) -D CHECK -c $< -o $@
driver_calib.o: driver_calib.c
	$(CC) $(CFLAGS) -D CALIB -c $< -o $@
driver.o: driver.c
	$(CC) $(CFLAGS) -c $<

kernel.o: kernel.c
	$(CC) $(OPTFLAGS) -D $(OPT) -c $< -o $@

clean:
	rm -rf $(OBJS_COMMON) driver_check.o driver_calib.o driver.o check calibrate measure
