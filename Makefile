all: main

SIMGRID_INSTALL_PATH = /usr/local
CC = g++ --std=c++17
CFLAGS = -g -O2 -Wall -Wextra

main: main.o
	$(CC) -L$(SIMGRID_INSTALL_PATH)/lib/ $(CFLAGS) $^ -lsimgrid -o $@

main.o: main.cpp
	$(CC) -I$(SIMGRID_INSTALL_PATH)/include  $(CFLAGS) -c -o  $@ $<

clean:
	rm -f main *.o *~
.PHONY: clean
