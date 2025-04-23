.PHONY: source.o main.o new_run run clean

source.o:
	@clang++ -std=c++11 -c source.cpp

main.o:
	@clang++ -c main.cpp

new_run: source.o main.cpp
	@clang++ -std=c++11 -c main.cpp
	@clang++ main.o source.o -std=c++11 -o run
	@./run
run: source.o main.o
	@clang++ main.o source.o -std=c++11 -o  run
	@./run

clean:
	@rm *.o run
