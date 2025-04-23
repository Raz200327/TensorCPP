.PHONY: source.o main.o new_run run clean

source.o:
	@clang++ -c source.cpp

main.o:
	@clang++ -c main.cpp

new_run: source.o main.cpp
	@clang++ -c main.cpp
	@clang++ main.o source.o -o run
	@./run
run: source.o main.o
	@clang++ main.o source.o -o run
	@./run

clean:
	@rm *.o run
