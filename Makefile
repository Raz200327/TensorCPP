.PHONY: clean run new_run main.o source.o
COMPILER = g++

source.o:
	@$(COMPILER) -std=c++14 -c source.cpp

main.o:
	@$(COMPILER) -std=c++14 -c main.cpp

new-run: source.o main.cpp
	@$(COMPILER) -std=c++14 -c main.cpp
	@$(COMPILER) -std=c++14 main.o source.o -o run
	@./run
run: source.o main.o
	@$(COMPILER) -std=c++14 main.o source.o -o run
	@./run

clean:
	@rm *.o run