.PHONY: clean run new_run main.o source.o
COMPILER = g++

source.o:
	@$(COMPILER) -c source.cpp

main.o:
	@$(COMPILER) -c main.cpp

new-run: source.o main.cpp
	@$(COMPILER) -c main.cpp
	@$(COMPILER) main.o source.o -o run
	@./run
run: source.o main.o
	@$(COMPILER) main.o source.o -o run
	@./run

clean:
	@rm *.o run