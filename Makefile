.PHONY: clean run new_run main.o activations.o neural_net.o transformer.o tensor.o
COMPILER = g++

main.o:
	@$(COMPILER) -std=c++14 -c main.cpp

activations.o:
	@$(COMPILER) -std=c++14 -c activations.cpp

neural_net.o:
	@$(COMPILER) -std=c++14 -c neural_net.cpp

transformer.o:
	@$(COMPILER) -std=c++14 -c transformer.cpp

tensor.o:
	@$(COMPILER) -std=c++14 -c tensor.cpp

new-run: main.o activations.o neural_net.o transformer.o tensor.o
	@$(COMPILER) -std=c++14 main.o activations.o neural_net.o transformer.o tensor.o -o run
	@./run
run: activations.o neural_net.o transformer.o tensor.o
	@$(COMPILER) -std=c++14 -c main.cpp
	@$(COMPILER) -std=c++14 main.o activations.o neural_net.o transformer.o tensor.o -o run
	@./run

clean:
	@rm *.o run