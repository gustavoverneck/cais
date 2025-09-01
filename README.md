## CAIS: C++ Artificial Intelligence Solver

**CAIS** is a (work in progress) C++ AI solver. The idea is to build a complete AI solutions platform over time.


## Dependencies 
[OpenCL-Wrapper](https://github.com/gustavoverneck/OpenCL-Wrapper)
[Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)


## Requirements

- **C++ Compiler**: g++ with C++17 support
- **Make**: GNU Make or compatible
- **Cross-Platform**: Works on Windows, Linux, and macOS


# How to Install and Run

1. Open a terminal in the project root directory.
2. Create a build directory and enter it:
	```sh
	mkdir build
	cd build
	```
3. Run CMake to generate build files:
	```sh
	cmake ..
	```
4. Build the project:
	```sh
	cmake --build .
	```
5. Run any example executable from the `build` directory, e.g.:
	```sh
	./example_add.exe
	./example_matmul.exe
	./example_scale.exe
	```