@echo off
if not  exist cuda.exe (
		nvcc -O3 .\my_select.cu .\euclidean_distance.cu .\vptree_sequential.cu .\CUDABuildvp.cu .\tester.cu -o cuda.exe
	if errorlevel 1 (
		echo "Unsuccessful nvcc"
		pause
		exit
	)
)