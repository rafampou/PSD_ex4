@echo off
if not  exist cuda.exe (
		nvcc -O3 .\my_select.cu .\euclidean_distance.cu .\vptree_sequential.cu .\CUDABuildvp.cu .\tester.cu -o cuda.exe
	if errorlevel 1 (
		echo "Unsuccessful nvcc"
		pause
		exit
	)
)

echo D=256 > results_D_512.txt

FOR /L %%I IN (1000 ,5000,70000) DO	(

echo %%I

:loop
TIMEOUT /T 1

cuda.exe %%I 512 > temp.txt

IF "%ERRORLEVEL%"=="1" (
	set /a I=I-1000
) ELSE (
	type temp.txt >> results_D_512.txt
)


)

	pause
