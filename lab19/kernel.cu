#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>

using namespace std;

#define INF 999999
#define MAX_NODES 100
#define WIDTH 800
#define HEIGHT 800
#define MAX_ITERATIONS 10000

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__device__ unsigned char computePixel(float x, float y, float a) {
	float suma = 0;
	float lastX = x;

	for (int i = 0; i < MAX_ITERATIONS; i++)
	{
		float newX = a * lastX * (1 - lastX);
		suma += logf(fabsf(a * (1 - 2 * lastX)));

		if (i > 100)
		{
			if (fabsf(newX - lastX) < 1e-6)
			{
				return (unsigned char)(suma * 255.0 / MAX_ITERATIONS);
			}
		}

		lastX = newX;
	}

	return 0;
}

__global__ void fractal(unsigned char* image, float aMin, float aMax, float bMin, float bMax, float dx, float dy) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float a = aMin + col * dx;
	float b = bMin + row * dy;
	float x = 0.5;
	float y = 0.5;

	unsigned char value = computePixel(x, y, a);
	image[row * WIDTH + col] = value;
}

int main() {
	printf("Krysa Volodymyr: \n");
	printf("19 laba - ");
	unsigned char *image;
	cudaMallocManaged(&image, WIDTH * HEIGHT * sizeof(unsigned char));
	float aMin = 2.4, aMax = 4.0, bMin = 0.1, bMax = 0.9;
	float dx = (aMax - aMin) / WIDTH;
	float dy = (bMax - bMin) / HEIGHT;

	dim3 blocks(WIDTH / 16, HEIGHT / 16);
	dim3 threads(16,16);

	fractal<<<blocks, threads>>>(image, aMin, aMax, bMin, bMax, dx, dy);
	cudaDeviceSynchronize();

	FILE *file = fopen("fractal.png", "wb");
	fprintf(file, "P5\n%d %d\n255\n", WIDTH, HEIGHT);
	//fwrite(image,sizeof(unsigned char), WIDTH * HEIGHT, file);
	fclose(file);
	cudaFree(image);
	return 0;
}