#include <stdio.h>
#include "bitmap.h"
#include <complex.h>
#include <math.h>
#include <omp.h>

#define TABLE_SIZE 100000
#define MAX_ITERATIONS 5000
#define PRECISION 1e-5

double complex f(double complex z) {
    return cpow(z, 3) - 1;
}

double complex f_derivative(double complex z) {
    return 3 * cpow(z, 2);
}

int get_convergence_index(double complex z) {
    double complex z_n = z;
    int stevec = 0;
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        double complex z_n1 = z_n - (f(z_n) / f_derivative(z_n));
        stevec++;
        if (cabs(z_n1 - z_n) < PRECISION) {
            double complex r1 = 1;
            double complex r2 = -0.5 + 0.5 * I * sqrt(3);
            double complex r3 = -0.5 - 0.5 * I * sqrt(3);
            if (cabs(z_n1 - r1) < PRECISION) {
                return 80*stevec;
            } else if (cabs(z_n1 - r2) < PRECISION) {
                return 160*stevec;
            } else if (cabs(z_n1 - r3) < PRECISION) {
                return 240*stevec;
            } else {
                return 0;
            }
        }
        z_n = z_n1;
    }
    return 0;
}

void fill_table(unsigned char table[TABLE_SIZE][TABLE_SIZE]) {
    int steviloo = TABLE_SIZE/2;
    int total_iterations = 2 * steviloo * 2 * steviloo;
    int completed_iterations = 0;
    #pragma omp parallel for collapse(2) reduction(+:completed_iterations)
    for (int i = -steviloo; i < steviloo; i++) {
        for (int j = -steviloo; j < steviloo; j++) {
            double complex z = i + j * I;
            int index = get_convergence_index(z);
            table[i + steviloo][j + steviloo] = index;
            completed_iterations++;
            if (completed_iterations % 1000 == 0) {
                printf("\rProgress: %.2f%%", (double)completed_iterations / total_iterations * 100);
                fflush(stdout);
            }
        }
    }
    printf("\n");
}

unsigned char table[TABLE_SIZE][TABLE_SIZE] = {0};
int main() {
    fill_table(table);
    shraniBMP(table, TABLE_SIZE, TABLE_SIZE, "fraktal.bmp");
    return 0;
}
