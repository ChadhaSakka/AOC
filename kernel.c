#if defined (OPT1)
#include <math.h>
#include <omp.h>
/*calcul unique de log(x)*/
void kernel(unsigned n, float a[n][n], const float b[n][n], float x) {
    unsigned i, j, k;
    float log_x = log(x);
    for (j = 0; j < n; j++)
        for (i = 0; i < n; i++)
            for (k = 0; k < 6; k++)
                a[i][j] += log_x * b[k][j];
}

#elif defined (OPT2)
#include <math.h>
#include <omp.h>
/*calcul unique de log(x) + inversion de boucles + calcul unique de bkj qui ne dépend pas de i*/
void kernel(unsigned n, float a[n][n], const float b[n][n], float x) {
    unsigned i, j, k;
    float log_x = log(x); 
    for (k = 0; k < 6; k++) {
        for (j = 0; j < n; j++) {
            float bkj = b[k][j];
            #pragma omp simd aligned(a:32)
            for (i = 0; i < n; i++) {
                a[i][j] += log_x * bkj;
            }
        }
    }
}

#elif defined (OPT3)
#include <math.h>
#include <omp.h>

void kernel(unsigned n, float a[n][n], const float b[n][n], float x) {
    unsigned i, j, k;
    float log_x = log(x); // Calcul unique de log(x)
    /* Inversion de boucles */
    for (k = 0; k < 6; k++) {
        #pragma omp simd
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) { 
                a[i][j] += log_x * b[k][j];
            }
        }
    }
}

#elif defined (OPT4)
#include <math.h>
#include <omp.h>
/* calcul unique de log(x) + inversion de boucles + reduction de multiplication + parallelism openmp  + vectorisation simd + alignement */
void kernel(unsigned n, float a[n][n], const float b[n][n], float x) {
    float log_x = log(x);
    #pragma omp parallel for
    for (unsigned j = 0; j < n; j++) {
        float sum = 0.0f;
        for (unsigned k = 0; k < 6; k++) {
            sum += b[k][j];
        }
        sum *= log_x;
        #pragma omp simd aligned(a:32)
        for (unsigned i = 0; i < n; i++) {
            a[i][j] += sum;
        }
    }
}
#else

/* original */

#include <math.h>
void kernel(unsigned n, float a[n][n], const float b[n][n], float x) {
    unsigned i, j, k;
    for (j = 0; j < n; j++)
        for (i = 0; i < n; i++)
            for (k = 0; k < 6; k++)
                a[i][j] += log(x) * b[k][j];
}
#endif
