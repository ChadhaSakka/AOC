#if defined (OPT1)
#include <math.h>
#include <omp.h>

void kernel(unsigned n, float a[n][n], const float b[n][n], float x) {
    unsigned i, j, k;
    float log_x = log(x); // Calcul unique de log(x)

    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
       #pragma omp simd
            for (k = 0; k < 6; k++) {
                a[i][j] += log_x * b[k][j];
            }
        }
    }
}

#elif defined OPT2
#include <math.h>

void kernel(unsigned n, float a[n][n], const float b[n][n], float x) {
    unsigned i, j, k;
    float log_x = log(x); // Calcul unique de log(x)
    
    /* Amélioration de la localité mémoire en accédant à b[k][j] de manière plus contigue ]*/
    for (k = 0; k < 6; k++) {
        for (j = 0; j < n; j++) {
            for (i = 0; i < n; i++) {
                a[i][j] += log_x * b[k][j];
            }
        }
    }
}


#else

/* original */
#include <math.h>

void kernel(unsigned n, float a[n][n], const float b[n][n], float x) {
    unsigned i, j, k;
    float log_x = log(x); // Calcul unique de log(x)

    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            for (k = 0; k < 6; k++) {
                a[i][j] += log_x * b[k][j];
            }
        }
    }
}
#endif
