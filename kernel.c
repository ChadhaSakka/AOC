#if defined (OPT1)
#include <math.h>
#include <omp.h>
/* Removing of store to load dependency (array ref replaced by scalar) */
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

#include <string.h> // memset

/* ijk -> ikj permutation to make stride 1 the innermost loop */
/*void kernel (unsigned n, float a[n][n], float b[n][n], float c[n][n]) {
   int i, j, k;

   memset (c, 0, n * n * sizeof c[0][0]);

   for (i=0; i<n; i++)
      for (k=0; k<n; k++)
         for (j=0; j<n; j++)
            c[i][j] += a[i][k] * b[k][j];
}*/

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
