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
/* VERSION OPTIMALE : calcul unique de log(x) + inversion de boucles + reduction de multiplication + parallelism openmp  + vectorisation simd + alignement */
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

#elif defined (OPTASM)
#include <math.h>
#include <omp.h>
/* VERSION OPTIMALE AVEC ASSEMBLEUR : calcul unique de log(x) + réduction via asm + OpenMP + SIMD */
void kernel(unsigned n, float a[n][n], const float b[n][n], float x) {
    float log_x = log(x);
    #pragma omp parallel for
    for (unsigned j = 0; j < n; j++) {
        float sum = 0.0f;
        // Déroulement manuel de la boucle k en assembleur
        __asm__ volatile (
            "xorps %%xmm1, %%xmm1\n\t"           // sum = 0 (xmm1 = 0)
            "addss %[b0], %%xmm1\n\t"            // sum += b[0][j]
            "addss %[b1], %%xmm1\n\t"            // sum += b[1][j]
            "addss %[b2], %%xmm1\n\t"            // sum += b[2][j]
            "addss %[b3], %%xmm1\n\t"            // sum += b[3][j]
            "addss %[b4], %%xmm1\n\t"            // sum += b[4][j]
            "addss %[b5], %%xmm1\n\t"            // sum += b[5][j]
            "mulss %[log_x], %%xmm1\n\t"         // sum *= log_x
            "movss %%xmm1, %[sum]\n\t"           // Stocker sum
            : [sum] "=m" (sum)                   // Sortie : sum
            : [b0] "m" (b[0][j]), [b1] "m" (b[1][j]), [b2] "m" (b[2][j]), 
              [b3] "m" (b[3][j]), [b4] "m" (b[4][j]), [b5] "m" (b[5][j]), 
              [log_x] "m" (log_x)                // Entrées : b[k][j], log_x
            : "xmm1"                             // Registre modifié
        );

        // Boucle SIMD sur i avec OpenMP
        #pragma omp simd aligned(a:32)
        for (unsigned i = 0; i < n; i++) {
            a[i][j] += sum;
        }
    }
}

#elif defined (OPTSEQ)
#include <math.h>
void kernel(unsigned n, float a[n][n], const float b[n][n], float x) {
    float log_x = log(x);
    for (unsigned j = 0; j < n; j++) {
        float s = b[0][j] + b[1][j] + b[2][j] + b[3][j] + b[4][j] + b[5][j];
        float t = log_x * s;
        for (unsigned i = 0; i < n; i++) {
            a[i][j] += t;
        }
    }
}


#elif defined (OPTSEQ2)
#include <math.h>
void kernel(unsigned n, float a[n][n], const float b[n][n], float x) {
    // Calcul de log(x) une seule fois
    float log_x = log(x);
    
    // Tableau temporaire pour stocker les valeurs précalculées
    float t[n];
    
    // Précalcul de t[j] = log_x * somme(b[k][j]) pour chaque j
    for (unsigned j = 0; j < n; j++) {
        float s = b[0][j] + b[1][j] + b[2][j] + b[3][j] + b[4][j] + b[5][j];
        t[j] = log_x * s;
    }
    
    // Mise à jour de a[i][j] ligne par ligne pour optimiser l'accès mémoire et la vectorisation
    for (unsigned i = 0; i < n; i++) {
        for (unsigned j = 0; j < n; j++) {
            a[i][j] += t[j];
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
