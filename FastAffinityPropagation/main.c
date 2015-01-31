//
//  main.c
//  FastAffinityPropagation
//
//  Created by Bismarrck on 1/20/15.
//  Copyright (c) 2015 Nexd. All rights reserved.
//

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <Accelerate/Accelerate.h>
#include "utarray.h"
#include "clib.h"
#include "pairwise.h"
#include "mathlib.h"

#ifndef DEBUG
#define DEBUG
#endif

/**
 * @function dcmp_sort_asc
 * A sorting function for double precision arrays for ascending order.
 */
int dcmp_sort_asc(const void *a, const void *b) {
  double *d1 = (double *)a;
  double *d2 = (double *)b;
  return dcmp(*d1, *d2);
}

/**
 * @function dcmp_sort_des
 * A sorting function for double precision arrays for descending order.
 */
int dcmp_sort_des(const void *a, const void *b) {
  return -1 * dcmp_sort_asc(a, b);
}

typedef struct {
  double *similarity;
  unsigned int N;
  unsigned int maxiter;
  double *preference;
  double damping;
  unsigned int ncheck;
  bool verbose;
  double *_Au;
  double *_Al;
  double *_Ru;
  bool *_edges;
  double *R;
  double *A;
  int *exemplar;
  int *clusters;
  unsigned int ncluster;
} AffinityPropagation;

/**
 * @function median
 * This function computes the median value of the given vector.
 *
 * @note Not implemented yet!
 */
double median(const double *vector, unsigned int ntotal) {
  double *vsort = NULL;
  ALLOCATE(vsort, ntotal, double);
  memcpy(vsort, vector, ntotal * sizeof(double));
  qsort(vsort, ntotal, sizeof(double), dcmp_sort_asc);
  double median = 0.0;
  if (ntotal % 2 == 0) {
    median = (vsort[ntotal / 2] + vsort[ntotal / 2 - 1]) * 0.5;
  } else {
    median = vsort[(ntotal - 1) / 2];
  }
  free(vsort);
  return median;
}

/**
 * @function AffinityPropagation_init
 * Initialize a new affinity propagation task.
 *
 * @param S           the similarity matrix.
 * @param N           the number of data points.
 * @param maxiter     the number of maximum iterations.
 * @param preference  the initial preferences for the data points.
 * @param verbose     if true, some internal logs will be printed.
 */
AffinityPropagation *
AffinityPropagation_init(const double *S, const unsigned int N,
                         unsigned int maxiter, unsigned int ncheck,
                         double *preference, bool verbose) {
  if (N >= 65536) {
    fprintf(stderr, "N >= 65536 not supported!");
    return NULL;
  }

  if (S == NULL) {
    fprintf(stderr, "The similarity matrix S must not be NULL!");
    return NULL;
  }

  AffinityPropagation *ap = malloc(sizeof(AffinityPropagation));

  unsigned int N2 = N * N;
  ap->similarity = calloc(sizeof(double), N2);
  memcpy(ap->similarity, S, sizeof(double) * N2);

  ap->N = N;
  ap->ncheck = ncheck;
  ap->maxiter = maxiter;
  ap->preference = calloc(sizeof(double), N);
  ap->verbose = verbose;
  ap->_Al = NULL;
  ap->_Au = NULL;
  ap->_Ru = NULL;
  ap->exemplar = NULL;
  ap->clusters = NULL;
  ap->_edges = NULL;

  if (preference) {
    memcpy(ap->preference, preference, sizeof(double) * N);
  } else {
    double med = median(S, N2);
#if defined(DEBUG)
    printf("The median of the S: %.6lf\n", med);
#endif
    for (int i = 0; i < N; i++) {
      ap->preference[i] = med;
    }
  }

  return ap;
}

/**
 * @function AffinityPropagation_free
 * Free the given object.
 */
void AffinityPropagation_free(AffinityPropagation *ap) {
  if (ap == NULL) {
    return;
  }

  DEALLOCATE(ap->_Ru);
  DEALLOCATE(ap->_Au);
  DEALLOCATE(ap->_Al);
  DEALLOCATE(ap->_edges)
  DEALLOCATE(ap->A);
  DEALLOCATE(ap->R);
  DEALLOCATE(ap->similarity);
  DEALLOCATE(ap->preference);
  DEALLOCATE(ap->clusters);
  DEALLOCATE(ap->exemplar);
  DEALLOCATE(ap);
};

/**
 * @function avalibility_lower
 * Precompute the lower bound of the availability all pairs.
 *
 * @see equation (5)
 */
void availability_lower(AffinityPropagation *ap) {

  clock_t tic = clock();

  unsigned int N2 = ap->N * ap->N;
  ALLOCATE(ap->_Al, N2, double);

  // Set the a_lower for all i,j pairs.
  for (int j = 0; j < ap->N; j++) {
    int jj = j * ap->N + j;
    double a_ij = dmin(0.0, ap->R[jj]);

    for (int i = 0; i < ap->N; i++) {
      int ij = i * ap->N + j;
      if (i == j) {
        ap->_Al[ij] = 0.0;
      } else {
        ap->_Al[ij] = a_ij;
      }
    }
  }

#if defined(DEBUG)
  double rmin = vdmin(ap->_Al, N2, NULL);
  double rmax = vdmax(ap->_Al, N2, NULL);
  printf("availability_lower range  : [%.2f, %.2f]\n", rmin, rmax);
#endif

  double time = (double)(clock() - tic) / (double)CLOCKS_PER_SEC;
  printf("Routine: %50s | time: %8.3f s\n", __func__, time);
}

/**
 * @function responsibility_upper
 * Precompute the upper bound of the responsibility for all pairs.
 *
 * @see equation (6)
 */
void responsibility_upper(AffinityPropagation *ap) {
  clock_t tic = clock();

  unsigned int N2 = ap->N * ap->N;

  double *AS, *Y1, *Y2;
  int *I;

  ALLOCATE(AS, N2, double);
  ALLOCATE(I, ap->N, int);
  ALLOCATE(Y1, ap->N, double);
  ALLOCATE(Y2, ap->N, double);
  ALLOCATE(ap->_Ru, N2, double);

  cblas_dcopy(N2, ap->similarity, 1, AS, 1);
  cblas_daxpy(N2, 1.0, ap->_Al, 1, AS, 1);

  for (int i = 0; i < ap->N; i++) {
    int i0 = i * ap->N;

    // Find the largest value for each row.
    Y1[i] = vdmax(&AS[i0], ap->N, &I[i]);

    // Set the largest value to -infinity and find the second largest value for
    // each row.
    AS[i0 + I[i]] = -FLT_MAX;
    Y2[i] = vdmax(&AS[i0], ap->N, NULL);

    // Restore the default value.
    AS[i0 + I[i]] = Y1[i];
  }

  // Since for each i, Al_ii = 0.0, the equation (6) can be written like this:
  // Ru_ij = S_ij - \max_{k != j}{ Al_ik + S_ik }
  for (int i = 0; i < ap->N; i++) {
    int i0 = i * ap->N;
    for (int j = 0; j < ap->N; j++) {
      ap->_Ru[i0 + j] = ap->similarity[i0 + j] - Y1[i];
    }
    ap->_Ru[i0 + I[i]] = ap->similarity[i0 + I[i]] - Y2[i];
  }

  DEALLOCATE(I);
  DEALLOCATE(Y1);
  DEALLOCATE(Y2);
  DEALLOCATE(AS);

#if defined(DEBUG)
  double rmin = vdmin(ap->_Ru, N2, NULL);
  double rmax = vdmax(ap->_Ru, N2, NULL);
  printf("responsibility_upper range: [%.2f, %.2f]\n", rmin, rmax);
#endif

  double time = (double)(clock() - tic) / (double)CLOCKS_PER_SEC;
  printf("Routine: %50s | time: %8.3f s\n", __func__, time);
}

/**
 * @function availability_upper
 * Precompute the upper bound of the availability of all pairs.
 *
 * @see equation (7)
 */
void availability_upper(AffinityPropagation *ap) {
  clock_t tic = clock();

  unsigned int N2 = ap->N * ap->N;
  ALLOCATE(ap->_Au, N2, double);

  double *Rp, *Rs, *dA;
  ALLOCATE(Rp, N2, double);
  ALLOCATE(Rs, ap->N, double);
  ALLOCATE(dA, ap->N, double);

  int k = 0;

  // Rp = [ \max{Ru_ij, 0.0} ] for i,j in [1, N] and the diagonal elements of Rp
  // are set to Ru_ii because of the Ru_jj term in equation (7).
  // Rs = Rp.sum(axis=0)
  for (int i = 0; i < ap->N; i++) {
    for (int j = 0; j < ap->N; j++) {
      if (i == j) {
        Rp[k] = ap->_Ru[k];
      } else {
        Rp[k] = dmax(ap->_Ru[k], 0.0);
      }
      Rs[j] += Rp[k];
      k++;
    }
  }

  // This part computes Rs - Au[i, :] row-by-row which is described by the term:
  // Ru_jj + \sum_{k != i,j}{ \max{ 0.0, Ru_kj } }
  cblas_dcopy(N2, Rp, 1, ap->_Au, 1);
  for (int i = 0; i < ap->N; i++) {
    int i0 = i * ap->N;
    catlas_daxpby(ap->N, 1.0, Rs, 1, -1.0, &ap->_Au[i0], 1);
    int ii = i0 + i;
    dA[i] = ap->_Au[ii];
  }

  // This part computes the final availability_upper matrix.
  k = 0;
  for (int i = 0; i < ap->N; i++) {
    for (int j = 0; j < ap->N; j++) {
      if (i == j) {
        ap->_Au[k] = dA[i];
      } else {
        ap->_Au[k] = dmin(0.0, ap->_Au[k]);
      }
      k++;
    }
  }

  DEALLOCATE(dA);
  DEALLOCATE(Rs);
  DEALLOCATE(Rp);

#if defined(DEBUG)
  double rmin = vdmin(ap->_Au, N2, NULL);
  double rmax = vdmax(ap->_Au, N2, NULL);
  printf("availability_upper range  : [%.2f, %.2f]\n", rmin, rmax);
#endif

  double time = (double)(clock() - tic) / (double)CLOCKS_PER_SEC;
  printf("Routine: %50s | time: %8.3f s\n", __func__, time);
}

/**
 * @function responsibility_init
 * Compute the initial responsibility for all data points pairs.
 */
void responsibility_init(AffinityPropagation *ap) {
  clock_t tic = clock();

  double *Y1, *Y2;
  int *I;
  ALLOCATE(Y1, ap->N, double);
  ALLOCATE(Y2, ap->N, double);
  ALLOCATE(I, ap->N, int);

  for (int i = 0; i < ap->N; i++) {
    int i0 = i * ap->N;
    Y1[i] = vdmax(&ap->similarity[i0], ap->N, &I[i]);
    ap->similarity[i0 + I[i]] = -FLT_MAX;
    Y2[i] = vdmax(&ap->similarity[i0], ap->N, NULL);
    ap->similarity[i0 + I[i]] = Y1[i];
  }

  for (int i = 0; i < ap->N; i++) {
    int i0 = i * ap->N;
    for (int j = 0; j < ap->N; j++) {
      ap->R[i0 + j] = ap->similarity[i0 + j] - Y1[i];
    }
    ap->R[i0 + I[i]] = ap->similarity[i0 + I[i]] - Y2[i];
  }

#if defined(DEBUG)

  double _y1max, _y1min, _y2max, _y2min;
  _y1max = vdmax(Y1, ap->N, NULL);
  _y1min = vdmin(Y1, ap->N, NULL);
  _y2max = vdmax(Y1, ap->N, NULL);
  _y2min = vdmin(Y1, ap->N, NULL);
  printf("Y1: [%.3f, %.3f]\n", _y1min, _y1max);
  printf("Y2: [%.3f, %.3f]\n", _y2min, _y2max);

#endif

  DEALLOCATE(Y1);
  DEALLOCATE(Y2);
  DEALLOCATE(I);

  double time = (double)(clock() - tic) / (double)CLOCKS_PER_SEC;
  printf("Routine: %50s | time: %8.3f s\n", __func__, time);
}

/**
 * @function AffinityPropagation_initBound
 * Precompute the boundary values of the responsibility and availability for
 * each pair.
 *
 * @see Algorithm 1, line 1-3
 */
void AffinityPropagation_initBound(AffinityPropagation *ap) {
  clock_t tic = clock();

  responsibility_init(ap);
  availability_lower(ap);
  responsibility_upper(ap);
  availability_upper(ap);

  double time = (double)(clock() - tic) / (double)CLOCKS_PER_SEC;
  printf("Routine: %50s | time: %8.3f s\n", __func__, time);
}

/**
 * @function AffinityPropagation_link
 * Link all possible data point pairs.
 *
 * IF ru_[i,j] >= 0 OR au_[i,j] + s[i,j] >= \max_{k != j}{ al_[i,k] + s[i,k] }
 * THEN
 *    link [i,j]
 * END IF
 *
 * @see Algorithm 1, line 4 - 8
 */
void AffinityPropagation_link(AffinityPropagation *ap) {
  clock_t tic = clock();

  unsigned int N2 = ap->N * ap->N;
  ALLOCATE(ap->_edges, N2, bool);
  for (int i = 0; i < N2; i++) {
    ap->_edges[i] = false;
  }

  for (int i = 0; i < ap->N; i++) {
    for (int j = i; j < ap->N; j++) {
      int ij = i * ap->N + j;
      int ji = j * ap->N + i;
      if (dcmp(ap->_Ru[ij], 0.0) >= 0 ||
          dcmp(ap->_Ru[ij] + ap->_Au[ij], 0.0) >= 0) {
        ap->_edges[ij] = ap->_edges[ji] = true;
      }
    }
  }

  if (ap->verbose) {
    int density = 0;
    for (int i = 0; i < N2; i++) {
      density += ap->_edges[i] ? 1 : 0;
    }
    printf("Graph Sparseness: %.4f\n", (double)(N2 - density) / N2);
  }

  double time = (double)(clock() - tic) / (double)CLOCKS_PER_SEC;
  printf("Routine: %50s | time: %8.3f s\n", __func__, time);
}

/**
 * @function AffinityPropagation_update_linked
 * Update each linked data pair iteratively with equation (1):
 *
 *  r_{i,j} = (1 - \lambda)\rho_{i,j}   + \lambda r_{i,j}
 *  a_{i,j} = (1 - \lambda)\alpha_{i,j} + \lambda a_{i,j}
 *
 * @see Algorithm 1, line 9 - 13
 */
void AffinityPropagation_update_linked(AffinityPropagation *ap) {
  clock_t tic = clock();

  unsigned int iter = 0;
  unsigned int N2 = ap->N * ap->N;

  double *rho, *alp, *AS, *max, *Y1, *Y2, *Rp, *Rs, *dA;
  int *I;

  ALLOCATE(rho, N2, double);
  ALLOCATE(alp, N2, double);
  ALLOCATE(AS, N2, double);
  ALLOCATE(max, ap->N * 2, double);
  ALLOCATE(I, ap->N, int)
  ALLOCATE(Y1, ap->N, double);
  ALLOCATE(Y2, ap->N, double);
  ALLOCATE(Rp, N2, double);
  ALLOCATE(Rs, ap->N, double);
  ALLOCATE(dA, ap->N, double);

  // Pre-set the Rp matrix so that only the linked parts of Rp will be updated
  // during the iteration.
  for (int k = 0; k < N2; k++) {
    Rp[k] = dmax(ap->R[k], 0.0);
  }

  while (iter < ap->maxiter) {

    // Compute \sum_{i=1,k=1}{a[i,k]+s[i,k]} in equation (2)
    cblas_dcopy(N2, ap->similarity, 1, AS, 1);
    cblas_daxpy(N2, 1.0, ap->A, 1, AS, 1);

    // Y1 and Y2 are the first and second largest values for each row in AS.
    for (int i = 0; i < ap->N; i++) {
      int i0 = i * ap->N;

      Y1[i] = vdmax(&AS[i0], ap->N, &I[i]);
      AS[i0 + I[i]] = -FLT_MAX;
      Y2[i] = vdmax(&AS[i0], ap->N, NULL);
      AS[i0 + I[i]] = Y1[i];
    }

    // Update the responsibility of the linked data point pairs.
    for (int i = 0; i < ap->N; i++) {
      int i0 = i * ap->N;
      for (int j = 0; j < ap->N; j++) {
        if (ap->_edges[i0 + j]) {
          rho[i0 + j] = ap->similarity[i0 + j] - Y1[i];
        }
      }
      if (ap->_edges[i0 + I[i]]) {
        rho[i0 + I[i]] = ap->similarity[i0 + I[i]] - Y2[i];
      }
    }

    // Update the availability of the linked data point pairs.
    int k = 0;
    for (int i = 0; i < ap->N; i++) {
      for (int j = 0; j < ap->N; j++) {
        if (ap->_edges[k]) {
          if (i == j) {
            Rp[k] = ap->R[k];
          } else {
            Rp[k] = dmax(ap->R[k], 0.0);
          }
          Rs[j] += Rp[k];
        }
        k++;
      }
    }

    k = 0;
    for (int i = 0; i < ap->N; i++) {
      int i0 = i * ap->N;
      for (int j = 0; j < ap->N; j++) {
        if (ap->_edges[k]) {
          alp[k] = Rs[j] - Rp[k];
        }
        k++;
      }
      dA[i] = alp[i0 + i];
    }

    k = 0;
    for (int i = 0; i < ap->N; i++) {
      for (int j = 0; j < ap->N; j++) {
        if (ap->_edges[k]) {
          if (i == j) {
            alp[k] = dA[i];
          } else {
            alp[k] = dmin(0.0, alp[k]);
          }
        }
        k++;
      }
    }

    catlas_dset(ap->N, 0.0, Rs, 1);
    catlas_dset(ap->N, 0.0, dA, 1);
    catlas_dset(N2, 0.0, Rp, 1);

    // The R and A shall be updated according to the links!
    k = 0;
    double minusdamp = 1.0 - ap->damping;
    for (int i = 0; i < ap->N; i++) {
      for (int j = 0; j < ap->N; j++) {
        if (ap->_edges[k]) {
          ap->R[k] = minusdamp * rho[k] + ap->damping * ap->R[k];
          ap->A[k] = minusdamp * alp[k] + ap->damping * ap->A[k];
        }
        k++;
      }
    }

    catlas_dset(N2, 0.0, rho, 1);
    catlas_dset(N2, 0.0, alp, 1);

    iter++;
    printf("Iteration %d finished!\n", iter);
  }

  DEALLOCATE(rho);
  DEALLOCATE(alp);
  DEALLOCATE(AS);
  DEALLOCATE(max);
  DEALLOCATE(I);
  DEALLOCATE(Y1);
  DEALLOCATE(Y2);
  DEALLOCATE(Rp);
  DEALLOCATE(Rs);
  DEALLOCATE(dA);

  double time = (double)(clock() - tic) / (double)CLOCKS_PER_SEC;
  printf("Routine: %50s | time: %8.3f s\n", __func__, time);
}

const double *AffinityPropagation_RA(AffinityPropagation *ap) {
  clock_t tic = clock();

  unsigned int N2 = ap->N * ap->N;
  double *RA;
  ALLOCATE(RA, N2, double);

  cblas_dcopy(N2, ap->R, 1, RA, 1);
  cblas_daxpy(N2, 1.0, ap->A, 1, RA, 1);

  double time = (double)(clock() - tic) / (double)CLOCKS_PER_SEC;
  printf("Routine: %50s | time: %8.3f s\n", __func__, time);
  return RA;
}

/**
 * @function AffinityPropagation_compute_unlinked
 * Compute the responsibility and availability for each unlinked data points
 * pairs. This function will be executed only once.
 *
 * @param RA  the matrix sum of the responsibility matrix and the availability
 *            matrix.
 *
 * @see Algorithm 1, line 14-16
 */
void AffinityPropagation_compute_unlinked(AffinityPropagation *ap,
                                          const double *RA) {
  clock_t tic = clock();

  unsigned int N2 = ap->N * ap->N;
  double *AS, *Y1, *Y2, *Rp, *Rs, *dA;
  int *I;

  ALLOCATE(AS, N2, double);
  ALLOCATE(Y1, ap->N, double);
  ALLOCATE(Y2, ap->N, double);
  ALLOCATE(I, ap->N, int);

  cblas_dcopy(N2, ap->A, 1, AS, 1);
  cblas_daxpy(N2, 1.0, ap->similarity, 1, AS, 1);

  for (int i = 0; i < ap->N; i++) {
    int i0 = i * ap->N;

    Y1[i] = vdmax(&AS[i0], ap->N, &I[i]);
    AS[i0 + I[i]] = -FLT_MAX;
    Y2[i] = vdmax(&AS[i0], ap->N, NULL);
    AS[i0 + I[i]] = Y1[i];
  }

  for (int i = 0; i < ap->N; i++) {
    int i0 = i * ap->N;
    for (int j = 0; j < ap->N; j++) {
      if (ap->_edges[i0 + j] == false) {
        ap->R[i0 + j] = ap->similarity[i0 + j] - Y1[i];
      }
    }
    if (ap->_edges[i0 + I[i]] == false) {
      ap->R[i0 + I[i]] = ap->similarity[i0 + I[i]] - Y2[i];
    }
  }

  DEALLOCATE(Y1);
  DEALLOCATE(Y2);
  DEALLOCATE(I);
  DEALLOCATE(AS);

  int k = 0;

  ALLOCATE(Rp, N2, double);
  ALLOCATE(Rs, ap->N, double);
  ALLOCATE(dA, ap->N, double);

  // Rp = [ \max{Ru_ij, 0.0} ] for i,j in [1, N] and the diagonal elements of Rp
  // are set to Ru_ii because of the Ru_jj term in equation (7).
  // Rs = Rp.sum(axis=0)
  for (int i = 0; i < ap->N; i++) {
    for (int j = 0; j < ap->N; j++) {
      if (i == j) {
        Rp[k] = ap->R[k];
      } else {
        Rp[k] = dmax(ap->R[k], 0.0);
      }
      Rs[j] += Rp[k];
      k++;
    }
  }

  k = 0;
  for (int i = 0; i < ap->N; i++) {
    int i0 = i * ap->N;
    for (int j = 0; j < ap->N; j++) {
      if (ap->_edges[k] == false) {
        ap->A[k] = Rs[j] - Rp[k];
      }
      k++;
    }
    dA[i] = ap->A[i0 + i];
  }

  k = 0;
  for (int i = 0; i < ap->N; i++) {
    for (int j = 0; j < ap->N; j++) {
      if (ap->_edges[k] == false) {
        if (i == j) {
          ap->A[k] = dA[i];
        } else {
          ap->A[k] = dmin(0.0, ap->A[k]);
        }
      }
      k++;
    }
  }

  DEALLOCATE(Rp);
  DEALLOCATE(Rs);
  DEALLOCATE(dA);

  double time = (double)(clock() - tic) / (double)CLOCKS_PER_SEC;
  printf("Routine: %50s | time: %8.3f s\n", __func__, time);
}

/**
 * @function AffinityPropagation_exemplar
 * Compute the exemplar and the connections.
 *
 * @see Algorithm 1, line 17-19
 */
void AffinityPropagation_exemplar(AffinityPropagation *ap, const double *RA) {
  clock_t tic = clock();

  ALLOCATE(ap->exemplar, ap->N, int);
  ALLOCATE(ap->clusters, ap->N, int);

  int counter = 0;
  for (int i = 0; i < ap->N; i++) {
    int exemplar = 0;
    vdmax((double *)&RA[i * ap->N], ap->N, &exemplar);
    ap->exemplar[i] = exemplar;
    bool flag = true;
    for (int j = 0; j < counter; j++) {
      if (ap->clusters[j] == exemplar) {
        flag = false;
        continue;
      }
    }
    if (flag) {
      ap->clusters[counter] = exemplar;
      counter++;
    }
  }

  ap->clusters = realloc(ap->clusters, counter);
  ap->ncluster = counter;

  double time = (double)(clock() - tic) / (double)CLOCKS_PER_SEC;
  printf("Routine: %50s | time: %8.3f s\n", __func__, time);
}

void AffinityPropagation_fit(AffinityPropagation *ap) {
  clock_t tic = clock();

  for (int i = 0; i < ap->N; i++) {
    int ii = i * ap->N + i;
    ap->similarity[ii] = ap->preference[i];
  }

  unsigned int N2 = ap->N * ap->N;
  ALLOCATE(ap->R, N2, double);
  ALLOCATE(ap->A, N2, double);

  AffinityPropagation_initBound(ap);
  AffinityPropagation_link(ap);
  AffinityPropagation_update_linked(ap);

  const double *RA = AffinityPropagation_RA(ap);
  AffinityPropagation_compute_unlinked(ap, RA);
  AffinityPropagation_exemplar(ap, RA);
  free((double *)RA);

  double time = (double)(clock() - tic) / (double)CLOCKS_PER_SEC;
  printf("Routine: %50s | time: %8.3f s\n", __func__, time);
}

double *Matrix_load(const char *filename, unsigned int row, unsigned int col) {
  FILE *F = fopen(filename, "r");
  if (F == NULL) {
    return NULL;
  }

  double *X = NULL;
  ALLOCATE(X, row * col, double);

  char buf[128];
  int k = 0;
  int n = row * col;
  while (fgets(buf, sizeof(buf), F) != NULL) {
    sscanf(buf, "%lf", &X[k]);
    k++;
    if (n == k) {
      break;
    }
  }

  return X;
}

int main(int argc, const char *argv[]) {

  //  FILE *fp = fopen(
  //      "/Users/bismarrck/Documents/Project/FastAffinityPropagation/data.txt",
  //      "r");
  //
  //  if (fp == NULL) {
  //    abort();
  //  }
  //
  //  int nrow = 1500;
  //  int ncol = 2;
  //
  //  char line[1024];
  //  double x = 0.0;
  //  double y = 0.0;
  //  int label = 0;
  //
  //  double *matrix = calloc(sizeof(double), nrow * ncol);
  //  int *true_labels = calloc(sizeof(int), nrow);
  //
  //  int k = 0;
  //  while (fgets(line, sizeof(line), fp) != NULL) {
  //    int n = sscanf(line, "%lf %lf %d", &x, &y, &label);
  //    if (n == 3) {
  //      matrix[k * 2 + 0] = x;
  //      matrix[k * 2 + 1] = y;
  //      true_labels[k] = label;
  //      k++;
  //    }
  //  }
  //  assert(k == nrow);
  //  fclose(fp);
  //
  //  double *S = pairwise_distance_matrix(matrix, nrow, 2, true);

  unsigned int dim = 1775;

  double *S = Matrix_load(
      "/Users/bismarrck/Documents/Project/FastAffinityPropagation/data.txt",
      dim, dim);

  unsigned int maxiter = 100;
  unsigned int ncheck = 5;

  AffinityPropagation *ap =
      AffinityPropagation_init(S, dim, maxiter, ncheck, NULL, true);
  AffinityPropagation_fit(ap);

  FILE *fp = fopen(
      "/Users/bismarrck/Documents/Project/FastAffinityPropagation/label.txt",
      "r");

  int labels[dim];
  int k = 0;
  char buf[256];
  while (fgets(buf, sizeof(buf), fp) != NULL) {
    sscanf(buf, "%d\n", &labels[k]);
    k++;
  }
  fclose(fp);

  for (int i = 0; i < dim; i++) {
    if (labels[i] != ap->exemplar[i]) {
      printf("%4d | RAW: %3d --> NEW: %3d\n", i, labels[i], ap->exemplar[i]);
    }
  }

  AffinityPropagation_free(ap);
  free(S);
  return 0;
}
