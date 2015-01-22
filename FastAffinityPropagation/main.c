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
#include <Accelerate/Accelerate.h>
#include "utarray.h"

#define ALLOCATE(p, n, type)                                                   \
  if (p == NULL) {                                                             \
    p = calloc(sizeof(type), n);                                               \
  } else {                                                                     \
    memset(p, 0, sizeof(type) * n);                                            \
  }

/**
 * @function dcmp
 * This function safely compares two double precision floats and returns 1 if
 * the former is larger, 0 if they are equal or -1 if the former is smaller.
 */
static int dcmp(double d1, double d2) {
  double df = d1 - d2;
  if (df > 1.0e-12) {
    return 1;
  } else if (df < -1.0e-12) {
    return -1;
  } else {
    return 0;
  }
}

static double dmax(double d1, double d2) {
  int vs = dcmp(d1, d2);
  if (vs >= 0) {
    return d1;
  } else {
    return d2;
  }
}

static double dmin(double d1, double d2) {
  int vs = dcmp(d1, d2);
  if (vs <= 0) {
    return d1;
  } else {
    return d2;
  }
}

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
  double *_s_sorted;
  double *_al_s_sorted;
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
  ap->_al_s_sorted = NULL;
  ap->_Ru = NULL;
  ap->_s_sorted = NULL;
  ap->_edges = NULL;

  if (preference) {
    memcpy(ap->preference, preference, sizeof(double) * N);
  } else {
    double med = median(S, N2);
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

  if (ap->_al_s_sorted) {
    free(ap->_al_s_sorted);
  }

  if (ap->_s_sorted) {
    free(ap->_s_sorted);
  }
  if (ap->_Ru) {
    free(ap->_Ru);
  }
  if (ap->_Au) {
    free(ap->_Au);
  }
  if (ap->_Al) {
    free(ap->_Al);
  }
  if (ap->A) {
    free(ap->A);
  }
  if (ap->R) {
    free(ap->R);
  }
  if (ap->similarity) {
    free(ap->similarity);
  }
  if (ap->preference) {
    free(ap->preference);
  }
  if (ap->clusters) {
    free(ap->clusters);
  }
  if (ap->_edges) {
    free(ap->_edges);
  }
  if (ap->exemplar) {
    free(ap->exemplar);
  }
  free(ap);
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

  // Compute the initial $r_{jj}$:
  // r[j,j] = s[j,j] - max{ s[j,k], j != k }
  for (int j = 0; j < ap->N; j++) {
    int j0 = j * ap->N;
    int jj = j * ap->N + j;
    ap->R[jj] = ap->similarity[jj];
    if (dcmp(ap->_s_sorted[j0], ap->similarity[jj]) == 0) {
      ap->R[jj] -= ap->_s_sorted[j0 + 1];
    } else {
      ap->R[jj] -= ap->_s_sorted[j0 + 0];
    }
  }

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

  for (int i = 0; i < ap->N; i++) {
    ap->R[i * ap->N + i] = 0.0;
  }

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

  double *AS = calloc(sizeof(double), N2);
  cblas_dcopy(N2, ap->similarity, 1, AS, 1);
  cblas_daxpy(N2, 1.0, ap->_Al, 1, AS, 1);

  ALLOCATE(ap->_Ru, N2, double);
  cblas_dcopy(N2, ap->similarity, 1, ap->_Ru, 1);

  int *I = calloc(sizeof(int), ap->N);
  double *Y = calloc(sizeof(double), ap->N);
  double *Y2 = calloc(sizeof(double), ap->N);

  for (int i = 0; i < ap->N; i++) {
    int i0 = i * ap->N;
    CBLAS_INDEX i1 = cblas_idamax(ap->N, &AS[i0], 1);
    I[i] = i1;
    Y[i] = AS[i0 + i1];

    AS[i0 + i1] = -FLT_MAX;
    CBLAS_INDEX i2 = cblas_idamax(ap->N, &AS[i0], 1);
    Y2[i] = AS[i0 + i2];

    AS[i0 + i1] = Y[i];
  }

  for (int i = 0; i < ap->N; i++) {
    int i0 = i * ap->N;
    for (int j = 0; j < ap->N; j++) {
      ap->_Ru[i0 + j] -= Y[i];
    }
    ap->_Ru[i0 + I[i]] = ap->similarity[i0 + I[i]] - Y2[i];
  }

  free(I);
  free(Y);
  free(Y2);
  free(AS);

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

  double *Rp = calloc(sizeof(double), N2);
  double *Rs = calloc(sizeof(double), ap->N);
  int k = 0;
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

  cblas_dcopy(N2, Rp, 1, ap->_Au, 1);
  double *dA = calloc(sizeof(double), ap->N);
  for (int i = 0; i < ap->N; i++) {
    int i0 = i * ap->N;
    catlas_daxpby(ap->N, 1.0, Rs, 1, -1.0, &ap->_Au[i0], 1);
    int ii = i0 + i;
    dA[i] = ap->_Au[ii];
  }

  k = 0;
  for (int i = 0; i < ap->N; i++) {
    for (int j = 0; j < ap->N; j++) {
      if (i == j) {
        ap->_Au[k] = dA[i];
      } else {
        ap->_Au[k] = dmin(0.0, ap->_Au[k]);
      }
    }
  }

  free(dA);
  free(Rs);
  free(Rp);

  double time = (double)(clock() - tic) / (double)CLOCKS_PER_SEC;
  printf("Routine: %50s | time: %8.3f s\n", __func__, time);
}

/**
 * @function similarity_sort
 * This function sorts the similarity matrix row-by-row. _s_sorted is the sorted
 * matrix.
 */
void similarity_sort(AffinityPropagation *ap) {
  clock_t tic = clock();

  unsigned int N2 = ap->N * ap->N;
  ALLOCATE(ap->_s_sorted, N2, double);
  memcpy(ap->_s_sorted, ap->similarity, sizeof(double) * N2);
  for (int i = 0; i < ap->N; i++) {
    int i0 = i * ap->N;
    qsort(&ap->_s_sorted[i0], ap->N, sizeof(double), dcmp_sort_des);
  }

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

  similarity_sort(ap);
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
    for (int j = i + 1; j < ap->N; j++) {
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
    printf("Graph Sparseness: %.4f\n",
           1.0 - (double)((int)N2 - density) / (double)N2);
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

  double *rho = NULL;
  double *alp = NULL;
  double *mat = NULL;
  double *max = NULL;
  ALLOCATE(rho, N2, double);
  ALLOCATE(alp, N2, double);
  ALLOCATE(mat, N2, double);
  ALLOCATE(max, ap->N * 2, double);

  while (iter < ap->maxiter) {

    // Compute \sum_{i=1,k=1}{a[i,k]+s[i,k]} in equation (2)
    cblas_dcopy(N2, ap->similarity, 1, mat, 1);
    cblas_daxpy(N2, 1.0, ap->A, 1, mat, 1);

    // Find the first and second largest values of each row
    for (int i = 0; i < ap->N; i++) {
      int i0 = i * ap->N;
      CBLAS_INDEX imax1 = cblas_idamax(ap->N, &mat[i0], 1);
      max[i * 2 + 0] = mat[i0 + imax1];
      mat[i0 + imax1] = -FLT_MAX;
      CBLAS_INDEX imax2 = cblas_idamax(ap->N, &mat[i0], 1);
      max[i * 2 + 1] = mat[i0 + imax2];
      mat[i0 + imax1] = max[i * 2 + 0];
    }

    // Update the responsibility and availability for each linked data point
    // pair [i,j].
    for (int i = 0; i < ap->N; i++) {

      int i0 = i * ap->N;

      for (int j = 0; j < ap->N; j++) {
        if (ap->_edges[j] == false) {
          continue;
        }

        int ij = i * ap->N + j;
        int jj = j * ap->N + j;

        double v = max[i0];
        if (dcmp(v, ap->A[ij]) == 0) {
          v = max[i0 + 1];
        }
        rho[ij] = ap->similarity[ij] - v;

        v = 0.0;
        for (int k = 0; k < ap->N; k++) {
          if (k == i) {
            continue;
          }
          int kj = k * ap->N + j;
          double rkj = ap->R[kj];
          v += dmax(rkj, 0.0);
        }

        if (i == j) {
          alp[ij] = v;
        } else {
          alp[ij] = dmin(0.0, v - dmax(0., ap->R[jj]));
        }
      }
    }

    cblas_dscal(N2, ap->damping, ap->R, 1);
    cblas_daxpy(N2, 1.0 - ap->damping, rho, 1, ap->R, 1);

    cblas_dscal(N2, ap->damping, ap->A, 1);
    cblas_daxpy(N2, 1.0 - ap->damping, alp, 1, ap->A, 1);

    iter++;
    printf("Iteration %d finished!\n", iter);
  }

  free(max);
  free(alp);
  free(rho);
  free(mat);

  double time = (double)(clock() - tic) / (double)CLOCKS_PER_SEC;
  printf("Routine: %50s | time: %8.3f s\n", __func__, time);
}

const double *AffinityPropagation_RA(AffinityPropagation *ap) {
  clock_t tic = clock();

  unsigned int N2 = ap->N * ap->N;
  double *RA = NULL;
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

  for (int i = 0; i < ap->N; i++) {
    int i0 = i * ap->N;

    for (int j = 0; j < ap->N; j++) {
      int ij = i0 + j;
      if (ap->_edges[ij]) {
        continue;
      }
      int jj = j * ap->N + j;

      double v = RA[i0];
      if (dcmp(v, ap->A[ij]) == 0) {
        v = RA[i0 + 1];
      }
      ap->R[ij] = ap->similarity[ij] - v;

      v = 0.0;
      for (int k = 0; k < ap->N; k++) {
        if (k == i) {
          continue;
        }
        int kj = k * ap->N + j;
        double rkj = ap->R[kj];
        v += dmax(rkj, 0.0);
      }

      if (i == j) {
        ap->A[ij] = v;
      } else {
        ap->A[ij] = dmin(0.0, v - dmax(0., ap->R[jj]));
      }
    }
  }

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
    int exemplar = cblas_idamax(ap->N, &RA[i * ap->N], 1);
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

double *pairwise_distance_matrix(double *points, int npoint, int dim,
                                 bool squared) {
  unsigned int N2 = npoint * npoint;
  double *dist = NULL;
  ALLOCATE(dist, N2, double);

  double *vec = NULL;
  ALLOCATE(vec, dim, double);
  double root = 2.0;

  for (int i = 0; i < npoint; i++) {
    double *pi = &points[i * dim];
    for (int j = i + 1; j < npoint; j++) {
      int ij = i * npoint + j;
      int ji = j * npoint + i;
      double *pj = &points[j * dim];
      memcpy(vec, pj, sizeof(double) * dim);

      cblas_daxpy(dim, -1.0, pi, 1, vec, 1);
      vvpows(vec, &root, vec, &dim);

      double d = cblas_dasum(dim, vec, 1);
      dist[ij] = d;
      dist[ji] = d;
    }
  }
  if (squared == false) {
    vvsqrt(dist, dist, (const int *)&N2);
  }
  return dist;
}

void AffinityPropagation_fit(AffinityPropagation *ap) {
  clock_t tic = clock();

  unsigned int N2 = ap->N * ap->N;
  ALLOCATE(ap->R, N2, double);
  ALLOCATE(ap->A, N2, double);

  AffinityPropagation_initBound(ap);
  AffinityPropagation_link(ap);
  //  AffinityPropagation_update_linked(ap);
  //
  //  const double *RA = AffinityPropagation_RA(ap);
  //  AffinityPropagation_compute_unlinked(ap, RA);
  //  AffinityPropagation_exemplar(ap, RA);
  //  free((double *)RA);

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
  double *S = Matrix_load(
      "/Users/bismarrck/Documents/Project/FastAffinityPropagation/data.txt",
      1500, 1500);

  AffinityPropagation *ap = AffinityPropagation_init(S, 1500, 5, 5, NULL, true);
  AffinityPropagation_fit(ap);
  AffinityPropagation_free(ap);

  free(S);
  return 0;
}
