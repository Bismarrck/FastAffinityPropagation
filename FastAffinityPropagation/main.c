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
  int pid;
  int *edge;
  int count;
  int _cap;
  bool _final;
} StaticLink;

StaticLink *StaticLink_init(int pid, int *edge, int count) {
  StaticLink *link = malloc(sizeof(StaticLink));
  link->pid = pid;

  if (count == 0 || edge == NULL) {
    link->_cap = 0;
    link->count = 0;
    link->edge = NULL;
    link->_final = false;
  } else {
    link->edge = NULL;
    ALLOCATE(link->edge, count, int);
    memcpy(link->edge, edge, sizeof(int) * count);
    link->_cap = count;
    link->_final = false;
  }

  return link;
}

void StaticLink_addEdge(StaticLink *link, int edge) {
  if (link->_final) {
    return;
  }
  if (link->edge == NULL) {
    ALLOCATE(link->edge, 10, int);
    memset(link->edge, -1, sizeof(int) * link->_cap);
    link->_cap = 10;
  }
  if (link->count == link->_cap) {
    link->_cap += 10;
    link->edge = realloc(link->edge, sizeof(int) * link->_cap);
  }
  link->edge[link->count] = edge;
  link->count++;
}

void StaticLink_deleteEdge(StaticLink *link, int edge) {
  if (link->_final) {
    return;
  }

  int k = -1;
  for (int i = 0; i < link->count; i++) {
    if (link->edge[i] == edge) {
      k = i;
      break;
    }
  }
  if (k == -1) {
    return;
  }

  int n = link->count - k - 1;
  memcpy(&link->edge[k], &link->edge[k + 1], sizeof(int) * n);
  link->count--;
  link->edge[link->count - 1] = -1;
}

void StaticLink_finalize(StaticLink *link) {
  link->_final = true;
  if (link->_cap > link->count) {
    link->edge = realloc(link->edge, sizeof(int) * link->count);
    link->_cap = link->count;
  }
}

void StaticLink_free(StaticLink *link) {
  if (link == NULL) {
    return;
  }
  if (link->edge) {
    free(link->edge);
  }
  free(link);
}

typedef struct {
  double *similarity;
  unsigned int N;
  unsigned int maxiter;
  double *preference;
  double damping;
  unsigned int ncheck;
  bool verbose;
  double *_a_upper;
  double *_a_lower;
  double *_r_upper;
  double *_s_sorted;
  double *_al_s_sorted;
  double *R;
  double *A;
  StaticLink **_links;
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
 *
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
  ap->_a_lower = NULL;
  ap->_a_upper = NULL;
  ap->_al_s_sorted = NULL;
  ap->_r_upper = NULL;
  ap->_s_sorted = NULL;
  ap->_links = NULL;

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
  if (ap->_r_upper) {
    free(ap->_r_upper);
  }
  if (ap->_a_upper) {
    free(ap->_a_upper);
  }
  if (ap->_a_lower) {
    free(ap->_a_lower);
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
  free(ap);
};

/**
 * @function avalibility_lower
 * Precompute the lower bound of the availability all pairs.
 *
 * @see equation (5)
 */
void availability_lower(AffinityPropagation *ap) {
  unsigned int N2 = ap->N * ap->N;
  ALLOCATE(ap->_a_lower, N2, double);

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
    double a_ij = dcmp(ap->R[jj], 0.0) <= 0 ? ap->R[jj] : 0.0;
    for (int i = 0; i < ap->N; i++) {
      int ij = i * ap->N + j;
      if (i == j) {
        ap->_a_lower[ij] = 0.0;
      } else {
        ap->_a_lower[ij] = a_ij;
      }
    }
  }
}

/**
 * @function responsibility_upper
 * Precompute the upper bound of the responsibility for all pairs.
 *
 * @see equation (6)
 */
void responsibility_upper(AffinityPropagation *ap) {
  unsigned int N2 = ap->N * ap->N;
  ALLOCATE(ap->_r_upper, N2, double);

  // Compute the a_[i,k] + s[i,k]
  ALLOCATE(ap->_al_s_sorted, N2, double);
  cblas_dcopy(N2, ap->_a_lower, 1, ap->_al_s_sorted, 1);
  cblas_daxpy(N2, 1.0, ap->similarity, 1, ap->_al_s_sorted, 1);

  // Sort the each row of as.
  for (int i = 0; i < ap->N; i++) {
    int i0 = i * ap->N;
    qsort(&ap->_al_s_sorted[i0], ap->N, sizeof(double), dcmp_sort_des);
  }

  int ij = 0;
  for (int i = 0; i < ap->N; i++) {
    int i0 = i * ap->N;
    for (int j = 0; j < ap->N; j++) {
      double sub = 0.0;
      if (i == j) {
        sub = ap->_s_sorted[i0];
      } else {
        if (dcmp(ap->_al_s_sorted[ij], ap->_al_s_sorted[i0]) == 0) {
          sub = ap->_al_s_sorted[i0 + 1];
        } else {
          sub = ap->_al_s_sorted[i0];
        }
      }
      ap->_r_upper[ij] = ap->similarity[ij] - sub;
      ij++;
    }
  }
}

/**
 * @function availability_upper
 * Precompute the upper bound of the availability of all pairs.
 *
 * @see equation (7)
 */
void availability_upper(AffinityPropagation *ap) {
  unsigned int N2 = ap->N * ap->N;
  ALLOCATE(ap->_a_upper, N2, double);

  for (int j = 0; j < ap->N; j++) {
    int jj = j * ap->N + j;
    double r_jj = ap->_r_upper[jj];

    for (int i = 0; i < ap->N; i++) {
      // sum = \sum_{k != i}{ \max{ 0, ru_[k,j] } }
      double sum = 0.0;
      int ij = i * ap->N + j;
      for (int k = 0; k < ap->N; k++) {
        if (k == i) {
          continue;
        }
        int kj = k * ap->N + j;
        double r_kj = ap->_r_upper[kj];
        sum += dcmp(r_kj, 0.0) == 1 ? r_kj : 0.0;
      }

      if (i == j) {
#warning Is this right?
        // au_[i,j] = \sum_{k != i}{ max{0, ru_[k,j]} }
        sum -= dcmp(r_jj, 0.0) > 0 ? r_jj : 0.0;
        ap->_a_upper[ij] = sum;
      } else {
        // au_[i,j] = \min{
        //    0.0,
        //    ru_[j,j] + \sum_{k != i}{ max{0, ru_[k,j]} } - max{0, ru_[j,j]}
        // }
        sum += r_jj - dcmp(r_jj, 0.0) > 0 ? r_jj : 0.0;
        ap->_a_upper[ij] = dcmp(sum, 0.0) < 0 ? sum : 0.0;
      }
    }
  }
}

/**
 * @function similarity_sort
 * This function sorts the similarity matrix row-by-row. _s_sorted is the sorted
 * matrix.
 */
void similarity_sort(AffinityPropagation *ap) {
  unsigned int N2 = ap->N * ap->N;
  ALLOCATE(ap->_s_sorted, N2, double);
  memcpy(ap->_s_sorted, ap->similarity, sizeof(double) * N2);
  for (int i = 0; i < ap->N; i++) {
    int i0 = i * ap->N;
    qsort(&ap->_s_sorted[i0], ap->N, sizeof(double), dcmp_sort_des);
  }
}

/**
 * @function AffinityPropagation_initBound
 * Precompute the boundary values of the responsibility and availability for
 * each pair.
 *
 * @see Algorithm 1, line 1-3
 */
void AffinityPropagation_initBound(AffinityPropagation *ap) {
  similarity_sort(ap);
  availability_lower(ap);
  responsibility_upper(ap);
  availability_upper(ap);
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

  ALLOCATE(ap->_links, ap->N, StaticLink *);
  for (int i = 0; i < ap->N; i++) {
    ap->_links[i] = NULL;
  }

  for (int i = 0; i < ap->N; i++) {
    int i0 = i * ap->N;

    for (int j = i + 1; j < ap->N; j++) {
      int ij = i * ap->N + j;

      double au_ij = ap->_a_upper[ij];
      double s_ij = ap->similarity[ij];
      double al_ij = ap->_a_lower[ij];
      double alsmax = ap->_al_s_sorted[i0];

      if (dcmp(alsmax, al_ij + s_ij) == 0) {
        alsmax = ap->_al_s_sorted[i0 + 1];
      }

      if (dcmp(ap->_r_upper[ij], 0.0) >= 0 || dcmp(au_ij + s_ij, alsmax) >= 0) {
        if (ap->_links[i] == NULL) {
          ap->_links[i] = StaticLink_init(i, NULL, 0);
        }
        StaticLink_addEdge(ap->_links[i], j);
      }
    }
  }
  for (int i = 0; i < ap->N; i++) {
    if (ap->_links[i] == NULL) {
      continue;
    }
    StaticLink_finalize(ap->_links[i]);
  }
}

/**
 * @function AffinityPropagation_update
 * Update each linked data pair iteratively with equation (1):
 *
 *  r_{i,j} = (1 - \lambda)\rho_{i,j}   + \lambda r_{i,j}
 *  a_{i,j} = (1 - \lambda)\alpha_{i,j} + \lambda a_{i,j}
 *
 * @see Algorithm 1, line 9 - 13
 */
void AffinityPropagation_update(AffinityPropagation *ap) {

  unsigned int iter = 0;
  unsigned int N2 = ap->N * ap->N;

  double *rho = NULL;
  double *alp = NULL;
  double *mat = NULL;
  ALLOCATE(rho, N2, double);
  ALLOCATE(alp, N2, double);
  ALLOCATE(mat, N2, double);

  while (iter < ap->maxiter) {

    // Compute \sum_{i=1,k=1}{a[i,k]+s[i,k]} in equation (2)
    cblas_dcopy(N2, ap->similarity, 1, mat, 1);
    cblas_daxpy(N2, 1.0, ap->A, 1, mat, 1);

    // Sort each row of the matrix
    for (int i = 0; i < ap->N; i++) {
      qsort(&mat[i * ap->N], ap->N, sizeof(double), dcmp_sort_des);
    }

    // Update the responsibility and availability for each linked data point
    // pair [i,j].
    for (int i = 0; i < ap->N; i++) {
      StaticLink *link = ap->_links[i];
      if (link == NULL) {
        continue;
      }

      int i0 = i * ap->N;

      for (int idx = 0; idx < link->count; idx++) {
        int j = link->edge[idx];
        int ij = i * ap->N + j;
        int jj = j * ap->N + j;

        double v = mat[i0];
        if (dcmp(v, ap->A[ij]) == 0) {
          v = mat[i0 + 1];
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
  }

  free(alp);
  free(rho);
  free(mat);
}

int main(int argc, const char *argv[]) {

  double *p = NULL;
  ALLOCATE(p, 5, double);

  return 0;
}
