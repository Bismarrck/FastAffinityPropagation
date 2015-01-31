//
//  pairwise.c
//  FastAffinityPropagation
//
//  Created by Bismarrck on 1/31/15.
//  Copyright (c) 2015 Nexd. All rights reserved.
//

#include "pairwise.h"
#include "clib.h"
#include <string.h>
#include <Accelerate/Accelerate.h>

double *pairwise_distance_matrix(double *points, int npoint, int dim,
                                 bool squared) {
  unsigned int N2 = npoint * npoint;

  double *dist;
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
