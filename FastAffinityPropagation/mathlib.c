//
//  mathlib.c
//  FastAffinityPropagation
//
//  Created by Bismarrck on 1/31/15.
//  Copyright (c) 2015 Nexd. All rights reserved.
//

#include "mathlib.h"
#include <float.h>

/**
 * @function dcmp
 * This function safely compares two double precision floats and returns 1 if
 * the former is larger, 0 if they are equal or -1 if the former is smaller.
 */
int dcmp(double d1, double d2) {
  double df = d1 - d2;
  if (df > 1.0e-12) {
    return 1;
  } else if (df < -1.0e-12) {
    return -1;
  } else {
    return 0;
  }
}

/**
 * @function dmax
 * Return the larger one given two double precision floats.
 */
double dmax(double d1, double d2) {
  int vs = dcmp(d1, d2);
  if (vs >= 0) {
    return d1;
  } else {
    return d2;
  }
}

/**
 * @function dmin
 * Return the smaller one given two double precision floats.
 */
double dmin(double d1, double d2) {
  int vs = dcmp(d1, d2);
  if (vs <= 0) {
    return d1;
  } else {
    return d2;
  }
}

/**
 * @function vdmax
 * Return the maximum and the position of the maximum given a double precision
 * vector.
 */
double vdmax(double *restrict v, const int n, int *i) {
  int k = 0;
  double maximum = -FLT_MAX;
  while (k < n) {
    if (dcmp(maximum, v[k]) < 0) {
      maximum = v[k];
      if (i) {
        *i = k;
      }
    }
    k++;
  }
  return maximum;
}

/**
 * @function vdmin
 * Return the minimum and the position of the minimum given a double precision
 * vector.
 */
double vdmin(double *restrict v, const int n, int *i) {
  int k = 0;
  double minimum = FLT_MAX;
  while (k < n) {
    if (dcmp(minimum, v[k]) > 0) {
      minimum = v[k];
      if (i) {
        *i = k;
      }
    }
    k++;
  }
  return minimum;
}
