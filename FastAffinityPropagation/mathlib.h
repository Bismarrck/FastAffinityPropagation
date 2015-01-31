//
//  mathlib.h
//  FastAffinityPropagation
//
//  Created by Bismarrck on 1/31/15.
//  Copyright (c) 2015 Nexd. All rights reserved.
//

#ifndef __FastAffinityPropagation__MATHLIB__
#define __FastAffinityPropagation__MATHLIB__

/**
 * @function dcmp
 * This function safely compares two double precision floats and returns 1 if
 * the former is larger, 0 if they are equal or -1 if the former is smaller.
 */
extern int dcmp(double d1, double d2);

/**
 * @function dmax
 * Return the larger one given two double precision floats.
 */
extern double dmax(double d1, double d2);
/**
 * @function dmin
 * Return the smaller one given two double precision floats.
 */
extern double dmin(double d1, double d2);

/**
 * @function vdmax
 * Return the maximum and the position of the maximum given a double precision
 * vector.
 */
extern double vdmax(double *restrict v, const int n, int *i);

/**
 * @function vdmin
 * Return the minimum and the position of the minimum given a double precision
 * vector.
 */
extern double vdmin(double *restrict v, const int n, int *i);

#endif /* defined(__FastAffinityPropagation__MATHLIB__) */
