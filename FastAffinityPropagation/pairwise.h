//
//  pairwise.h
//  FastAffinityPropagation
//
//  Created by Bismarrck on 1/31/15.
//  Copyright (c) 2015 Nexd. All rights reserved.
//

#ifndef __FastAffinityPropagation__PAIRWISE__
#define __FastAffinityPropagation__PAIRWISE__

#include <stdbool.h>

double *pairwise_distance_matrix(double *points, int npoint, int dim,
                                 bool squared);

#endif /* defined(__FastAffinityPropagation__pairwise__) */
