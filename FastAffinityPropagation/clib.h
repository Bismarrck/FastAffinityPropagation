//
//  clib.h
//  FastAffinityPropagation
//
//  Created by Bismarrck on 1/31/15.
//  Copyright (c) 2015 Nexd. All rights reserved.
//

#ifndef __FastAffinityPropagation__CLIB__
#define __FastAffinityPropagation__CLIB__

#define ALLOCATE(p, n, type) (p = calloc(sizeof(type), n));
#define DEALLOCATE(p)                                                          \
  if (p != NULL) {                                                             \
    free(p);                                                                   \
  }

#endif /* defined(__FastAffinityPropagation__CLIB__) */
