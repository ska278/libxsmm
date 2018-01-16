/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_FSSPMDM_H
#define LIBXSMM_FSSPMDM_H

#include "libxsmm_typedefs.h"


/** Opaque types for fsspmdm */
typedef struct LIBXSMM_RETARGETABLE libxsmm_dfsspmdm libxsmm_dfsspmdm;
typedef struct LIBXSMM_RETARGETABLE libxsmm_sfsspmdm libxsmm_sfsspmdm;

LIBXSMM_API libxsmm_dfsspmdm* libxsmm_dfsspmdm_create( libxsmm_blasint M,   libxsmm_blasint   N, libxsmm_blasint   K,
                                                       libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
                                                       const double alpha, const double beta,
                                                       const double* a_dense );

LIBXSMM_API void libxsmm_dfsspmdm_execute( const libxsmm_dfsspmdm* handle, const double* B, double* C );

LIBXSMM_API void libxsmm_dfsspmdm_destroy( libxsmm_dfsspmdm* handle );

LIBXSMM_API libxsmm_sfsspmdm* libxsmm_sfsspmdm_create( libxsmm_blasint M,   libxsmm_blasint   N, libxsmm_blasint   K,
                                                       libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
                                                       const float alpha, const float beta,
                                                       const float* a_dense );

LIBXSMM_API void libxsmm_sfsspmdm_execute( const libxsmm_sfsspmdm* handle, const float* B, float* C );

LIBXSMM_API void libxsmm_sfsspmdm_destroy( libxsmm_sfsspmdm* handle );

#endif /*LIBXSMM_FSSPMDM_H*/

