/******************************************************************************
** Copyright (c) 2015-2018, Intel Corporation                                **
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
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_GEMM_H
#define LIBXSMM_GEMM_H

#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if !defined(LIBXSMM_GEMM_WRAP_DYNAMIC) && defined(LIBXSMM_BUILD) && \
  (!defined(__BLAS) || (0 != __BLAS)) && defined(__GNUC__) && \
  !(defined(__APPLE__) && defined(__MACH__) && LIBXSMM_VERSION3(6, 1, 0) >= \
    LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) && \
  !defined(_WIN32) && !defined(__CYGWIN__)
# include <dlfcn.h>
# define LIBXSMM_GEMM_WRAP_DYNAMIC
#endif
#include <limits.h>
#include <stdio.h>
#include <math.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_GEMM_MMBATCH) && defined(LIBXSMM_BUILD) && \
    (defined(LIBXSMM_CONFIG_WRAP) && 0 != (LIBXSMM_CONFIG_WRAP)) && \
    (defined(LIBXSMM_GEMM_WRAP_STATIC) || defined(LIBXSMM_GEMM_WRAP_DYNAMIC) || \
    !defined(NDEBUG) || defined(_WIN32)) /* debug purpose */
# define LIBXSMM_GEMM_MMBATCH
#endif

/** Undefine (disarm) MKL's DIRECT_CALL macros. */
#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# if defined(sgemm_)
#   undef sgemm_
# endif
# if defined(dgemm_)
#   undef dgemm_
# endif
#endif

#if !defined(LIBXSMM_GEMM_COLLAPSE)
# define LIBXSMM_GEMM_COLLAPSE 2
#endif

#if !defined(LIBXSMM_GEMM_BATCHSCALE)
# define LIBXSMM_GEMM_BATCHSCALE 1.5
#endif

#if !defined(LIBXSMM_NO_BLAS)
# if !defined(__BLAS) || (0 != __BLAS)
#   define LIBXSMM_NO_BLAS 0
# else
#   define LIBXSMM_NO_BLAS 1
# endif
#endif

#define LIBXSMM_GEMM_NO_BYPASS(FLAGS, ALPHA, BETA) ( \
  0 == ((FLAGS) & (LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B)) && \
        (LIBXSMM_FEQ(1, ALPHA) /*|| LIBXSMM_FEQ(-1, ALPHA)*/) && \
        (LIBXSMM_FEQ(1, BETA) || LIBXSMM_FEQ(0, BETA)))

#if !defined(LIBXSMM_GEMM_TILED_INNER_FALLBACK)
# define LIBXSMM_GEMM_TILED_INNER_FALLBACK
#endif
#if defined(LIBXSMM_GEMM_TILED_INNER_FALLBACK)
# define LIBXSMM_GEMM_TILED_FALLBACK_CHECK(CONDITION) if (CONDITION)
# define LIBXSMM_GEMM_TILED_FALLBACK(TYPE, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) else { \
    LIBXSMM_FALLBACK0(TYPE, libxsmm_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  }
#else
# define LIBXSMM_GEMM_TILED_FALLBACK_CHECK(CONDITION)
# define LIBXSMM_GEMM_TILED_FALLBACK(TYPE, FLAGS, TILE_M, TILE_N, TILE_K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif

#define LIBXSMM_GEMM_TILED_KERNEL(KERNEL_INNER_BETA1, TYPE, FLAGS, POS_I, POS_J, MAX_K, TILE_M, TILE_N, TILE_K, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const libxsmm_blasint libxsmm_tiled_xgemm_kernel_tm_ = LIBXSMM_MIN(TILE_M, (M) - (POS_I)); \
  const libxsmm_blasint libxsmm_tiled_xgemm_kernel_tn_ = LIBXSMM_MIN(TILE_N, (N) - (POS_J)); \
  const libxsmm_blasint libxsmm_tiled_xgemm_kernel_tk_ = ((TILE_K) <= (K) ? (TILE_K) : ((K) - (MAX_K))); \
  const TYPE* libxsmm_tiled_xgemm_kernel_ia_ = (A) + (POS_I); \
  const TYPE* libxsmm_tiled_xgemm_kernel_ib_ = (B) + (POS_J) * (LDB); \
  const TYPE* libxsmm_tiled_xgemm_kernel_pa_ = libxsmm_tiled_xgemm_kernel_ia_ + (libxsmm_tiled_xgemm_kernel_tk_) * (LDA); \
  const TYPE* libxsmm_tiled_xgemm_kernel_pb_ = libxsmm_tiled_xgemm_kernel_ib_ + (libxsmm_tiled_xgemm_kernel_tk_); \
  TYPE *const libxsmm_tiled_xgemm_kernel_ic_ = (C) + (POS_J) * (LDC) + (POS_I), libxsmm_tiled_xgemm_kernel_beta_ = BETA; \
  libxsmm_gemm_descriptor libxsmm_tiled_xgemm_kernel_desc_ = { 0 }; \
  libxsmm_xmmfunction libxsmm_gemm_tiled_kernel_ = { 0 }; \
  libxsmm_blasint libxsmm_tiled_xgemm_kernel_k_ = 0; \
  assert(0 != (A) && 0 != (B) && 0 != (C)); \
  if (((TILE_M) == libxsmm_tiled_xgemm_kernel_tm_) && ((TILE_N) == libxsmm_tiled_xgemm_kernel_tn_) && ((TILE_K) == libxsmm_tiled_xgemm_kernel_tk_)) { \
    if (libxsmm_tiled_xgemm_kernel_k_ < (MAX_K)) { /* peel */ \
      LIBXSMM_GEMM_DESCRIPTOR(libxsmm_tiled_xgemm_kernel_desc_, LIBXSMM_GEMM_PRECISION(TYPE), FLAGS, TILE_M, TILE_N, TILE_K, \
        LDA, LDB, LDC, ALPHA, BETA, libxsmm_gemm_tiled_prefetch); \
      libxsmm_gemm_tiled_kernel_ = libxsmm_xmmdispatch(&libxsmm_tiled_xgemm_kernel_desc_); \
      LIBXSMM_GEMM_TILED_FALLBACK_CHECK(0 != libxsmm_gemm_tiled_kernel_.LIBXSMM_TPREFIX(TYPE, mm)) \
      { \
        LIBXSMM_MMCALL_PRF(libxsmm_gemm_tiled_kernel_.LIBXSMM_TPREFIX(TYPE, mm), \
          libxsmm_tiled_xgemm_kernel_ia_, libxsmm_tiled_xgemm_kernel_ib_, libxsmm_tiled_xgemm_kernel_ic_, \
          libxsmm_tiled_xgemm_kernel_pa_, libxsmm_tiled_xgemm_kernel_pb_, libxsmm_tiled_xgemm_kernel_ic_); \
      } \
      LIBXSMM_GEMM_TILED_FALLBACK(TYPE, FLAGS, TILE_M, TILE_N, TILE_K, \
        ALPHA, libxsmm_tiled_xgemm_kernel_ia_, LDA, libxsmm_tiled_xgemm_kernel_ib_, LDB, \
         BETA, libxsmm_tiled_xgemm_kernel_ic_, LDC); \
      libxsmm_tiled_xgemm_kernel_ia_ = libxsmm_tiled_xgemm_kernel_pa_; \
      libxsmm_tiled_xgemm_kernel_ib_ = libxsmm_tiled_xgemm_kernel_pb_; \
      libxsmm_tiled_xgemm_kernel_pa_ += (TILE_K) * (LDA); \
      libxsmm_tiled_xgemm_kernel_pb_ += TILE_K; \
      libxsmm_tiled_xgemm_kernel_k_ = TILE_K; \
      libxsmm_tiled_xgemm_kernel_beta_ = 1; \
    } \
    for (; libxsmm_tiled_xgemm_kernel_k_ < (MAX_K); libxsmm_tiled_xgemm_kernel_k_ += TILE_K) { /* inner */ \
      LIBXSMM_MMCALL_PRF((KERNEL_INNER_BETA1).LIBXSMM_TPREFIX(TYPE, mm), \
        libxsmm_tiled_xgemm_kernel_ia_, libxsmm_tiled_xgemm_kernel_ib_, libxsmm_tiled_xgemm_kernel_ic_, \
        libxsmm_tiled_xgemm_kernel_pa_, libxsmm_tiled_xgemm_kernel_pb_, libxsmm_tiled_xgemm_kernel_ic_); \
      libxsmm_tiled_xgemm_kernel_ia_ = libxsmm_tiled_xgemm_kernel_pa_; \
      libxsmm_tiled_xgemm_kernel_ib_ = libxsmm_tiled_xgemm_kernel_pb_; \
      libxsmm_tiled_xgemm_kernel_pa_ += (TILE_K) * (LDA); \
      libxsmm_tiled_xgemm_kernel_pb_ += TILE_K; \
    } \
  } \
  if (libxsmm_tiled_xgemm_kernel_k_ < (K)) { /* remainder */ \
    LIBXSMM_GEMM_DESCRIPTOR(libxsmm_tiled_xgemm_kernel_desc_, LIBXSMM_GEMM_PRECISION(TYPE), FLAGS, \
      libxsmm_tiled_xgemm_kernel_tm_, libxsmm_tiled_xgemm_kernel_tn_, (K) - libxsmm_tiled_xgemm_kernel_k_, \
      LDA, LDB, LDC, ALPHA, libxsmm_tiled_xgemm_kernel_beta_, libxsmm_gemm_tiled_prefetch); \
    libxsmm_gemm_tiled_kernel_ = libxsmm_xmmdispatch(&libxsmm_tiled_xgemm_kernel_desc_); \
    LIBXSMM_GEMM_TILED_FALLBACK_CHECK(0 != libxsmm_gemm_tiled_kernel_.LIBXSMM_TPREFIX(TYPE, mm)) \
    { \
      LIBXSMM_MMCALL_PRF(libxsmm_gemm_tiled_kernel_.LIBXSMM_TPREFIX(TYPE, mm), \
        libxsmm_tiled_xgemm_kernel_ia_, libxsmm_tiled_xgemm_kernel_ib_, libxsmm_tiled_xgemm_kernel_ic_, \
        libxsmm_tiled_xgemm_kernel_pa_, libxsmm_tiled_xgemm_kernel_pb_, libxsmm_tiled_xgemm_kernel_ic_); \
    } \
    LIBXSMM_GEMM_TILED_FALLBACK(TYPE, FLAGS, \
      libxsmm_tiled_xgemm_kernel_tm_, libxsmm_tiled_xgemm_kernel_tn_, (K) - libxsmm_tiled_xgemm_kernel_k_, \
      ALPHA, libxsmm_tiled_xgemm_kernel_ia_, LDA, libxsmm_tiled_xgemm_kernel_ib_, LDB, \
      libxsmm_tiled_xgemm_kernel_beta_, libxsmm_tiled_xgemm_kernel_ic_, LDC); \
  } \
}

#if defined(NDEBUG)
# define LIBXSMM_TILED_XGEMM_FALLBACK_PRINT(TYPE, FLAGS, MM, NN, KK, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
# define LIBXSMM_TILED_XGEMM_FALLBACK_PRINT(TYPE, FLAGS, MM, NN, KK, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    if (((INT_MAX) - 1) != libxsmm_verbosity \
      && (unsigned int)LIBXSMM_ABS(libxsmm_verbosity) > libxsmm_update_mmstatistic(LIBXSMM_GEMM_PRECISION(TYPE), MM, NN, KK, 1/*try*/, 0)) \
    { \
      const char libxsmm_tiled_xgemm_transa_ = (char)(0 == ((FLAGS) & LIBXSMM_GEMM_FLAG_TRANS_A) ? 'N' : 'T'); \
      const char libxsmm_tiled_xgemm_transb_ = (char)(0 == ((FLAGS) & LIBXSMM_GEMM_FLAG_TRANS_B) ? 'N' : 'T'); \
      const TYPE libxsmm_tiled_xgemm_alpha_ = (TYPE)(ALPHA), libxsmm_tiled_xgemm_beta_ = (TYPE)(BETA); \
      if (0 < libxsmm_verbosity) { /* print fallback */ \
        LIBXSMM_FLOCK(stderr); \
        fprintf(stderr, "LIBXSMM FALLBACK: "); libxsmm_gemm_print(stderr, LIBXSMM_GEMM_PRECISION(TYPE), \
          &libxsmm_tiled_xgemm_transa_, &libxsmm_tiled_xgemm_transb_, &(MM), &(NN), &(KK), \
          &libxsmm_tiled_xgemm_alpha_, 0/*A*/, &(LDA), 0/*B*/, &(LDB), &libxsmm_tiled_xgemm_beta_, 0/*C*/, &(LDC)); \
        fprintf(stderr, "\n"); \
        LIBXSMM_FUNLOCK(stderr); \
      } \
      else { /* dump matrices */ \
        libxsmm_gemm_print(0, LIBXSMM_GEMM_PRECISION(TYPE), \
          &libxsmm_tiled_xgemm_transa_, &libxsmm_tiled_xgemm_transb_, &(MM), &(NN), &(KK), \
          &libxsmm_tiled_xgemm_alpha_, A, &(LDA), B, &(LDB), &libxsmm_tiled_xgemm_beta_, C, &(LDC)); \
      } \
    }
#endif

#define LIBXSMM_TILED_XGEMM(PARALLEL, LOOP_START, KERNEL_START, SYNC, \
  MIN_TASKS, OVERHEAD, NT, TYPE, FLAGS, TILE_M, TILE_N, TILE_K, MM, NN, KK, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
{ /* use NN, etc. rather than N due to below char. constant */ \
  libxsmm_blasint libxsmm_tiled_xgemm_tm_ = 0, libxsmm_tiled_xgemm_tn_ = 0, libxsmm_tiled_xgemm_tk_ = 0; \
  libxsmm_blasint libxsmm_tiled_xgemm_num_m_ = 0, libxsmm_tiled_xgemm_num_n_ = 0, libxsmm_tiled_xgemm_num_k_ = 0; \
  libxsmm_xmmfunction libxsmm_tiled_xgemm_kernel_ = { 0 }; \
  if (0 != LIBXSMM_GEMM_NO_BYPASS(FLAGS, ALPHA, BETA)) { \
    assert(0 != (TILE_M) && 0 != (TILE_N) && 0 != (TILE_K)); \
    libxsmm_tiled_xgemm_num_m_ = ((MM) + (TILE_M) - 1) / (TILE_M); \
    libxsmm_tiled_xgemm_num_n_ = ((NN) + (TILE_N) - 1) / (TILE_N); \
    libxsmm_tiled_xgemm_num_k_ = ((KK) + (TILE_K) - 1) / (TILE_K); \
    { /* opening scope for additional variable declarations */ \
      const libxsmm_blasint libxsmm_tiled_xgemm_num_t_ = (OVERHEAD(NT) < libxsmm_tiled_xgemm_num_k_ && 1 < (LIBXSMM_GEMM_COLLAPSE)) \
        ? (libxsmm_tiled_xgemm_num_m_ * libxsmm_tiled_xgemm_num_n_) \
        : (libxsmm_tiled_xgemm_num_n_ <= libxsmm_tiled_xgemm_num_m_ ? libxsmm_tiled_xgemm_num_m_ : libxsmm_tiled_xgemm_num_n_); \
      const libxsmm_blasint libxsmm_tiled_xgemm_min_ntasks_ = MIN_TASKS(NT); \
      libxsmm_gemm_descriptor libxsmm_tiled_xgemm_desc_ = { 0 }; \
      if (libxsmm_tiled_xgemm_min_ntasks_ <= libxsmm_tiled_xgemm_num_t_) { /* ensure enough parallel slack */ \
        assert(0 != libxsmm_tiled_xgemm_num_m_ && 0 != libxsmm_tiled_xgemm_num_n_); \
        libxsmm_tiled_xgemm_tm_ = (MM) / libxsmm_tiled_xgemm_num_m_; \
        libxsmm_tiled_xgemm_tn_ = (NN) / libxsmm_tiled_xgemm_num_n_; \
      } \
      else if (OVERHEAD(NT) < libxsmm_tiled_xgemm_num_k_) { \
        const libxsmm_blasint libxsmm_tiled_xgemm_ratio_ = LIBXSMM_SQRT2(libxsmm_tiled_xgemm_min_ntasks_ / libxsmm_tiled_xgemm_num_t_); \
        libxsmm_tiled_xgemm_tn_ = (libxsmm_tiled_xgemm_num_n_ * libxsmm_tiled_xgemm_ratio_); \
        libxsmm_tiled_xgemm_tm_ = (libxsmm_tiled_xgemm_min_ntasks_ + libxsmm_tiled_xgemm_tn_ - 1) / libxsmm_tiled_xgemm_tn_; \
      } \
      else if (libxsmm_tiled_xgemm_num_n_ <= libxsmm_tiled_xgemm_num_m_) { \
        libxsmm_tiled_xgemm_tm_ = ((MM) + libxsmm_tiled_xgemm_min_ntasks_ - 1) / libxsmm_tiled_xgemm_min_ntasks_; \
        libxsmm_tiled_xgemm_tn_ = TILE_N; \
      } \
      else { \
        libxsmm_tiled_xgemm_tm_ = TILE_M; \
        libxsmm_tiled_xgemm_tn_ = ((NN) + libxsmm_tiled_xgemm_min_ntasks_ - 1) / libxsmm_tiled_xgemm_min_ntasks_; \
      } \
      libxsmm_tiled_xgemm_tk_ = TILE_K; \
      { /* adjust for non-square operand shapes */ \
        float libxsmm_tiled_xgemm_rm_ = 1.f, libxsmm_tiled_xgemm_rn_ = ((float)(NN)) / (MM), libxsmm_tiled_xgemm_rk_ = ((float)(KK)) / (MM); \
        if (1.f < libxsmm_tiled_xgemm_rn_) { libxsmm_tiled_xgemm_rm_ /= libxsmm_tiled_xgemm_rn_; libxsmm_tiled_xgemm_rn_ = 1.f; libxsmm_tiled_xgemm_rk_ /= libxsmm_tiled_xgemm_rn_; } \
        if (1.f < libxsmm_tiled_xgemm_rk_) { libxsmm_tiled_xgemm_rm_ /= libxsmm_tiled_xgemm_rk_; libxsmm_tiled_xgemm_rn_ /= libxsmm_tiled_xgemm_rk_; libxsmm_tiled_xgemm_rk_ = 1.f; } \
        libxsmm_tiled_xgemm_tm_ = (libxsmm_blasint)(libxsmm_tiled_xgemm_tm_ * libxsmm_tiled_xgemm_rm_ /*+ 0.5f*/); \
        libxsmm_tiled_xgemm_tn_ = (libxsmm_blasint)(libxsmm_tiled_xgemm_tn_ * libxsmm_tiled_xgemm_rn_ /*+ 0.5f*/); \
        libxsmm_tiled_xgemm_tk_ = (libxsmm_blasint)(libxsmm_tiled_xgemm_tk_ * libxsmm_tiled_xgemm_rk_ /*+ 0.5f*/); \
        libxsmm_tiled_xgemm_tm_ = (libxsmm_blasint)(1ULL << LIBXSMM_LOG2(libxsmm_tiled_xgemm_tm_)); \
        libxsmm_tiled_xgemm_tn_ = (libxsmm_blasint)(1ULL << LIBXSMM_LOG2(libxsmm_tiled_xgemm_tn_)); \
        libxsmm_tiled_xgemm_tk_ = (libxsmm_blasint)(1ULL << LIBXSMM_LOG2(libxsmm_tiled_xgemm_tk_)); \
        libxsmm_tiled_xgemm_tm_ = LIBXSMM_CLMP(libxsmm_tiled_xgemm_tm_, 8, MM); \
        libxsmm_tiled_xgemm_tn_ = LIBXSMM_CLMP(libxsmm_tiled_xgemm_tn_, 8, NN); \
        libxsmm_tiled_xgemm_tk_ = LIBXSMM_CLMP(libxsmm_tiled_xgemm_tk_, 8, KK); \
      } \
      LIBXSMM_GEMM_DESCRIPTOR(libxsmm_tiled_xgemm_desc_, LIBXSMM_GEMM_PRECISION(TYPE), FLAGS, \
        libxsmm_tiled_xgemm_tm_, libxsmm_tiled_xgemm_tn_, libxsmm_tiled_xgemm_tk_, \
        LDA, LDB, LDC, ALPHA, 1/*beta*/, libxsmm_gemm_tiled_prefetch); \
      libxsmm_tiled_xgemm_kernel_ = libxsmm_xmmdispatch(&libxsmm_tiled_xgemm_desc_); \
    } \
  } \
  if (0 != libxsmm_tiled_xgemm_kernel_.LIBXSMM_TPREFIX(TYPE, mm)) { assert(0 != libxsmm_tiled_xgemm_tk_); { \
    const int libxsmm_tiled_xgemm_amortized_ = (OVERHEAD(NT) * libxsmm_tiled_xgemm_tn_) < (KK); \
    const libxsmm_blasint libxsmm_tiled_xgemm_max_k_ = ((KK) / libxsmm_tiled_xgemm_tk_) * libxsmm_tiled_xgemm_tk_; \
    libxsmm_blasint libxsmm_tiled_xgemm_m_ = MM, libxsmm_tiled_xgemm_n_ = NN; \
    libxsmm_blasint libxsmm_tiled_xgemm_dm_ = libxsmm_tiled_xgemm_tm_, libxsmm_tiled_xgemm_dn_ = libxsmm_tiled_xgemm_tn_; \
    libxsmm_blasint libxsmm_tiled_xgemm_swap_ = 0; \
    if ((1 == (LIBXSMM_GEMM_COLLAPSE) || 0 == libxsmm_tiled_xgemm_amortized_) && \
      libxsmm_tiled_xgemm_tn_ * (MM) < libxsmm_tiled_xgemm_tm_ * (NN)) /* approx. of num_m < num_n */ \
    { \
      libxsmm_tiled_xgemm_swap_ = libxsmm_tiled_xgemm_dm_; libxsmm_tiled_xgemm_dm_ = libxsmm_tiled_xgemm_dn_; libxsmm_tiled_xgemm_dn_ = libxsmm_tiled_xgemm_swap_; \
      libxsmm_tiled_xgemm_swap_ = libxsmm_tiled_xgemm_m_; libxsmm_tiled_xgemm_m_ = libxsmm_tiled_xgemm_n_; libxsmm_tiled_xgemm_n_ = libxsmm_tiled_xgemm_swap_; \
    } \
    if (0 != libxsmm_tiled_xgemm_amortized_) { /* amortized overhead */ \
      PARALLEL \
      { \
        libxsmm_blasint libxsmm_tiled_xgemm_i_, libxsmm_tiled_xgemm_j_ = 0; \
        LOOP_START(LIBXSMM_GEMM_COLLAPSE) \
        for (libxsmm_tiled_xgemm_i_ = 0; libxsmm_tiled_xgemm_i_ < libxsmm_tiled_xgemm_m_; libxsmm_tiled_xgemm_i_ += libxsmm_tiled_xgemm_dm_) { \
          for (libxsmm_tiled_xgemm_j_ = 0; libxsmm_tiled_xgemm_j_ < libxsmm_tiled_xgemm_n_; libxsmm_tiled_xgemm_j_ += libxsmm_tiled_xgemm_dn_) { \
            KERNEL_START(firstprivate(libxsmm_tiled_xgemm_i_, libxsmm_tiled_xgemm_j_)) \
            LIBXSMM_GEMM_TILED_KERNEL(libxsmm_tiled_xgemm_kernel_, TYPE, FLAGS, \
              0 == libxsmm_tiled_xgemm_swap_ ? libxsmm_tiled_xgemm_i_ : libxsmm_tiled_xgemm_j_, \
              0 == libxsmm_tiled_xgemm_swap_ ? libxsmm_tiled_xgemm_j_ : libxsmm_tiled_xgemm_i_, \
              libxsmm_tiled_xgemm_max_k_, libxsmm_tiled_xgemm_tm_, libxsmm_tiled_xgemm_tn_, libxsmm_tiled_xgemm_tk_, \
              MM, NN, KK, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
          } \
        } \
      } \
    } \
    else { \
      PARALLEL \
      { \
        libxsmm_blasint libxsmm_tiled_xgemm_i_, libxsmm_tiled_xgemm_j_ = 0; \
        LOOP_START(1/*COLLAPSE*/) \
        for (libxsmm_tiled_xgemm_i_ = 0; libxsmm_tiled_xgemm_i_ < libxsmm_tiled_xgemm_m_; libxsmm_tiled_xgemm_i_ += libxsmm_tiled_xgemm_dm_) { \
          KERNEL_START(firstprivate(libxsmm_tiled_xgemm_i_)) \
          for (libxsmm_tiled_xgemm_j_ = 0; libxsmm_tiled_xgemm_j_ < libxsmm_tiled_xgemm_n_; libxsmm_tiled_xgemm_j_ += libxsmm_tiled_xgemm_dn_) { \
            LIBXSMM_GEMM_TILED_KERNEL(libxsmm_tiled_xgemm_kernel_, TYPE, FLAGS, \
              0 == libxsmm_tiled_xgemm_swap_ ? libxsmm_tiled_xgemm_i_ : libxsmm_tiled_xgemm_j_, \
              0 == libxsmm_tiled_xgemm_swap_ ? libxsmm_tiled_xgemm_j_ : libxsmm_tiled_xgemm_i_, \
              libxsmm_tiled_xgemm_max_k_, libxsmm_tiled_xgemm_tm_, libxsmm_tiled_xgemm_tn_, libxsmm_tiled_xgemm_tk_, \
              MM, NN, KK, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
          } \
        } \
      } \
    } \
    SYNC \
  }} \
  else { /* fall-back */ \
    assert(0 == LIBXSMM_NO_BLAS); \
    LIBXSMM_FALLBACK1(TYPE, libxsmm_blasint, FLAGS, MM, NN, KK, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
    LIBXSMM_TILED_XGEMM_FALLBACK_PRINT(TYPE, FLAGS, MM, NN, KK, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}

#if (!defined(__BLAS) || (0 != __BLAS))
# define LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, CALLER, SYMBOL) if (0 == (ORIGINAL)) { \
    union { const void* pv; LIBXSMM_GEMMFUNCTION_TYPE(TYPE) pf; \
      void (*sf)(LIBXSMM_GEMM_CONST char*, LIBXSMM_GEMM_CONST char*, \
        LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, \
        LIBXSMM_GEMM_CONST float*, LIBXSMM_GEMM_CONST float*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST float*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, \
        LIBXSMM_GEMM_CONST float*, float*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*); \
      void (*df)(LIBXSMM_GEMM_CONST char*, LIBXSMM_GEMM_CONST char*, \
        LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, \
        LIBXSMM_GEMM_CONST double*, LIBXSMM_GEMM_CONST double*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST double*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, \
        LIBXSMM_GEMM_CONST double*, double*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*); \
    } libxsmm_gemm_wrapper_blas_; \
    libxsmm_gemm_wrapper_blas_.LIBXSMM_TPREFIX(TYPE,f) = (SYMBOL); \
    if (libxsmm_gemm_wrapper_blas_.pv != (CALLER)) { \
      /*LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_STORE, LIBXSMM_BITS)(&(ORIGINAL), libxsmm_gemm_wrapper_blas_.pf, LIBXSMM_ATOMIC_RELAXED);*/ \
      ORIGINAL = libxsmm_gemm_wrapper_blas_.pf; \
    } \
  }
# define LIBXSMM_GEMV_WRAPPER_BLAS(TYPE, ORIGINAL, CALLER, SYMBOL) if (0 == (ORIGINAL)) { \
    union { const void* pv; LIBXSMM_GEMVFUNCTION_TYPE(TYPE) pf; \
      void (*sf)(LIBXSMM_GEMM_CONST char*, \
        LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, \
        LIBXSMM_GEMM_CONST float*, LIBXSMM_GEMM_CONST float*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST float*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, \
        LIBXSMM_GEMM_CONST float*, float*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*); \
      void (*df)(LIBXSMM_GEMM_CONST char*, \
        LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, \
        LIBXSMM_GEMM_CONST double*, LIBXSMM_GEMM_CONST double*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST double*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, \
        LIBXSMM_GEMM_CONST double*, double*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*); \
    } libxsmm_gemv_wrapper_blas_; \
    libxsmm_gemv_wrapper_blas_.LIBXSMM_TPREFIX(TYPE,f) = (SYMBOL); \
    if (libxsmm_gemv_wrapper_blas_.pv != (CALLER)) { \
      /*LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_STORE, LIBXSMM_BITS)(&(ORIGINAL), libxsmm_gemv_wrapper_blas_.pf, LIBXSMM_ATOMIC_RELAXED);*/ \
      ORIGINAL = libxsmm_gemv_wrapper_blas_.pf; \
    } \
  }
#else
# define LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, CALLER, SYMBOL) LIBXSMM_UNUSED(CALLER)
# define LIBXSMM_GEMV_WRAPPER_BLAS(TYPE, ORIGINAL, CALLER, SYMBOL) LIBXSMM_UNUSED(CALLER)
#endif

#if defined(LIBXSMM_GEMM_WRAP) && defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT) && \
  !(defined(__APPLE__) && defined(__MACH__) /*&& defined(__clang__)*/) && !defined(__CYGWIN__)
# if (2 != (LIBXSMM_GEMM_WRAP)) /* SGEMM and DGEMM */
#   define LIBXSMM_GEMM_WRAPPER_STATIC(TYPE, ORIGINAL, CALLER) LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, CALLER, \
      LIBXSMM_FSYMBOL(LIBXSMM_CONCATENATE(__real_, LIBXSMM_TPREFIX(TYPE, gemm))))
#   define LIBXSMM_GEMV_WRAPPER_STATIC(TYPE, ORIGINAL, CALLER) LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, CALLER, \
      LIBXSMM_FSYMBOL(LIBXSMM_CONCATENATE(__real_, LIBXSMM_TPREFIX(TYPE, gemv))))
# else /* DGEMM only */
#   define LIBXSMM_GEMM_WRAPPER_STATIC(TYPE, ORIGINAL, CALLER) if (0 != LIBXSMM_EQUAL(TYPE, double)) { \
      LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, CALLER, LIBXSMM_FSYMBOL(__real_dgemm)) \
    }
#   define LIBXSMM_GEMV_WRAPPER_STATIC(TYPE, ORIGINAL, CALLER) if (0 != LIBXSMM_EQUAL(TYPE, double)) { \
      LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, CALLER, LIBXSMM_FSYMBOL(__real_dgemv)) \
    }
# endif
# define LIBXSMM_GEMM_WRAP_STATIC
#else
# define LIBXSMM_GEMM_WRAPPER_STATIC(TYPE, ORIGINAL, CALLER)
# define LIBXSMM_GEMV_WRAPPER_STATIC(TYPE, ORIGINAL, CALLER)
#endif

#if defined(LIBXSMM_GEMM_WRAP_DYNAMIC)
# define LIBXSMM_GEMM_WRAPPER_DYNAMIC(TYPE, ORIGINAL, CALLER) \
    if (0 == (ORIGINAL)) { \
      union { const void* pv; LIBXSMM_GEMMFUNCTION_TYPE(TYPE) pf; } libxsmm_gemm_wrapper_dynamic_ = { 0 }; \
      dlerror(); /* clear an eventual error status */ \
      libxsmm_gemm_wrapper_dynamic_.pv = dlsym(RTLD_NEXT, LIBXSMM_STRINGIFY(LIBXSMM_GEMM_SYMBOL(TYPE))); \
      if (libxsmm_gemm_wrapper_dynamic_.pv != (CALLER)) { \
        /*LIBXSMM_ATOMIC_STORE(&(ORIGINAL), libxsmm_gemm_wrapper_dynamic_.pf, LIBXSMM_ATOMIC_RELAXED);*/ \
        ORIGINAL = libxsmm_gemm_wrapper_dynamic_.pf; \
      } \
      LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, CALLER, LIBXSMM_GEMM_SYMBOL(TYPE)); \
    }
# define LIBXSMM_GEMV_WRAPPER_DYNAMIC(TYPE, ORIGINAL, CALLER) \
    if (0 == (ORIGINAL)) { \
      union { const void* pv; LIBXSMM_GEMVFUNCTION_TYPE(TYPE) pf; } libxsmm_gemv_wrapper_dynamic_ = { 0 }; \
      dlerror(); /* clear an eventual error status */ \
      libxsmm_gemv_wrapper_dynamic_.pv = dlsym(RTLD_NEXT, LIBXSMM_STRINGIFY(LIBXSMM_GEMV_SYMBOL(TYPE))); \
      if (libxsmm_gemv_wrapper_dynamic_.pv != (CALLER)) { \
        /*LIBXSMM_ATOMIC_STORE(&(ORIGINAL), libxsmm_gemv_wrapper_dynamic_.pf, LIBXSMM_ATOMIC_RELAXED);*/ \
        ORIGINAL = libxsmm_gemv_wrapper_dynamic_.pf; \
      } \
      LIBXSMM_GEMV_WRAPPER_BLAS(TYPE, ORIGINAL, CALLER, LIBXSMM_GEMV_SYMBOL(TYPE)); \
    }
#else
# define LIBXSMM_GEMM_WRAPPER_DYNAMIC(TYPE, ORIGINAL, CALLER) LIBXSMM_GEMM_WRAPPER_BLAS( \
    TYPE, ORIGINAL, CALLER, LIBXSMM_GEMM_SYMBOL(TYPE))
# define LIBXSMM_GEMV_WRAPPER_DYNAMIC(TYPE, ORIGINAL, CALLER) LIBXSMM_GEMV_WRAPPER_BLAS( \
    TYPE, ORIGINAL, CALLER, LIBXSMM_GEMV_SYMBOL(TYPE))
#endif

#if defined(NDEBUG) /* library code is expected to be mute */
# define LIBXSMM_GEMM_WRAPPER(TYPE, ORIGINAL, CALLER) if (0 == (ORIGINAL)) { \
    LIBXSMM_GEMM_WRAPPER_STATIC(TYPE, ORIGINAL, CALLER); \
    LIBXSMM_GEMM_WRAPPER_DYNAMIC(TYPE, ORIGINAL, CALLER); \
  }
#else
# define LIBXSMM_GEMM_WRAPPER(TYPE, ORIGINAL, CALLER) if (0 == (ORIGINAL)) { \
    LIBXSMM_GEMM_WRAPPER_STATIC(TYPE, ORIGINAL, CALLER); \
    LIBXSMM_GEMM_WRAPPER_DYNAMIC(TYPE, ORIGINAL, CALLER); \
    if (0 == (ORIGINAL)) { \
      static int libxsmm_gemm_wrapper_error_once_ = 0; \
      if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&libxsmm_gemm_wrapper_error_once_, 1, LIBXSMM_ATOMIC_RELAXED)) { \
        fprintf(stderr, "LIBXSMM ERROR: application must be linked against LAPACK/BLAS!\n"); \
      } \
    } \
  }
#endif


/** Provides GEMM functions available via BLAS; NOT thread-safe. */
LIBXSMM_API void libxsmm_gemm_init(int archid);

/** Finalizes the GEMM facility; NOT thread-safe. */
LIBXSMM_API void libxsmm_gemm_finalize(void);

LIBXSMM_API unsigned char libxsmm_gemm_typesize(libxsmm_gemm_precision precision);

LIBXSMM_API int libxsmm_gemm_prefetch2uid(libxsmm_gemm_prefetch_type prefetch);
LIBXSMM_API libxsmm_gemm_prefetch_type libxsmm_gemm_uid2prefetch(int uid);

LIBXSMM_API int libxsmm_dgemm_descriptor_init(libxsmm_gemm_descriptor* descriptor,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  double alpha, double beta, int flags, int prefetch);
LIBXSMM_API int libxsmm_sgemm_descriptor_init(libxsmm_gemm_descriptor* descriptor,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  float alpha, float beta, int flags, int prefetch);
LIBXSMM_API int libxsmm_wgemm_descriptor_init(libxsmm_gemm_descriptor* descriptor,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  int alpha, int beta, int flags, int prefetch);

#if defined(LIBXSMM_GEMM_WRAP_STATIC)
LIBXSMM_API void LIBXSMM_FSYMBOL(__real_sgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*, const float* b, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(__real_dgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*, const double* b, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
#endif /*defined(LIBXSMM_GEMM_WRAP_STATIC)*/

#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)
LIBXSMM_API void LIBXSMM_FSYMBOL(__wrap_sgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*, const float* b, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(__wrap_dgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*, const double* b, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
#endif

LIBXSMM_GEMM_SYMBOL_DECL(LIBXSMM_GEMM_CONST, float);
LIBXSMM_GEMM_SYMBOL_DECL(LIBXSMM_GEMM_CONST, double);

typedef union LIBXSMM_RETARGETABLE libxsmm_gemm_batchitem {
  struct {
    const void *a, *b;
    void *c;
  } value;
  struct {
    libxsmm_gemm_descriptor desc;
    unsigned int count;
    const char* symbol;
  } stat;
  /* TODO: consider padding */
} libxsmm_gemm_batchitem;

LIBXSMM_API int libxsmm_mmbatch_internal(libxsmm_xmmfunction kernel, libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const void* a, const void* b, void* c, libxsmm_blasint batchsize, int tid, int nthreads,
  const libxsmm_gemm_descriptor* info);

LIBXSMM_API int libxsmm_dmmbatch_blas(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const double* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb, const double* beta, void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize);
LIBXSMM_API int libxsmm_smmbatch_blas(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const float* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb, const float* beta, void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize);

typedef void (*libxsmm_mmbatch_flush_function)(void);

/** Configuration table containing the tile sizes separate for DP and SP. */
LIBXSMM_API_VARIABLE /*const*/ unsigned int (*libxsmm_gemm_tile)[3/*M,N,K*/][8/*size-range*/];
/** auto-batch descriptor (filter). */
LIBXSMM_API_VARIABLE libxsmm_gemm_descriptor libxsmm_gemm_batchdesc;
/** Records a batch of SMMs. */
LIBXSMM_API_VARIABLE libxsmm_gemm_batchitem* libxsmm_gemm_batcharray;
/** Lock: libxsmm_mmbatch_begin, libxsmm_mmbatch_end, internal_mmbatch_flush. */
LIBXSMM_API_VARIABLE LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK_DEFAULT) libxsmm_gemm_batchlock;
/** Maximum size of the recorded batch. */
LIBXSMM_API_VARIABLE unsigned int libxsmm_gemm_batchsize;
/** Grain/chunk size when processing batches. */
LIBXSMM_API_VARIABLE int libxsmm_gemm_chunksize;
/** Determines the default prefetch strategy, which is used in case of LIBXSMM_PREFETCH_AUTO. */
LIBXSMM_API_VARIABLE int libxsmm_gemm_auto_prefetch_default;
/** Determines the prefetch strategy, which is used in case of LIBXSMM_PREFETCH_AUTO. */
LIBXSMM_API_VARIABLE int libxsmm_gemm_auto_prefetch;
/** Prefetch strategy for tiled GEMM. */
LIBXSMM_API_VARIABLE int libxsmm_gemm_tiled_prefetch;
/** Determines if OpenMP tasks are used. */
LIBXSMM_API_VARIABLE int libxsmm_gemm_tasks;
/**
 * Intercepted GEMM
 * - odd: sequential and non-tiled (small problem sizes only)
 * - even (or negative): parallelized and tiled (all problem sizes)
 * - 3: GEMV is intercepted; small problem sizes
 * - 4: GEMV is intercepted; all problem sizes
 */
LIBXSMM_API_VARIABLE int libxsmm_gemm_wrap;

#endif /*LIBXSMM_GEMM_H*/

