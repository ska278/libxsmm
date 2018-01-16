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
#include "libxsmm_gemm.h"
#include <libxsmm_mhd.h>
#include "libxsmm_hash.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_GEMM_CHECK) && !defined(NDEBUG)
# define LIBXSMM_GEMM_CHECK
#endif
#if !defined(LIBXSMM_GEMM_BATCHSIZE)
# define LIBXSMM_GEMM_BATCHSIZE 1024
#endif
#if !defined(LIBXSMM_GEMM_CHUNKSIZE)
# define LIBXSMM_GEMM_CHUNKSIZE -1
#endif

#if !defined(LIBXSMM_NO_SYNC) /** Locks for the batch interface (duplicated C indexes). */
# define LIBXSMM_GEMM_LOCKIDX(IDX, NPOT) LIBXSMM_MOD2(LIBXSMM_CONCATENATE(libxsmm_crc32_u,LIBXSMM_BLASINT_NBITS)(2507/*seed*/, IDX), NPOT)
# define LIBXSMM_GEMM_LOCKPTR(PTR, NPOT) LIBXSMM_MOD2(libxsmm_crc32_u64(1975/*seed*/, (uintptr_t)(PTR)), NPOT)
# if !defined(LIBXSMM_GEMM_MAXNLOCKS)
#   define LIBXSMM_GEMM_MAXNLOCKS 1024
# endif
# if !defined(LIBXSMM_GEMM_LOCKFWD)
#   define LIBXSMM_GEMM_LOCKFWD
# endif
LIBXSMM_API_VARIABLE unsigned int internal_gemm_nlocks; /* populated number of locks */
LIBXSMM_API_VARIABLE union LIBXSMM_RETARGETABLE {
  char pad[LIBXSMM_CACHELINE];
  LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK_DEFAULT) state;
} internal_gemm_lock[LIBXSMM_GEMM_MAXNLOCKS];
#endif


LIBXSMM_API_DEFINITION LIBXSMM_GEMM_WEAK libxsmm_sgemm_function libxsmm_original_sgemm(const char* caller)
{
  static volatile libxsmm_sgemm_function original = 0;
  LIBXSMM_GEMM_WRAPPER(float, original, caller);
  assert(0 != original);
  return original;
}


LIBXSMM_API_DEFINITION LIBXSMM_GEMM_WEAK libxsmm_dgemm_function libxsmm_original_dgemm(const char* caller)
{
  static volatile libxsmm_dgemm_function original = 0;
  LIBXSMM_GEMM_WRAPPER(double, original, caller);
  assert(0 != original);
  return original;
}


LIBXSMM_API_DEFINITION void libxsmm_gemm_init(int archid)
{
  /* setup tile sizes according to CPUID or environment (LIBXSMM_TGEMM_M, LIBXSMM_TGEMM_N, LIBXSMM_TGEMM_K) */
  static unsigned int tile_configs[/*configs*/][2/*DP/SP*/][3/*TILE_M,TILE_N,TILE_K*/][8/*size-range*/] = {
    /* generic (hsw) */
    { { {  25,  50,  69, 169, 169, 169, 169, 169 }, {  37,  98,  78,  39,  39,  39,  39,  39 }, { 100,  81,  55,  37,  37,  37,  37,  37 } },   /* DP */
      { {  43,  49, 107, 103, 103, 103, 103, 103 }, {  38,  52, 113, 141, 141, 141, 141, 141 }, { 232,  89, 100,  76,  76,  76,  76,  76 } } }, /* SP */
    /* mic (knl/knm) */
    { { { 168, 130, 131, 110, 110, 110, 110, 256 }, {  10,  28,  20,  24,  24,  24,  24,  10 }, {  39,  43,  40,  63,  63,  63,  63,  77 } },   /* DP */
      { {  69, 152, 149, 172, 172, 172, 172, 172 }, {  11,  14,  18,  28,  28,  28,  28,  28 }, { 100, 103,  61,  63,  63,  63,  63,  63 } } }, /* SP */
    /* core (skx) */
    { { {  39,  52,  57, 201, 256, 201, 201, 201 }, {  26,  86, 115,  14,  27,  14,  14,  14 }, { 256, 101, 102,  53, 114,  53,  53,  53 } },   /* DP */
      { {  41, 119, 102, 106, 106, 106, 106, 106 }, {  32,  65, 108, 130, 130, 130, 130, 130 }, {  73,  90,  86,  89,  89,  89,  89,  89 } } }  /* SP */
  };
  const unsigned int config = (LIBXSMM_X86_AVX512_CORE <= libxsmm_target_archid ? 2 : ((LIBXSMM_X86_AVX512_MIC <= archid && LIBXSMM_X86_AVX512_CORE > libxsmm_target_archid) ? 1 : 0));
  unsigned int i;

  { /* setup prefetch strategy for tiled GEMMs */
    const char *const env_p = getenv("LIBXSMM_TGEMM_PREFETCH");
    const int tiled_prefetch_default = LIBXSMM_PREFETCH_AL2_AHEAD;
    const int uid = ((0 == env_p || 0 == *env_p) ? -1/*default*/ : atoi(env_p));
    libxsmm_gemm_tiled_prefetch = (0 <= uid ? libxsmm_gemm_uid2prefetch(uid) : tiled_prefetch_default);
  }
#if !defined(LIBXSMM_NO_SYNC)
  { /* initialize locks for the batch interface */
    const char *const env_locks = getenv("LIBXSMM_GEMM_NLOCKS");
    const int nlocks = ((0 == env_locks || 0 == *env_locks) ? -1/*default*/ : atoi(env_locks));
    internal_gemm_nlocks = LIBXSMM_UP2POT(0 > nlocks ? (LIBXSMM_GEMM_MAXNLOCKS) : LIBXSMM_MIN(nlocks, LIBXSMM_GEMM_MAXNLOCKS));
    for (i = 0; i < internal_gemm_nlocks; ++i) LIBXSMM_LOCK_INIT(LIBXSMM_LOCK_DEFAULT, &internal_gemm_lock[i].state, &libxsmm_lock_attr_default);
  }
#endif
#if defined(LIBXSMM_GEMM_MMBATCH)
  { const char *const env_w = getenv("LIBXSMM_GEMM_WRAP");
    /* intercepted GEMMs (1: sequential and non-tiled, 2: parallelized and tiled) */
    libxsmm_gemm_wrap = ((0 == env_w || 0 == *env_w) ? (LIBXSMM_WRAP) : atoi(env_w));
    if (0 != libxsmm_gemm_wrap) {
      const char *const env_b = getenv("LIBXSMM_GEMM_BATCHSIZE");
      const unsigned int batchsize = ((0 == env_b || 0 == *env_b || 0 >= atoi(env_b)) ? (LIBXSMM_GEMM_BATCHSIZE) : atoi(env_b));
      const void *const extra = 0;
      /* draw default/non-scratch memory, but utilize the scratch memory allocator */
      assert(1 < (LIBXSMM_GEMM_BATCHSCALE));
      if (EXIT_SUCCESS == libxsmm_xmalloc((void**)&libxsmm_gemm_batcharray,
        (size_t)((LIBXSMM_GEMM_BATCHSCALE) * sizeof(libxsmm_gemm_batchitem) * batchsize),
        0, LIBXSMM_MALLOC_FLAG_SCRATCH, &extra, sizeof(extra)))
      {
        LIBXSMM_LOCK_INIT(LIBXSMM_LOCK_DEFAULT, &libxsmm_gemm_batchlock, &libxsmm_lock_attr_default);
        libxsmm_gemm_batchsize = batchsize;
      }
      if ((3 <= libxsmm_verbosity || 0 > libxsmm_verbosity) && (0 == env_w || 0 == *env_w)) { /* enables the auto-batch statistic */
        libxsmm_gemm_batchdesc.flags = LIBXSMM_MMBATCH_FLAG_STATISTIC;
      }
    }
  }
#endif
  {
    const char *const env_m = getenv("LIBXSMM_TGEMM_M"), *const env_n = getenv("LIBXSMM_TGEMM_N"), *const env_k = getenv("LIBXSMM_TGEMM_K");
    const int m = ((0 == env_m || 0 == *env_m) ? 0 : atoi(env_m));
    const int n = ((0 == env_n || 0 == *env_n) ? 0 : atoi(env_n));
    const int k = ((0 == env_k || 0 == *env_k) ? 0 : atoi(env_k));
    for (i = 0; i < 8; ++i) {
      if (0 < m) tile_configs[config][0/*DP*/][0/*M*/][i] = tile_configs[config][1/*SP*/][0/*M*/][i] = m;
      if (0 < n) tile_configs[config][0/*DP*/][1/*N*/][i] = tile_configs[config][1/*SP*/][1/*N*/][i] = n;
      if (0 < k) tile_configs[config][0/*DP*/][2/*K*/][i] = tile_configs[config][1/*SP*/][2/*K*/][i] = k;
    }
    libxsmm_gemm_tile = tile_configs[config];
  }
  { /* grain/chunk size when processing batches */
    const char *const env_c = getenv("LIBXSMM_GEMM_CHUNKSIZE");
    libxsmm_gemm_chunksize = ((0 == env_c || 0 == *env_c || 0 > atoi(env_c)) ? (LIBXSMM_GEMM_CHUNKSIZE) : atoi(env_c));
  }
  { /* determines if OpenMP tasks are used (when available) */
    const char *const env_t = getenv("LIBXSMM_GEMM_TASKS");
    libxsmm_gemm_tasks = ((0 == env_t || 0 == *env_t) ? 0/*disabled*/ : atoi(env_t));
  }
}


LIBXSMM_API_DEFINITION void libxsmm_gemm_finalize(void)
{
#if !defined(LIBXSMM_NO_SYNC)
  unsigned int i; for (i = 0; i < internal_gemm_nlocks; ++i) LIBXSMM_LOCK_DESTROY(LIBXSMM_LOCK_DEFAULT, &internal_gemm_lock[i].state);
#endif
#if defined(LIBXSMM_GEMM_MMBATCH)
  if (0 != libxsmm_gemm_batcharray) {
    void* extra = 0;
    if (EXIT_SUCCESS == libxsmm_get_malloc_xinfo(libxsmm_gemm_batcharray, 0/*size*/, 0/*flags*/, &extra) && 0 != extra) {
      const libxsmm_mmbatch_flush_function flush = *(libxsmm_mmbatch_flush_function*)extra;
      if (0 != flush) flush();
    }
    libxsmm_free(libxsmm_gemm_batcharray);
    LIBXSMM_LOCK_DESTROY(LIBXSMM_LOCK_DEFAULT, &libxsmm_gemm_batchlock);
  }
#endif
}


LIBXSMM_API_DEFINITION unsigned char libxsmm_gemm_typesize(libxsmm_gemm_precision precision)
{
  return libxsmm_typesize((libxsmm_datatype)precision);
}


LIBXSMM_API_DEFINITION int libxsmm_gemm_prefetch2uid(libxsmm_gemm_prefetch_type prefetch)
{
  switch (prefetch) {
    case LIBXSMM_PREFETCH_SIGONLY:            return 2;
    case LIBXSMM_PREFETCH_BL2_VIA_C:          return 3;
    case LIBXSMM_PREFETCH_AL2_AHEAD:          return 4;
    case LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD: return 5;
    case LIBXSMM_PREFETCH_AL2:                return 6;
    case LIBXSMM_PREFETCH_AL2BL2_VIA_C:       return 7;
    case LIBXSMM_PREFETCH_AL2_JPST:           return 8;
    case LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST:  return 9;
    /*case LIBXSMM_PREFETCH_AL2CL2BL2_VIA_C:    return 10;*/
    case LIBXSMM_PREFETCH_AL1:                return 10;
    case LIBXSMM_PREFETCH_BL1:                return 11;
    case LIBXSMM_PREFETCH_CL1:                return 12;
    case LIBXSMM_PREFETCH_AL1_BL1:            return 13;
    case LIBXSMM_PREFETCH_BL1_CL1:            return 14;
    case LIBXSMM_PREFETCH_AL1_CL1:            return 15;
    case LIBXSMM_PREFETCH_AL1_BL1_CL1:        return 16;
    default: {
      assert(LIBXSMM_PREFETCH_NONE == prefetch);
      return 0;
    }
  }
}


LIBXSMM_API_DEFINITION libxsmm_gemm_prefetch_type libxsmm_gemm_uid2prefetch(int uid)
{
  switch (uid) {
    case  1: return LIBXSMM_PREFETCH_NONE;                /* nopf */
    case  2: return LIBXSMM_PREFETCH_SIGONLY;             /* pfsigonly */
    case  3: return LIBXSMM_PREFETCH_BL2_VIA_C;           /* BL2viaC */
    case  4: return LIBXSMM_PREFETCH_AL2_AHEAD;           /* curAL2 */
    case  5: return LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD;  /* curAL2_BL2viaC */
    case  6: return LIBXSMM_PREFETCH_AL2;                 /* AL2 */
    case  7: return LIBXSMM_PREFETCH_AL2BL2_VIA_C;        /* AL2_BL2viaC */
    case  8: return LIBXSMM_PREFETCH_AL2_JPST;            /* AL2jpst */
    case  9: return LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST;   /* AL2jpst_BL2viaC */
    /*case 10: return LIBXSMM_PREFETCH_AL2CL2BL2_VIA_C;*/     /* AL2_BL2viaC_CL2 */
    case 10: return LIBXSMM_PREFETCH_AL1;
    case 11: return LIBXSMM_PREFETCH_BL1;
    case 12: return LIBXSMM_PREFETCH_CL1;
    case 13: return LIBXSMM_PREFETCH_AL1_BL1;
    case 14: return LIBXSMM_PREFETCH_BL1_CL1;
    case 15: return LIBXSMM_PREFETCH_AL1_CL1;
    case 16: return LIBXSMM_PREFETCH_AL1_BL1_CL1;
    default: {
      if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
        static int error_once = 0;
        if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
          fprintf(stderr, "LIBXSMM WARNING: invalid prefetch strategy requested!\n");
        }
      }
      return LIBXSMM_PREFETCH_NONE;
    }
  }
}


LIBXSMM_API_DEFINITION int libxsmm_dgemm_descriptor_init(libxsmm_gemm_descriptor* descriptor,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  double alpha, double beta, int flags, int prefetch)
{
  int result;
  if (LIBXSMM_GEMM_NO_BYPASS(flags, alpha, beta) && 0 != descriptor
    && LIBXSMM_GEMM_NO_BYPASS_DIMS(lda, ldb, ldc)
    && LIBXSMM_GEMM_NO_BYPASS_DIMS(m, n, k))
  {
    LIBXSMM_GEMM_DESCRIPTOR(*descriptor,
      LIBXSMM_GEMM_PRECISION(double), flags, m, n, k, lda, ldb, ldc, alpha, beta,
      0 > prefetch ? libxsmm_gemm_auto_prefetch_default : prefetch);
    result = EXIT_SUCCESS;
  }
  else { /* quiet error (unsupported) */
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_sgemm_descriptor_init(libxsmm_gemm_descriptor* descriptor,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  float alpha, float beta, int flags, int prefetch)
{
  int result;
  if (LIBXSMM_GEMM_NO_BYPASS(flags, alpha, beta) && 0 != descriptor
    && LIBXSMM_GEMM_NO_BYPASS_DIMS(lda, ldb, ldc)
    && LIBXSMM_GEMM_NO_BYPASS_DIMS(m, n, k))
  {
    LIBXSMM_GEMM_DESCRIPTOR(*descriptor,
      LIBXSMM_GEMM_PRECISION(float), flags, m, n, k, lda, ldb, ldc, alpha, beta,
      0 > prefetch ? libxsmm_gemm_auto_prefetch_default : prefetch);
    result = EXIT_SUCCESS;
  }
  else { /* unsupported */
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_wgemm_descriptor_init(libxsmm_gemm_descriptor* descriptor,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  int alpha, int beta, int flags, int prefetch)
{
  int result;
  if (LIBXSMM_GEMM_NO_BYPASS(flags, alpha, beta) && 0 != descriptor
    && LIBXSMM_GEMM_NO_BYPASS_DIMS(lda, ldb, ldc)
    && LIBXSMM_GEMM_NO_BYPASS_DIMS(m, n, k))
  {
    LIBXSMM_GEMM_DESCRIPTOR(*descriptor,
      LIBXSMM_GEMM_PRECISION(short), flags, m, n, k, lda, ldb, ldc, alpha, beta,
      0 > prefetch ? libxsmm_gemm_auto_prefetch_default : prefetch);
    result = EXIT_SUCCESS;
  }
  else { /* unsupported */
    result = EXIT_FAILURE;
  }
  return result;
}


/*DEPRECATED*/LIBXSMM_API_DEFINITION libxsmm_gemm_descriptor* libxsmm_create_dgemm_descriptor(
  char transa, char transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  double alpha, double beta, int prefetch)
{
  libxsmm_gemm_descriptor* result = (libxsmm_gemm_descriptor*)malloc(sizeof(libxsmm_gemm_descriptor));
  if (EXIT_SUCCESS != libxsmm_dgemm_descriptor_init(result, m, n, k, lda, ldb, ldc,
    alpha, beta, LIBXSMM_GEMM_FLAGS(transa, transb), prefetch))
  {
    free(result);
    result = 0;
  }
  if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM WARNING: create_dgemm_descriptor is deprecated, use libxsmm_gemm_descriptor_init instead!\n");
    }
  }
  return result;
}


/*DEPRECATED*/LIBXSMM_API_DEFINITION void libxsmm_release_gemm_descriptor(const libxsmm_gemm_descriptor* descriptor)
{
  free((void*)descriptor);
  if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM WARNING: release_gemm_descriptor is deprecated, libxsmm_gemm_descriptor_init makes it superfluous!\n");
    }
  }
}


LIBXSMM_API_DEFINITION void libxsmm_gemm_print(void* ostream,
  libxsmm_gemm_precision precision, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc)
{
  const libxsmm_blasint nn = *(n ? n : m), kk = *(k ? k : m), ilda = *(lda ? lda : m), ildb = (ldb ? *ldb : kk), ildc = *(ldc ? ldc : m);
  const char ctransa = (char)(0 != transa ? (*transa) : (0 == (LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_A) ? 'N' : 'T'));
  const char ctransb = (char)(0 != transb ? (*transb) : (0 == (LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_B) ? 'N' : 'T'));
  libxsmm_mhd_elemtype mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_CHAR;
  char string_a[128], string_b[128], typeprefix = 0;

  switch (precision) {
    case LIBXSMM_GEMM_PRECISION_F64: {
      LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "%g", 0 != alpha ? *((const double*)alpha) : LIBXSMM_ALPHA);
      LIBXSMM_SNPRINTF(string_b, sizeof(string_b), "%g", 0 != beta  ? *((const double*)beta)  : LIBXSMM_BETA);
      mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_F64;
      typeprefix = 'd';
    } break;
    case LIBXSMM_GEMM_PRECISION_F32: {
      LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "%g", 0 != alpha ? *((const float*)alpha) : LIBXSMM_ALPHA);
      LIBXSMM_SNPRINTF(string_b, sizeof(string_b), "%g", 0 != beta  ? *((const float*)beta)  : LIBXSMM_BETA);
      mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_F32;
      typeprefix = 's';
    }
    default: /* TODO: support I16, etc. */;
  }

  if (0 != typeprefix) {
    if (0 != ostream) { /* print information about GEMM call */
      if (0 != a && 0 != b && 0 != c) {
        fprintf((FILE*)ostream, "%cgemm('%c', '%c', %" PRIuPTR "/*m*/, %" PRIuPTR "/*n*/, %" PRIuPTR "/*k*/,\n"
                                "  %s/*alpha*/, %p/*a*/, %" PRIuPTR "/*lda*/,\n"
                                "              %p/*b*/, %" PRIuPTR "/*ldb*/,\n"
                                "   %s/*beta*/, %p/*c*/, %" PRIuPTR "/*ldc*/)",
          typeprefix, ctransa, ctransa, (uintptr_t)*m, (uintptr_t)nn, (uintptr_t)kk,
          string_a, a, (uintptr_t)ilda, b, (uintptr_t)ildb, string_b, c, (uintptr_t)ildc);
      }
      else {
        fprintf((FILE*)ostream, "%cgemm(trans=%c%c mnk=%" PRIuPTR ",%" PRIuPTR ",%" PRIuPTR
                                                 " ldx=%" PRIuPTR ",%" PRIuPTR ",%" PRIuPTR " a,b=%s,%s)",
          typeprefix, ctransa, ctransa, (uintptr_t)*m, (uintptr_t)nn, (uintptr_t)kk,
          (uintptr_t)ilda, (uintptr_t)ildb, (uintptr_t)ildc, string_a, string_b);
      }
    }
    else { /* dump A, B, and C matrices into MHD files */
      char extension_header[256];
      size_t data_size[2], size[2];

      if (0 != a) {
        LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "TRANS = %c\nALPHA = %s", ctransa, string_a);
        LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_a_%p.mhd", a);
        data_size[0] = (size_t)ilda; data_size[1] = (size_t)kk; size[0] = (size_t)(*m); size[1] = (size_t)kk;
        libxsmm_mhd_write(string_a, 0/*offset*/, size, data_size, 2/*ndims*/, 1/*ncomponents*/, mhd_elemtype, a,
          0/*header_size*/, extension_header, 0/*extension*/, 0/*extension_size*/);
      }
      if (0 != b) {
        LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "\nTRANS = %c", ctransb);
        LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_b_%p.mhd", b);
        data_size[0] = (size_t)ildb; data_size[1] = (size_t)nn; size[0] = (size_t)kk; size[1] = (size_t)nn;
        libxsmm_mhd_write(string_a, 0/*offset*/, size, data_size, 2/*ndims*/, 1/*ncomponents*/, mhd_elemtype, b,
          0/*header_size*/, extension_header, 0/*extension*/, 0/*extension_size*/);
      }
      if (0 != c) {
        LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "BETA = %s", string_b);
        LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_c_%p.mhd", c);
        data_size[0] = (size_t)ildc; data_size[1] = (size_t)nn; size[0] = (size_t)(*m); size[1] = (size_t)nn;
        libxsmm_mhd_write(string_a, 0/*offset*/, size, data_size, 2/*ndims*/, 1/*ncomponents*/, mhd_elemtype, c,
          0/*header_size*/, extension_header, 0/*extension*/, 0/*extension_size*/);
      }
    }
  }
}


LIBXSMM_API_DEFINITION void libxsmm_blas_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  const libxsmm_blasint nn = *(n ? n : m), kk = *(k ? k : m), ilda = *(lda ? lda : m), ildb = (ldb ? *ldb : kk), ildc = *(ldc ? ldc : m);
  const float ralpha = (0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA));
  const float rbeta = (0 != beta ? *beta : ((float)LIBXSMM_BETA));
  const int flags = LIBXSMM_GEMM_PFLAGS(transa, transb, LIBXSMM_FLAGS);
  LIBXSMM_BLAS_SGEMM(flags, *m, nn, kk, ralpha, a, ilda, b, ildb, rbeta, c, ildc);
}


LIBXSMM_API_DEFINITION void libxsmm_blas_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  const libxsmm_blasint nn = *(n ? n : m), kk = *(k ? k : m), ilda = *(lda ? lda : m), ildb = (ldb ? *ldb : kk), ildc = *(ldc ? ldc : m);
  const double ralpha = (0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA));
  const double rbeta = (0 != beta ? *beta : ((double)LIBXSMM_BETA));
  const int flags = LIBXSMM_GEMM_PFLAGS(transa, transb, LIBXSMM_FLAGS);
  LIBXSMM_BLAS_DGEMM(flags, *m, nn, kk, ralpha, a, ilda, b, ildb, rbeta, c, ildc);
}


LIBXSMM_API_DEFINITION void libxsmm_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  const libxsmm_blasint nn = *(n ? n : m), kk = *(k ? k : m), ilda = *(lda ? lda : m), ildb = (ldb ? *ldb : kk), ildc = *(ldc ? ldc : m);
  const float ralpha = (0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA));
  const float rbeta = (0 != beta ? *beta : ((float)LIBXSMM_BETA));
  const int flags = LIBXSMM_GEMM_PFLAGS(transa, transb, LIBXSMM_FLAGS);
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
  const char *const check = getenv("LIBXSMM_CHECK");
  float *const d = (float*)((0 == LIBXSMM_GEMM_NO_BYPASS(flags, ralpha, rbeta)
      || 0 == check || 0 == *check || 0 == check[0]) ? 0
    : libxsmm_aligned_scratch((size_t)((*m) * nn * sizeof(float)), 0/*auto-aligned*/));
  if (0 != d) {
    libxsmm_blasint i, j;
    for (i = 0; i < nn; ++i) {
      for (j = 0; j < (*m); ++j) {
        d[i*(*m)+j] = c[i*ildc+j];
      }
    }
  }
#endif
  LIBXSMM_SGEMM(flags, *m, nn, kk, ralpha, a, ilda, b, ildb, rbeta, c, ildc);
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
  if (0 != d) {
    libxsmm_matdiff_info diff;
    libxsmm_blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, d, m);
    if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE_F32, *m, nn, d, c, m, ldc, &diff)) {
      LIBXSMM_FLOCK(stderr);
      libxsmm_gemm_print(stderr, LIBXSMM_GEMM_PRECISION_F32, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      fprintf(stderr, " L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
      LIBXSMM_FUNLOCK(stderr);
    }
    libxsmm_free(d);
  }
#endif
}


LIBXSMM_API_DEFINITION void libxsmm_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  const libxsmm_blasint nn = *(n ? n : m), kk = *(k ? k : m), ilda = *(lda ? lda : m), ildb = (ldb ? *ldb : kk), ildc = *(ldc ? ldc : m);
  const double ralpha = (0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA));
  const double rbeta = (0 != beta ? *beta : ((double)LIBXSMM_BETA));
  const int flags = LIBXSMM_GEMM_PFLAGS(transa, transb, LIBXSMM_FLAGS);
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
  const char *const check = getenv("LIBXSMM_CHECK");
  double *const d = (double*)((0 == LIBXSMM_GEMM_NO_BYPASS(flags, ralpha, rbeta)
      || 0 == check || 0 == *check || 0 == check[0]) ? 0
    : libxsmm_aligned_scratch((size_t)((*m) * nn * sizeof(double)), 0/*auto-aligned*/));
  if (0 != d) {
    libxsmm_blasint i, j;
    for (i = 0; i < nn; ++i) {
      for (j = 0; j < (*m); ++j) {
        d[i*(*m)+j] = c[i*ildc+j];
      }
    }
  }
#endif
  LIBXSMM_DGEMM(flags, *m, nn, kk, ralpha, a, ilda, b, ildb, rbeta, c, ildc);
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
  if (0 != d) {
    libxsmm_matdiff_info diff;
    libxsmm_blas_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, d, m);
    if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE_F64, *m, nn, d, c, m, ldc, &diff)) {
      LIBXSMM_FLOCK(stderr);
      libxsmm_gemm_print(stderr, LIBXSMM_GEMM_PRECISION_F64, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      fprintf(stderr, " L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
      LIBXSMM_FUNLOCK(stderr);
    }
    libxsmm_free(d);
  }
#endif
}


LIBXSMM_API_DEFINITION int libxsmm_mmbatch_internal(libxsmm_xmmfunction kernel, libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const void* a, const void* b, void* c, libxsmm_blasint batchsize, int tid, int nthreads,
  const libxsmm_gemm_descriptor* info)
{
  int result = EXIT_SUCCESS;
  const libxsmm_blasint size = LIBXSMM_ABS(batchsize);
  const libxsmm_blasint tasksize = (size + nthreads - 1) / nthreads;
  const libxsmm_blasint begin = tid * tasksize, span = begin + tasksize;
  const libxsmm_blasint end = LIBXSMM_MIN(span, size);

  assert(0 != info);
  if (begin < end) {
    const libxsmm_blasint typesize = libxsmm_gemm_typesize((libxsmm_gemm_precision)info->datatype);
    const char *const a0 = (const char*)a, *const b0 = (const char*)b;
    char *const c0 = (char*)c;
    libxsmm_blasint i, ni;

    assert(0 < typesize);
    if (0 != index_stride) { /* stride arrays contain indexes */
      if (((int)sizeof(libxsmm_blasint)) <= index_stride) {
        const char *const sa = (const char*)stride_a, *const sb = (const char*)stride_b, *const sc = (const char*)stride_c;
        libxsmm_blasint ii = begin * index_stride, ic = (0 != sc ? (*((const libxsmm_blasint*)(sc + ii)) - index_base) : 0);
        const char* ai = a0 + (0 != sa ? ((*((const libxsmm_blasint*)(sa + ii)) - index_base) * typesize) : 0);
        const char* bi = b0 + (0 != sb ? ((*((const libxsmm_blasint*)(sb + ii)) - index_base) * typesize) : 0);
        char*       ci = c0 + ic * typesize;
        const libxsmm_blasint end1 = (end != size ? end : (end - 1));
#if !defined(LIBXSMM_NO_SYNC)
        if (1 == nthreads || 0 == internal_gemm_nlocks || 0 > batchsize || 0 == info->beta)
#endif
        { /* no locking */
          for (i = begin; i < end1; i = ni) {
            ni = i + 1; ii = ni * index_stride;
            {
              const char *const an = a0 + (0 != sa ? ((*((const libxsmm_blasint*)(sa + ii)) - index_base) * typesize) : 0);
              const char *const bn = b0 + (0 != sb ? ((*((const libxsmm_blasint*)(sb + ii)) - index_base) * typesize) : 0);
              char       *const cn = c0 + (0 != sc ? ((*((const libxsmm_blasint*)(sc + ii)) - index_base) * typesize) : 0);
              kernel.xmm(ai, bi, ci, an, bn, cn); /* with prefetch */
              ai = an; bi = bn; ci = cn;
            }
          }
          if (end != end1) { /* remainder multiplication */
            kernel.xmm(ai, bi, ci, ai, bi, ci); /* pseudo-prefetch */
          }
        }
#if !defined(LIBXSMM_NO_SYNC)
        else { /* synchronize among C-indexes */
          LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK_DEFAULT)* lock = &internal_gemm_lock[LIBXSMM_GEMM_LOCKIDX(ic, internal_gemm_nlocks)].state;
# if defined(LIBXSMM_GEMM_LOCKFWD)
          LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK_DEFAULT)* lock0 = 0;
# endif
          for (i = begin; i < end1; i = ni) {
            ni = i + 1; ii = ni * index_stride; ic = (0 != sc ? (*((const libxsmm_blasint*)(sc + ii)) - index_base) : 0);
            {
              const char *const an = a0 + (0 != sa ? ((*((const libxsmm_blasint*)(sa + ii)) - index_base) * typesize) : 0);
              const char *const bn = b0 + (0 != sb ? ((*((const libxsmm_blasint*)(sb + ii)) - index_base) * typesize) : 0);
              char       *const cn = c0 + ic * typesize;
              LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK_DEFAULT) *const lock1 = &internal_gemm_lock[LIBXSMM_GEMM_LOCKIDX(ic, internal_gemm_nlocks)].state;
# if defined(LIBXSMM_GEMM_LOCKFWD)
              if (lock != lock0) { lock0 = lock; LIBXSMM_LOCK_ACQUIRE(LIBXSMM_LOCK_DEFAULT, lock); }
# else
              LIBXSMM_LOCK_ACQUIRE(LIBXSMM_LOCK_DEFAULT, lock);
# endif
              kernel.xmm(ai, bi, ci, an, bn, cn); /* with prefetch */
# if defined(LIBXSMM_GEMM_LOCKFWD)
              if (lock != lock1 || ni == end1) { LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK_DEFAULT, lock); lock = lock1; }
# else
              LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK_DEFAULT, lock); lock = lock1;
# endif
              ai = an; bi = bn; ci = cn; /* next */
            }
          }
          if (end != end1) { /* remainder multiplication */
            LIBXSMM_LOCK_ACQUIRE(LIBXSMM_LOCK_DEFAULT, lock);
            kernel.xmm(ai, bi, ci, ai, bi, ci); /* pseudo-prefetch */
            LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK_DEFAULT, lock);
          }
        }
#endif /*!defined(LIBXSMM_NO_SYNC)*/
      }
      else { /* incorrect argument(s) */
        result = EXIT_FAILURE;
      }
    }
    else { /* singular strides are measured in Bytes */
      const libxsmm_blasint stride_unit = sizeof(libxsmm_blasint);
      const libxsmm_blasint da = (0 != stride_a ? (*stride_a - index_base * stride_unit) : 0);
      const libxsmm_blasint db = (0 != stride_b ? (*stride_b - index_base * stride_unit) : 0);
      const libxsmm_blasint dc = (0 != stride_c ? (*stride_c - index_base * stride_unit) : 0);

      if (typesize <= da && typesize <= db && typesize <= dc) {
        const libxsmm_blasint end1 = (end != size ? end : (end - 1));
        const char *ai = a0 + da * begin, *bi = b0 + db * begin;
        char* ci = c0 + dc * begin;

#if !defined(LIBXSMM_NO_SYNC)
        if (1 == nthreads || 0 == internal_gemm_nlocks || 0 > batchsize || 0 == info->beta)
#endif
        { /* no locking */
          for (i = begin; i < end1; ++i) {
#if defined(LIBXSMM_GEMM_CHECK)
            if (0 != *((const void**)ai) && 0 != *((const void**)bi) && 0 != *((const void**)ci))
#endif
            {
              const char *const an = ai + da, *const bn = bi + db;
              char *const cn = ci + dc;
              kernel.xmm( /* with prefetch */
                *((const void**)ai), *((const void**)bi), *((void**)ci),
                *((const void**)an), *((const void**)bn), *((const void**)cn));
              ai = an; bi = bn; ci = cn; /* next */
            }
          }
          if (end != end1 /* remainder multiplication */
#if defined(LIBXSMM_GEMM_CHECK)
            && 0 != *((const void**)ai) && 0 != *((const void**)bi) && 0 != *((const void**)ci)
#endif
            )
          {
            kernel.xmm( /* pseudo-prefetch */
              *((const void**)ai), *((const void**)bi), *((void**)ci),
              *((const void**)ai), *((const void**)bi), *((const void**)ci));
          }
        }
#if !defined(LIBXSMM_NO_SYNC)
        else { /* synchronize among C-indexes */
          void* cc = *((void**)ci);
          LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK_DEFAULT)* lock = &internal_gemm_lock[LIBXSMM_GEMM_LOCKPTR(cc, internal_gemm_nlocks)].state;
# if defined(LIBXSMM_GEMM_LOCKFWD)
          LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK_DEFAULT)* lock0 = 0;
# endif
          for (i = begin; i < end1; i = ni) {
            ni = i + 1;
# if defined(LIBXSMM_GEMM_CHECK)
            if (0 != *((const void**)ai) && 0 != *((const void**)bi) && 0 != cc)
# endif
            {
              const char *const an = ai + da, *const bn = bi + db;
              char *const cn = ci + dc;
              void *const nc = *((void**)cn);
              LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK_DEFAULT) *const lock1 = &internal_gemm_lock[LIBXSMM_GEMM_LOCKPTR(nc, internal_gemm_nlocks)].state;
# if defined(LIBXSMM_GEMM_LOCKFWD)
              if (lock != lock0) { lock0 = lock; LIBXSMM_LOCK_ACQUIRE(LIBXSMM_LOCK_DEFAULT, lock); }
# else
              LIBXSMM_LOCK_ACQUIRE(LIBXSMM_LOCK_DEFAULT, lock);
# endif
              kernel.xmm( /* with prefetch */
                *((const void**)ai), *((const void**)bi), cc,
                *((const void**)an), *((const void**)bn), *((const void**)cn));
# if defined(LIBXSMM_GEMM_LOCKFWD)
              if (lock != lock1 || ni == end1) { LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK_DEFAULT, lock); lock = lock1; }
# else
              LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK_DEFAULT, lock); lock = lock1;
# endif
              ai = an; bi = bn; ci = cn; cc = nc; /* next */
            }
          }
          if (end != end1 /* remainder multiplication */
# if defined(LIBXSMM_GEMM_CHECK)
            && 0 != *((const void**)ai) && 0 != *((const void**)bi) && 0 != cc
# endif
            )
          {
            LIBXSMM_LOCK_ACQUIRE(LIBXSMM_LOCK_DEFAULT, lock);
            kernel.xmm( /* pseudo-prefetch */
              *((const void**)ai), *((const void**)bi), cc,
              *((const void**)ai), *((const void**)bi), cc);
            LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK_DEFAULT, lock);
          }
        }
#endif /*!defined(LIBXSMM_NO_SYNC)*/
      }
      else { /* incorrect argument(s) */
        result = EXIT_FAILURE;
      }
    }
  }
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_mmbatch(libxsmm_xmmfunction kernel, libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const void* a, const void* b, void* c, libxsmm_blasint batchsize, int tid, int nthreads)
{
  const libxsmm_kernel_info* info;
  libxsmm_code_pointer code;
  libxsmm_kernel_kind kind;
  int result;
  code.xgemm = kernel;
  info = libxsmm_get_kernel_info(code, &kind, 0/*size*/);
  if (0 != info && LIBXSMM_KERNEL_KIND_MATMUL == kind && 0 != a && 0 != b && 0 != c
    /* use (signed) integer types, but check sanity of input */
    && 0 <= tid && tid < nthreads)
  {
    LIBXSMM_INIT
    result = libxsmm_mmbatch_internal(kernel, index_base, index_stride,
      stride_a, stride_b, stride_c, a, b, c, batchsize,
      tid, nthreads, &info->xgemm);
  }
  else { /* incorrect argument(s) */
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_dmmbatch_blas(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const double* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb, const double* beta, void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize)
{
  int result = EXIT_SUCCESS;

  if (0 != a && 0 != b && 0 != c) {
    const libxsmm_blasint end = LIBXSMM_ABS(batchsize);
    libxsmm_blasint i;

    if (0 != index_stride) { /* stride arrays contain indexes */
      if (((int)sizeof(libxsmm_blasint)) <= index_stride) {
        const libxsmm_blasint da = (0 != stride_a ? (*stride_a - index_base) : 0);
        const libxsmm_blasint db = (0 != stride_b ? (*stride_b - index_base) : 0);
        const libxsmm_blasint dc = (0 != stride_c ? (*stride_c - index_base) : 0);
        const char *const sa = (const char*)stride_a, *const sb = (const char*)stride_b, *const sc = (const char*)stride_c;
        const double *const a0 = (const double*)a, *const b0 = (const double*)b, *ai = a0 + da, *bi = b0 + db;
        double *const c0 = (double*)c, *ci = c0 + dc;
        for (i = 0; i < end; ++i) {
          const libxsmm_blasint ii = (i + 1) * index_stride;
          const double *const an = a0 + (0 != stride_a ? (*((const libxsmm_blasint*)(sa + ii)) - index_base) : 0);
          const double *const bn = b0 + (0 != stride_b ? (*((const libxsmm_blasint*)(sb + ii)) - index_base) : 0);
          double       *const cn = c0 + (0 != stride_c ? (*((const libxsmm_blasint*)(sc + ii)) - index_base) : 0);
#if defined(LIBXSMM_GEMM_CHECK)
          if (0 != ai && 0 != bi && 0 != ci)
#endif
          {
            libxsmm_blas_dgemm(transa, transb, &m, &n, &k, alpha, ai, lda, bi, ldb, beta, ci, ldc);
          }
          ai = an; bi = bn; ci = cn;
        }
      }
      else { /* incorrect argument(s) */
        result = EXIT_FAILURE;
      }
    }
    else { /* singular strides are measured in Bytes */
      const libxsmm_blasint stride_unit = sizeof(libxsmm_blasint);
      const libxsmm_blasint da = (0 != stride_a ? (*stride_a - index_base * stride_unit) : 0);
      const libxsmm_blasint db = (0 != stride_b ? (*stride_b - index_base * stride_unit) : 0);
      const libxsmm_blasint dc = (0 != stride_c ? (*stride_c - index_base * stride_unit) : 0);
      if (((int)sizeof(double)) <= LIBXSMM_MIN(LIBXSMM_MIN(da, db), dc)) {
        const char *const a0 = (const char*)a, *const b0 = (const char*)b, *ai = a0, *bi = b0;
        char *const c0 = (char*)c, *ci = c0;
        for (i = 0; i < end; ++i) {
          const char *const an = ai + da, *const bn = bi + db;
          char *const cn = ci + dc;
#if defined(LIBXSMM_GEMM_CHECK)
          if (0 != *((const double**)ai) && 0 != *((const double**)bi) && 0 != *((const double**)ci))
#endif
          {
            libxsmm_blas_dgemm(transa, transb, &m, &n, &k, alpha, *((const double**)ai), lda, *((const double**)bi), ldb, beta, *((double**)ci), ldc);
          }
          ai = an; bi = bn; ci = cn; /* next */
        }
      }
      else { /* incorrect argument(s) */
        result = EXIT_FAILURE;
      }
    }
  }
  else { /* incorrect argument(s) */
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_smmbatch_blas(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const float* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb, const float* beta, void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize)
{
  int result = EXIT_SUCCESS;

  if (0 != a && 0 != b && 0 != c) {
    const libxsmm_blasint end = LIBXSMM_ABS(batchsize);
    libxsmm_blasint i;

    if (0 != index_stride) { /* stride arrays contain indexes */
      if (((int)sizeof(libxsmm_blasint)) <= index_stride) {
        const libxsmm_blasint da = (0 != stride_a ? (*stride_a - index_base) : 0);
        const libxsmm_blasint db = (0 != stride_b ? (*stride_b - index_base) : 0);
        const libxsmm_blasint dc = (0 != stride_c ? (*stride_c - index_base) : 0);
        const char *const sa = (const char*)stride_a, *const sb = (const char*)stride_b, *const sc = (const char*)stride_c;
        const float *a0 = (const float*)a, *b0 = (const float*)b, *ai = a0 + da, *bi = b0 + db;
        float *c0 = (float*)c, *ci = c0 + dc;
        for (i = 0; i < end; ++i) {
          const libxsmm_blasint ii = (i + 1) * index_stride;
          const float *const an = a0 + (0 != stride_a ? (*((const libxsmm_blasint*)(sa + ii)) - index_base) : 0);
          const float *const bn = b0 + (0 != stride_b ? (*((const libxsmm_blasint*)(sb + ii)) - index_base) : 0);
          float       *const cn = c0 + (0 != stride_c ? (*((const libxsmm_blasint*)(sc + ii)) - index_base) : 0);
#if defined(LIBXSMM_GEMM_CHECK)
          if (0 != ai && 0 != bi && 0 != ci)
#endif
          {
            libxsmm_blas_sgemm(transa, transb, &m, &n, &k, alpha, ai, lda, bi, ldb, beta, ci, ldc);
          }
          ai = an; bi = bn; ci = cn;
        }
      }
      else { /* incorrect argument(s) */
        result = EXIT_FAILURE;
      }
    }
    else { /* singular strides are measured in Bytes */
      const libxsmm_blasint stride_unit = sizeof(libxsmm_blasint);
      const libxsmm_blasint da = (0 != stride_a ? (*stride_a - index_base * stride_unit) : 0);
      const libxsmm_blasint db = (0 != stride_b ? (*stride_b - index_base * stride_unit) : 0);
      const libxsmm_blasint dc = (0 != stride_c ? (*stride_c - index_base * stride_unit) : 0);
      if (((int)sizeof(float)) <= LIBXSMM_MIN(LIBXSMM_MIN(da, db), dc)) {
        const char *a0 = (const char*)a, *b0 = (const char*)b, *ai = a0, *bi = b0;
        char *c0 = (char*)c, *ci = c0;
        for (i = 0; i < end; ++i) {
          const char *const an = ai + da;
          const char *const bn = bi + db;
          char *const cn = ci + dc;
#if defined(LIBXSMM_GEMM_CHECK)
          if (0 != *((const float**)ai) && 0 != *((const float**)bi) && 0 != *((const float**)ci))
#endif
          {
            libxsmm_blas_sgemm(transa, transb, &m, &n, &k, alpha, *((const float**)ai), lda, *((const float**)bi), ldb, beta, *((float**)ci), ldc);
          }
          ai = an; bi = bn; ci = cn; /* next */
        }
      }
      else { /* incorrect argument(s) */
        result = EXIT_FAILURE;
      }
    }
  }
  else { /* incorrect argument(s) */
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_DEFINITION void libxsmm_gemm_batch(libxsmm_gemm_precision precision,
  const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc, libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize)
{
  libxsmm_gemm_descriptor descriptor;
  const int flags = LIBXSMM_GEMM_PFLAGS(transa, transb, LIBXSMM_FLAGS), prefetch = LIBXSMM_PREFETCH_AUTO;
  int result = libxsmm_gemm_descriptor_init(&descriptor, precision, m, n, k, lda, ldb, ldc, alpha, beta, &flags, &prefetch);
  const libxsmm_xmmfunction nullfn = { 0 }, kernel = (EXIT_SUCCESS == result ? libxsmm_xmmdispatch(&descriptor) : nullfn);
  static int error_once = 0;

  if (0 != kernel.xmm) {
    result = libxsmm_mmbatch(kernel, index_base, index_stride,
      stride_a, stride_b, stride_c, a, b, c, batchsize,
      0/*tid*/, 1/*nthreads*/);
  }
  else { /* fall-back */
    switch (precision) {
      case LIBXSMM_GEMM_PRECISION_F64: {
        result = libxsmm_dmmbatch_blas(transa, transb, m, n, k,
          (const double*)alpha, a, lda, b, ldb, (const double*)beta, c, ldc,
          index_base, index_stride, stride_a, stride_b, stride_c, batchsize);
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
        result = libxsmm_smmbatch_blas(transa, transb, m, n, k,
          (const float*)alpha, a, lda, b, ldb, (const float*)beta, c, ldc,
          index_base, index_stride, stride_a, stride_b, stride_c, batchsize);
      } break;
      default: result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS != result
    && 0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: libxsmm_gemm_batch failed!\n");
  }
}


#if defined(LIBXSMM_BUILD)

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_sgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*,
  const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_sgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_dgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*,
  const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_dgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_blas_sgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*,
  const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_blas_sgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_blas_dgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*,
  const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_blas_dgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_mmbatch)(libxsmm_xmmfunction kernel, const libxsmm_blasint* index_base,
  const libxsmm_blasint* index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const void* a, const void* b, void* c, const libxsmm_blasint* batchsize, const int* tid, const int* nthreads);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_mmbatch)(libxsmm_xmmfunction kernel, const libxsmm_blasint* index_base,
  const libxsmm_blasint* index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const void* a, const void* b, void* c, const libxsmm_blasint* batchsize, const int* tid, const int* nthreads)
{
  static int error_once = 0;
  assert(0 != a && 0 != b && 0 != c && 0 != index_base && 0 != index_stride && 0 != batchsize && 0 != tid && 0 != nthreads);
  if (EXIT_SUCCESS != libxsmm_mmbatch(kernel, *index_base, *index_stride, stride_a, stride_b, stride_c, a, b, c, *batchsize, *tid, *nthreads)
    && 0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: libxsmm_mmbatch failed!\n");
  }
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_gemm_batch)(const libxsmm_gemm_precision* precision,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc, const libxsmm_blasint* index_base, const libxsmm_blasint* index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const libxsmm_blasint* batchsize);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_gemm_batch)(const libxsmm_gemm_precision* precision,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc, const libxsmm_blasint* index_base, const libxsmm_blasint* index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const libxsmm_blasint* batchsize)
{
  assert(0 != precision && 0 != m && 0 != n && 0 != k && 0 != index_base && 0 != index_stride && 0 != batchsize);
  libxsmm_gemm_batch(*precision, transa, transb, *m, *n, *k, alpha, a, lda, b, ldb, beta, c, ldc,
    *index_base, *index_stride, stride_a, stride_b, stride_c, *batchsize);
}

#endif /*defined(LIBXSMM_BUILD)*/

