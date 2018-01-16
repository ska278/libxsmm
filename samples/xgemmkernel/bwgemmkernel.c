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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

static unsigned int g_jit_code_reps = 0;

LIBXSMM_INLINE void print_help(void) {
  printf("\n\n");
  printf("Usage (dense*dense=dense):\n");
  printf("    M\n");
  printf("    N\n");
  printf("    K\n");
  printf("    LDA\n");
  printf("    LDB\n");
  printf("    LDC\n");
  printf("    alpha: -1 or 1\n");
  printf("    beta: 0 or 1\n");
  printf("    0: unaligned A, otherwise aligned\n");
  printf("    0: unaligned C, otherwise aligned\n");
  printf("    PREFETCH: nopf (none), pfsigonly, BL2viaC, AL2, curAL2, AL2jpst, AL2_BL2viaC, curAL2_BL2viaC, AL2jpst_BL2viaC, AL1_BL1_CL1\n");
  printf("    PRECISION: I16\n");
  printf("    #repetitions\n");
  printf("\n\n");
}

LIBXSMM_INLINE
void init_short( short*                       io_a,
                 short*                       io_b,
                 int*                         io_c,
                 int*                         io_c_gold,
                 const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int l_i, l_j;

  /* touch A */
  for ( l_i = 0; l_i < (unsigned int)i_xgemm_desc->lda; l_i++) {
    for ( l_j = 0; l_j < (unsigned int)i_xgemm_desc->k; l_j++) {
      io_a[(l_j * i_xgemm_desc->lda) + l_i] = (short)(drand48()*10.0);
    }
  }
  /* touch B */
  for ( l_i = 0; l_i < (unsigned int)i_xgemm_desc->ldb; l_i++ ) {
    for ( l_j = 0; l_j < (unsigned int)i_xgemm_desc->n; l_j++ ) {
      io_b[(l_j * i_xgemm_desc->ldb) + l_i] = (short)(drand48()*10.0);
    }
  }
  /* touch C */
  for ( l_i = 0; l_i < (unsigned int)i_xgemm_desc->ldc; l_i++) {
    for ( l_j = 0; l_j < (unsigned int)i_xgemm_desc->n; l_j++) {
      io_c[(l_j * i_xgemm_desc->ldc) + l_i] = (int)0;
      io_c_gold[(l_j * i_xgemm_desc->ldc) + l_i] = (int)0;
    }
  }
}

LIBXSMM_INLINE
void init_byte( unsigned char*                io_a,
                char*                         io_b,
                int*                          io_c,
                int*                          io_c_gold,
                const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int l_i, l_j;

  /* touch A */
  for ( l_i = 0; l_i < (unsigned int)i_xgemm_desc->lda; l_i++) {
    for ( l_j = 0; l_j < (unsigned int)(i_xgemm_desc->k/2); l_j++) {
      io_a[(l_j * i_xgemm_desc->lda) + l_i] = (unsigned char)(drand48()*10.0);
    }
  }
  /* touch B */
  for ( l_i = 0; l_i < (unsigned int)i_xgemm_desc->ldb; l_i++ ) {
    for ( l_j = 0; l_j < (unsigned int)i_xgemm_desc->n; l_j++ ) {
      io_b[(l_j * i_xgemm_desc->ldb) + l_i] = (char)(drand48()*10.0);
    }
  }
  /* touch C */
  for ( l_i = 0; l_i < (unsigned int)i_xgemm_desc->ldc; l_i++) {
    for ( l_j = 0; l_j < (unsigned int)i_xgemm_desc->n; l_j++) {
      io_c[(l_j * i_xgemm_desc->ldc) + l_i] = (int)0;
      io_c_gold[(l_j * i_xgemm_desc->ldc) + l_i] = (int)0;
    }
  }
}

LIBXSMM_INLINE
void run_gold_short( const short*                   i_a,
                     const short*                   i_b,
                     int*                           o_c,
                     const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int l_m, l_n, l_k, l_k2, l_t, l_k_block = 2;
  double l_runtime;

  const unsigned long long l_start = libxsmm_timer_tick();

  for ( l_t = 0; l_t < g_jit_code_reps; l_t++ ) {
    for ( l_n = 0; l_n < (unsigned int)i_xgemm_desc->n; l_n++ ) {
      for ( l_k = 0; l_k < (unsigned int)(i_xgemm_desc->k/l_k_block); l_k++ ) {
        for ( l_m = 0; l_m < (unsigned int)i_xgemm_desc->m; l_m++ ) {
          for ( l_k2 = 0; l_k2 < l_k_block; l_k2++) {
            o_c[(l_n * i_xgemm_desc->ldc) + l_m] += i_a[(l_k * (i_xgemm_desc->lda*l_k_block)) + (l_m*l_k_block) + l_k2] * i_b[(l_n * i_xgemm_desc->ldb) + (l_k*l_k_block) + l_k2];
          }
        }
      }
    }
  }

  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  printf("%fs for C\n", l_runtime);
  printf("%f GOPS for C\n", ((double)((double)g_jit_code_reps * (double)i_xgemm_desc->m * (double)i_xgemm_desc->n * (double)i_xgemm_desc->k) * 2.0) / (l_runtime * 1.0e9));
}

LIBXSMM_INLINE
void run_gold_byte( const unsigned char*          i_a,
                    const char*                   i_b,
                    int*                          o_c,
                    const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int l_m, l_n, l_k, l_k2, l_t, l_k_block = 4;
  double l_runtime;

  const unsigned long long l_start = libxsmm_timer_tick();

  for ( l_t = 0; l_t < g_jit_code_reps; l_t++ ) {
    for ( l_n = 0; l_n < (unsigned int)i_xgemm_desc->n; l_n++ ) {
      for ( l_k = 0; l_k < (unsigned int)(i_xgemm_desc->k/l_k_block); l_k++ ) {
        for ( l_m = 0; l_m < (unsigned int)i_xgemm_desc->m; l_m++ ) {
          for ( l_k2 = 0; l_k2 < l_k_block; l_k2++) {
            o_c[(l_n * i_xgemm_desc->ldc) + l_m] += i_a[(l_k * (i_xgemm_desc->lda*l_k_block)) + (l_m*l_k_block) + l_k2] * i_b[(l_n * i_xgemm_desc->ldb) + (l_k*l_k_block) + l_k2];
          }
        }
      }
    }
  }

  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  printf("%fs for C\n", l_runtime);
  printf("%f GOPS for C\n", ((double)((double)g_jit_code_reps * (double)i_xgemm_desc->m * (double)i_xgemm_desc->n * (double)i_xgemm_desc->k) * 2.0) / (l_runtime * 1.0e9));
}

LIBXSMM_INLINE
void run_jit_short( const short*                     i_a,
                    const short*                     i_b,
                    int*                             o_c,
                    const int                        i_M,
                    const int                        i_N,
                    const int                        i_K,
                    const libxsmm_gemm_prefetch_type i_prefetch ) {
  /* define function pointer */
  libxsmm_wmmfunction l_test_jit;
  double l_jittime = 0.0, l_runtime = 0.0;
  int l_alpha = 1;
  int l_beta = 1;
  unsigned long long l_start;
  unsigned int l_t;

  if ( !(LIBXSMM_FEQ(l_beta, 0.0) || LIBXSMM_FEQ(l_beta, 1.0)) ) {
    fprintf(stderr, "JIT double: beta needs to be 0.0 or 1.0!\n");
    exit(-1);
  }
  if ( !LIBXSMM_FEQ(l_alpha, 1.0) ) {
    fprintf(stderr, "JIT double: alpha needs to be 1.0!\n");
    exit(-1);
  }

  l_start = libxsmm_timer_tick();
  l_test_jit = libxsmm_wmmdispatch(i_M, i_N, i_K, &i_M, &i_K, &i_M, &l_alpha, &l_beta, NULL, &i_prefetch );
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  printf("function pointer address: %llx\n", (unsigned long long)l_test_jit);

  if (l_test_jit == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(-1);
  }

  l_start = libxsmm_timer_tick();

  if ( i_prefetch == LIBXSMM_PREFETCH_NONE ) {
    for ( l_t = 0; l_t < g_jit_code_reps; l_t++ ) {
      l_test_jit(i_a, i_b, o_c);
    }
  } else {
    for ( l_t = 0; l_t < g_jit_code_reps; l_t++ ) {
      l_test_jit(i_a, i_b, o_c, i_a, i_b, o_c);
    }
  }

  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  printf("%fs for creating jit\n", l_jittime);
  printf("%fs for executing jit\n", l_runtime);
  printf("%f GOPS for jit\n", ((double)((double)g_jit_code_reps * (double)i_M * (double)i_N * (double)i_K) * 2.0) / (l_runtime * 1.0e9));
}

#if 0
LIBXSMM_INLINE
void run_jit_float( const float*                     i_a,
                    const float*                     i_b,
                    float*                           o_c,
                    const int                        i_M,
                    const int                        i_N,
                    const int                        i_K,
                    const libxsmm_gemm_prefetch_type i_prefetch ) {
  /* define function pointer */
  libxsmm_smmfunction l_test_jit;
  double l_jittime = 0.0, l_runtime = 0.0;
  float l_alpha = 1.0f;
  float l_beta = 1.0f;
  unsigned long long l_start;
  unsigned int l_t;

  if ( !(LIBXSMM_FEQ(l_beta, 0.0f) || LIBXSMM_FEQ(l_beta, 1.0f)) ) {
    fprintf(stderr, "JIT float: beta needs to be 0.0 or 1.0!\n");
    exit(-1);
  }
  if ( !LIBXSMM_FEQ(l_alpha, 1.0f) ) {
    fprintf(stderr, "JIT float: alpha needs to be 1.0!\n");
    exit(-1);
  }

  l_start = libxsmm_timer_tick();
  l_test_jit = libxsmm_smmdispatch(i_M, i_N, i_K, &i_M, &i_K, &i_M, &l_alpha, &l_beta, NULL, &i_prefetch );
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  printf("function pointer address: %llx\n", (unsigned long long)l_test_jit);

  if (l_test_jit == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(-1);
  }

  l_start = libxsmm_timer_tick();

  if ( i_prefetch == LIBXSMM_PREFETCH_NONE ) {
    for ( l_t = 0; l_t < g_jit_code_reps; l_t++ ) {
      l_test_jit(i_a, i_b, o_c);
    }
  } else {
    for ( l_t = 0; l_t < g_jit_code_reps; l_t++ ) {
      l_test_jit(i_a, i_b, o_c, i_a, i_b, o_c);
    }
  }

  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  printf("%fs for creating jit\n", l_jittime);
  printf("%fs for executing jit\n", l_runtime);
  printf("%f GOPS for jit\n", ((double)((double)g_jit_code_reps * (double)i_M * (double)i_N * (double)i_K) * 2.0) / (l_runtime * 1.0e9));
}
#endif

LIBXSMM_INLINE
void max_error( const int*                     i_c,
                const int*                     i_c_gold,
                const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int l_i, l_j;
  double l_max_error = 0.0;

  for ( l_i = 0; l_i < (unsigned int)i_xgemm_desc->m; l_i++) {
    for ( l_j = 0; l_j < (unsigned int)i_xgemm_desc->n; l_j++) {
      const double error = LIBXSMM_ABS( i_c_gold[(l_j * i_xgemm_desc->ldc) + l_i] - i_c[(l_j * i_xgemm_desc->ldc) + l_i]);
#if 0
      printf("Entries in row %i, column %i, gold: %f, jit: %f\n", l_i+1, l_j+1, i_c_gold[(l_j*i_xgemm_desc->ldc)+l_i], i_c[(l_j*i_xgemm_desc->ldc)+l_i]);
#endif
      if (l_max_error < error) l_max_error = error;
    }
  }

  printf("max. error: %f\n", l_max_error);
}


int main(int argc, char* argv []) {
  char* l_precision = NULL;
  int l_m = 0;
  int l_n = 0;
  int l_k = 0;
  int l_lda = 0;
  int l_ldb = 0;
  int l_ldc = 0;
  int l_aligned_a = 0;
  int l_aligned_c = 0;
  int l_alpha = 0;
  int l_beta = 0;
  int l_short_precision = 0;
  libxsmm_gemm_prefetch_type l_prefetch = LIBXSMM_PREFETCH_NONE;

  libxsmm_gemm_descriptor l_xgemm_desc;
  /* init data structures */
  int* l_c_gold_w = 0;
  short* l_a_w = 0;
  short* l_b_w;
  int* l_c_w;
  int* l_c_gold_b = 0;
  unsigned char* l_a_b = 0;
  char* l_b_b = 0;
  int* l_c_b = 0;

  /* check argument count for a valid range */
  if ( argc != 14 ) {
    print_help();
    return -1;
  }

  /* xgemm sizes */
  l_m = atoi(argv[1]);
  l_n = atoi(argv[2]);
  l_k = atoi(argv[3]);
  l_lda = atoi(argv[4]);
  l_ldb = atoi(argv[5]);
  l_ldc = atoi(argv[6]);

  /* some sugar */
  l_alpha = atoi(argv[7]);
  l_beta = atoi(argv[8]);
  l_aligned_a = atoi(argv[9]);
  l_aligned_c = atoi(argv[10]);

  /* arch specific stuff */
  l_precision = argv[12];
  g_jit_code_reps = atoi(argv[13]);

  /* set value of prefetch flag */
  if (strcmp("nopf", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_NONE;
  }
  else if (strcmp("pfsigonly", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_SIGONLY;
  }
  else if (strcmp("BL2viaC", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_BL2_VIA_C;
  }
  else if (strcmp("curAL2", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2_AHEAD;
  }
  else if (strcmp("curAL2_BL2viaC", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD;
  }
  else if (strcmp("AL2", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2;
  }
  else if (strcmp("AL2_BL2viaC", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C;
  }
  else if (strcmp("AL2jpst", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2_JPST;
  }
  else if (strcmp("AL2jpst_BL2viaC", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST;
  }
  else if (strcmp("AL1_BL1_CL1", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL1_BL1_CL1;
  }
  else {
    print_help();
    return -1;
  }

  /* check and evaluate precision flag */
  if ( strcmp(l_precision, "I16") == 0 ) {
    l_short_precision = 1;
  } else if ( strcmp(l_precision, "I8") == 0 ) {
    l_short_precision = 0;
  } else {
    print_help();
    return -1;
  }

  /* check alpha */
  if ((l_alpha != 1)) {
    print_help();
    return -1;
  }

  /* check beta */
  if ((l_beta != 0) && (l_beta != 1)) {
    print_help();
    return -1;
  }

  /* short is only supported for now */
  if ( l_short_precision == 0 ) {
    print_help();
    return -1;
  }

  LIBXSMM_GEMM_DESCRIPTOR(l_xgemm_desc, 0 == l_short_precision ? LIBXSMM_GEMM_PRECISION_I16 : LIBXSMM_GEMM_PRECISION_I16,
    (0 != l_aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0) | (0 != l_aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0),
    l_m, l_n, l_k, l_lda, l_ldb, l_ldc,
    l_alpha, l_beta, l_prefetch);

  if ( l_short_precision == 1 ) {
    l_a_w = (short*)libxsmm_aligned_malloc(l_xgemm_desc.lda * l_xgemm_desc.k * sizeof(short), 64);
    l_b_w = (short*)libxsmm_aligned_malloc(l_xgemm_desc.ldb * l_xgemm_desc.n * sizeof(short), 64);
    l_c_w = (int*)libxsmm_aligned_malloc(l_xgemm_desc.ldc * l_xgemm_desc.n * sizeof(int), 64);
    l_c_gold_w = (int*)libxsmm_aligned_malloc(l_xgemm_desc.ldc * l_xgemm_desc.n * sizeof(int), 64);
    init_short(l_a_w, l_b_w, l_c_w, l_c_gold_w, &l_xgemm_desc);
  } else {
#if 0
    l_a_b = (unsigned char*)libxsmm_aligned_malloc(l_xgemm_desc.lda * l_xgemm_desc.k * sizeof(unsigned char), 64);
    l_b_b = (char*)libxsmm_aligned_malloc(l_xgemm_desc.ldb * l_xgemm_desc.n * sizeof(char), 64);
    l_c_b = (int*)libxsmm_aligned_malloc(l_xgemm_desc.ldc * l_xgemm_desc.n * sizeof(int), 64);
    l_c_gold_b = (int*)libxsmm_aligned_malloc(l_xgemm_desc.ldc * l_xgemm_desc.n * sizeof(int), 64);
    init_byte(l_a_b, l_b_b, l_c_b, l_c_gold_b, &l_xgemm_desc);
#endif
  }

  /* print some output... */
  printf("------------------------------------------------\n");
  printf("RUNNING (%ux%u) X (%ux%u) = (%ux%u)", l_xgemm_desc.m, l_xgemm_desc.k, l_xgemm_desc.k, l_xgemm_desc.n, l_xgemm_desc.m, l_xgemm_desc.n);
  if ( l_short_precision == 1 ) {
    printf(", int16\n");
  } else {
    printf(", int8\n");
  }
  printf("------------------------------------------------\n");

  /* run C */
  if ( l_short_precision == 1 ) {
    run_gold_short( l_a_w, l_b_w, l_c_gold_w, &l_xgemm_desc );
  } else {
    /*run_gold_byte( l_a_b, l_b_b, l_c_gold_b, &l_xgemm_desc );*/
  }

  /* run jit */
  if ( l_short_precision == 1 ) {
    run_jit_short( l_a_w, l_b_w, l_c_w, l_m, l_n, l_k, l_prefetch );
  } else {
    /*run_jit_byte( l_a_b, l_b_b, l_c_b, l_m, l_n, l_k, l_prefetch );*/
  }

  /* test result */
  if ( l_short_precision == 1 ) {
    max_error( l_c_w, l_c_gold_w, &l_xgemm_desc );
  } else {
    /*max_error( l_c_b, l_c_gold_b, &l_xgemm_desc );*/
  }

  /* free */
  if ( l_short_precision == 1 ) {
    libxsmm_free(l_a_w);
    libxsmm_free(l_b_w);
    libxsmm_free(l_c_w);
    libxsmm_free(l_c_gold_w);
  } else {
    libxsmm_free(l_a_b);
    libxsmm_free(l_b_b);
    libxsmm_free(l_c_b);
    libxsmm_free(l_c_gold_b);
  }

  printf("------------------------------------------------\n");
  return 0;
}

