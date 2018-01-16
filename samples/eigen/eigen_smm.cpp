/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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

/** This sample uses LIBXSMM's header-only implementation. */
#include <libxsmm_source.h>

#if !defined(__EIGEN)
/*# define __EIGEN*/
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if defined(__EIGEN)
# include <Eigen/Dense>
#endif
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(REAL_TYPE)
# define REAL_TYPE double
#endif


#if defined(__EIGEN)

template<typename T>
LIBXSMM_INLINE LIBXSMM_RETARGETABLE
void smm_eigen_dynamic(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const T *LIBXSMM_RESTRICT a, const T *LIBXSMM_RESTRICT b, T *LIBXSMM_RESTRICT c)
{
  typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor|Eigen::AutoAlign> matrix_type;
  matrix_type::Map(c, m, n).noalias() =
    matrix_type::Map(a, m, k) *
    matrix_type::Map(b, k, n);
}

#endif /*defined(__EIGEN)*/


template<typename T>
LIBXSMM_INLINE LIBXSMM_RETARGETABLE
void smm_xsmm_specialized(const libxsmm_mmfunction<T>& xmm,
  const T *LIBXSMM_RESTRICT a, const T *LIBXSMM_RESTRICT b, T *LIBXSMM_RESTRICT c,
  const T* next_a, const T* next_b, const T* next_c)
{
#if (0 != LIBXSMM_PREFETCH)
  xmm(a, b, c, next_a, next_b, next_c);
#else
  xmm(a, b, c);
  LIBXSMM_UNUSED(next_a);
  LIBXSMM_UNUSED(next_b);
  LIBXSMM_UNUSED(next_c);
#endif
}


template<typename T>
LIBXSMM_INLINE LIBXSMM_RETARGETABLE
void init(libxsmm_blasint seed, T *LIBXSMM_RESTRICT dst,
  libxsmm_blasint nrows, libxsmm_blasint ncols, libxsmm_blasint ld, double scale)
{
  const double seed1 = scale * (seed + 1);
  libxsmm_blasint i;
  for (i = 0; i < ncols; ++i) {
    libxsmm_blasint j = 0;
    for (; j < nrows; ++j) {
      const libxsmm_blasint k = i * ld + j;
      dst[k] = static_cast<T>(seed1 / (k + 1));
    }
    for (; j < ld; ++j) {
      const libxsmm_blasint k = i * ld + j;
      dst[k] = static_cast<T>(seed);
    }
  }
}


int main(int argc, char* argv[])
{
  int result = EXIT_SUCCESS;
  try {
    typedef REAL_TYPE T;
    const libxsmm_blasint benchmark = 1 < argc ? std::atoi(argv[1]) : 0;
    const libxsmm_blasint m = (2 < argc ? std::atoi(argv[2]) : 23);
    const libxsmm_blasint k = (4 < argc ? std::atoi(argv[4]) : m);
    const libxsmm_blasint n = (3 < argc ? std::atoi(argv[3]) : k);
    const libxsmm_blasint q = (5 < argc ? std::atoi(argv[5]) : 0/*auto*/);
    const libxsmm_blasint nrepeat = (6 < argc ? std::atoi(argv[6]) : (0 >= q ? 13 : 1));

    const libxsmm_blasint lda = m, ldb = k, ldc = m;
    const char transa = 'N', transb = 'N';
    const T alpha = 1, beta = 1;

    const libxsmm_blasint asize = lda * k, bsize = ldb * n, csize = ldc * n, aspace = LIBXSMM_ALIGNMENT / sizeof(T);
    const libxsmm_blasint max_size = ((2ULL << 30/*2 GB*/) / ((asize + bsize + csize) * sizeof(T)));
    const libxsmm_blasint s = LIBXSMM_MIN(0 < q ? q : max_size, max_size);
    const size_t bwsize_batched = static_cast<size_t>((asize/*load*/ + bsize/*load*/ + 2 * csize/*RFO*/) * sizeof(T)); // batched (A, B, and C)
    const size_t bwsize = static_cast<size_t>((asize/*load*/ + bsize/*load*/) * sizeof(T)); // omit size of A, B, or C since it is held in cache
    const double gflops = 2.0 * nrepeat * s * m * n * k * 1E-9, scale = 1.0 / s;
#if defined(_OPENMP)
    const libxsmm_blasint chunksize = s / omp_get_max_threads();
#endif

    struct raii { // avoid std::vector (first-touch init. causes NUMA issue)
      T *a, *b, *c;
      raii(libxsmm_blasint asize_, libxsmm_blasint bsize_, libxsmm_blasint csize_)
        : a(new T[static_cast<size_t>(asize_)]), b(new T[static_cast<size_t>(bsize_)])
        , c(new T[static_cast<size_t>(csize_)]) {}
      ~raii() { delete[] a; delete[] b; delete[] c; }
    } buffer(s * asize + aspace - 1, s * bsize + aspace - 1, s * csize + aspace - 1);
    T *const a = LIBXSMM_ALIGN(buffer.a, LIBXSMM_ALIGNMENT);
    T *const b = LIBXSMM_ALIGN(buffer.b, LIBXSMM_ALIGNMENT);
    T *c = LIBXSMM_ALIGN(buffer.c, LIBXSMM_ALIGNMENT);

#if defined(_OPENMP)
#   pragma omp parallel for schedule(static)
#endif
    for (libxsmm_blasint i = 0; i < s; ++i) {
      init(42 + i, a + i * asize, m, k, lda, scale);
      init(24 + i, b + i * bsize, k, n, ldb, scale);
      init(22 + i, c + i * csize, m, n, ldc, scale);
    }

#if defined(LIBXSMM_OFFLOAD_TARGET)
#   pragma offload target(LIBXSMM_OFFLOAD_TARGET) in(a: length(s * asize)) in(b: length(s * bsize)) inout(c: length(s * csize))
#endif
    {
#if defined(MKL_ENABLE_AVX512)
      mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
      // initialize LIBXSMM
      libxsmm_init();

      fprintf(stdout, "m=%lli n=%lli k=%lli size=%lli memory=%.1f MB (%s)\n\n",
        static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(k), static_cast<long long>(s),
        1.0 * (s * (asize + bsize + csize) * sizeof(T)) / (1 << 20), 8 == sizeof(T) ? "DP" : "SP");

      const libxsmm_mmfunction<T> xmm(LIBXSMM_GEMM_FLAGS(transa, transb), m, n, k, lda, ldb, ldc, alpha, beta, LIBXSMM_PREFETCH_AUTO);

      switch (benchmark) {
      case 0: if (xmm) {
        fprintf(stdout, "LIBXSMM batched (A,B,C)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
            const T *const ai = a + i * asize, *const bi = b + i * bsize;
            T *const ci = c + i * csize;
            smm_xsmm_specialized<T>(xmm, ai, bi, ci,
              LIBXSMM_PREFETCH_A(ai + asize), LIBXSMM_PREFETCH_B(bi + bsize),
              LIBXSMM_PREFETCH_C(ci + csize));
          }
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles);
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f FLOPS/cycle\n", (nrepeat * s * (2.0 * m * n * k - m * n)) / ncycles);
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", nrepeat * s * bwsize_batched / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /*break;*/

#if defined(__EIGEN)
      case 1: {
        fprintf(stdout, "Eigen/dynamic batched (A,B,C)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
            const T *const ai = a + i * asize, *const bi = b + i * bsize;
            T *const ci = c + i * csize;
            smm_eigen_dynamic(m, n, k, ai, bi, ci);
          }
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles);
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f FLOPS/cycle\n", (nrepeat * s * (2.0 * m * n * k - m * n)) / ncycles);
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", nrepeat * s * bwsize_batched / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      }
#endif /*defined(__EIGEN)*/
      break;
      case 2: if (xmm) {
        fprintf(stdout, "LIBXSMM streamed (A,C)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
            const T *const ai = a + i * asize;
            T *const ci = c + i * csize;
            smm_xsmm_specialized<T>(xmm, ai, b, ci,
              LIBXSMM_PREFETCH_A(ai + asize), LIBXSMM_PREFETCH_B(b),
              LIBXSMM_PREFETCH_C(ci + csize));
          }
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles);
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f FLOPS/cycle\n", (nrepeat * s * (2.0 * m * n * k - m * n)) / ncycles);
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", nrepeat * s * bwsize / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /*break;*/

#if defined(__EIGEN)
      case 3: {
        fprintf(stdout, "Eigen/dynamic streamed (A,C)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
            const T *const ai = a + i * asize;
            T *const ci = c + i * csize;
            smm_eigen_dynamic(m, n, k, ai, b, ci);
          }
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles);
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f FLOPS/cycle\n", (nrepeat * s * (2.0 * m * n * k - m * n)) / ncycles);
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", nrepeat * s * bwsize / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      }
#endif /*defined(__EIGEN)*/
      break;
      case 4: if (xmm) {
        fprintf(stdout, "LIBXSMM streamed (B,C)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
            const T *const bi = b + i * bsize;
            T *const ci = c + i * csize;
            smm_xsmm_specialized<T>(xmm, a, bi, ci,
              LIBXSMM_PREFETCH_A(a), LIBXSMM_PREFETCH_B(bi + bsize),
              LIBXSMM_PREFETCH_C(ci + csize));
          }
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles);
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f FLOPS/cycle\n", (nrepeat * s * (2.0 * m * n * k - m * n)) / ncycles);
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", nrepeat * s * bwsize / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /*break;*/

#if defined(__EIGEN)
      case 5: {
        fprintf(stdout, "Eigen/dynamic streamed (B,C)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
            const T *const bi = b + i * bsize;
            T *const ci = c + i * csize;
            smm_eigen_dynamic(m, n, k, a, bi, ci);
          }
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles);
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f FLOPS/cycle\n", (nrepeat * s * (2.0 * m * n * k - m * n)) / ncycles);
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", nrepeat * s * bwsize / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      }
#endif /*defined(__EIGEN)*/
      break;
      case 6: if (xmm) {
        fprintf(stdout, "LIBXSMM streamed (A,B)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
#if defined(_OPENMP) /* attempt to write to disjunct cachelines */
            const libxsmm_blasint j = omp_get_thread_num() * chunksize * csize;
#else
            const libxsmm_blasint j = 0;
#endif
            const T *const ai = a + i * asize, *const bi = b + i * bsize;
            smm_xsmm_specialized<T>(xmm, ai, bi, c + j,
              LIBXSMM_PREFETCH_A(ai + asize), LIBXSMM_PREFETCH_B(bi + bsize),
              LIBXSMM_PREFETCH_C(c + j));
          }
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles);
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f FLOPS/cycle\n", (nrepeat * s * (2.0 * m * n * k - m * n)) / ncycles);
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", nrepeat * s * bwsize / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /*break;*/

#if defined(__EIGEN)
      case 7: {
        fprintf(stdout, "Eigen/dynamic streamed (A,B)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
#if defined(_OPENMP) /* attempt to write to disjunct cachelines */
            const libxsmm_blasint j = omp_get_thread_num() * chunksize * csize;
#else
            const libxsmm_blasint j = 0;
#endif
            const T *const ai = a + i * asize, *const bi = b + i * bsize;
            smm_eigen_dynamic(m, n, k, ai, bi, c + j);
          }
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles);
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f FLOPS/cycle\n", (nrepeat * s * (2.0 * m * n * k - m * n)) / ncycles);
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", nrepeat * s * bwsize / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      }
#endif /*defined(__EIGEN)*/
      break;
      case 8: if (xmm) {
        fprintf(stdout, "LIBXSMM cached...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
#if defined(_OPENMP) /* attempt to write to disjunct cachelines */
            const libxsmm_blasint j = omp_get_thread_num() * chunksize * csize;
#else
            const libxsmm_blasint j = 0;
#endif
            smm_xsmm_specialized<T>(xmm, a, b, c + j,
              LIBXSMM_PREFETCH_A(a), LIBXSMM_PREFETCH_B(b),
              LIBXSMM_PREFETCH_C(c + j));
          }
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles);
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f FLOPS/cycle\n", (nrepeat * s * (2.0 * m * n * k - m * n)) / ncycles);
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /*break;*/

#if defined(__EIGEN)
      case 9: {
        fprintf(stdout, "Eigen/dynamic cached...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
#if defined(_OPENMP) /* attempt to write to disjunct cachelines */
            const libxsmm_blasint j = omp_get_thread_num() * chunksize * csize;
#else
            const libxsmm_blasint j = 0;
#endif
            smm_eigen_dynamic(m, n, k, a, b, c + j);
          }
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles);
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f FLOPS/cycle\n", (nrepeat * s * (2.0 * m * n * k - m * n)) / ncycles);
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      }
#endif /*defined(__EIGEN)*/
      break;
      default: throw "invalid case selected!";
      } /*switch*/

      // finalize LIBXSMM
      libxsmm_finalize();
      fprintf(stdout, "Finished\n");
    }
  }
  catch(const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    result = EXIT_FAILURE;
  }
  catch(const char* message) {
    fprintf(stderr, "Error: %s\n", message);
    result = EXIT_FAILURE;
  }
  catch(...) {
    fprintf(stderr, "Error: unknown exception caught!\n");
    result = EXIT_FAILURE;
  }

  return result;
}

