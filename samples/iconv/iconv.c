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
#include <libxsmm_mhd.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(_WIN32)
# include <io.h>
# if !defined(F_OK)
#   define F_OK 0
# endif
# define FEXIST(FILENAME) _access(FILENAME, F_OK)
#else
# include <unistd.h>
# define FEXIST(FILENAME) access(FILENAME, F_OK)
#endif
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if 1
# define MALLOC(SIZE) libxsmm_aligned_malloc(SIZE, 0/*auto*/)
# define FREE(POINTER) libxsmm_free(POINTER)
#else
# define MALLOC(SIZE) malloc(SIZE)
# define FREE(POINTER) free(POINTER)
#endif

#if !defined(USE_OUTPUT_PADDING) && 0
# define USE_OUTPUT_PADDING
#endif

#if !defined(USE_OVERWRITE)
# define USE_OVERWRITE
#endif


int main(int argc, char* argv[])
{
  const char* filename_in = (1 < argc ? argv[1] : "iconv_in.mhd");
  const size_t nrepeat = (size_t)LIBXSMM_MAX(2 < argc ? strtoul(argv[2], 0, 10) : 1, 1);
  const int kw = LIBXSMM_MAX(3 < argc ? atoi(argv[3]) : 39, 1);
  const int kh = LIBXSMM_MAX(4 < argc ? atoi(argv[4]) : kw, 1);
  const char *const filename_out = (5 < argc ? argv[5] : "iconv_out.mhd");
  int result = 0 != strcmp(filename_in, filename_out) ? EXIT_SUCCESS : EXIT_FAILURE;
  size_t ndims = 2, size_in[2] = { 0 }, size_out[2], pitch[2], offset[2], ncomponents = 1, header_size = 0, extension_size;
  void *conv_input_buffer = 0, *conv_filter_buffer = 0, *conv_output_buffer = 0;
  libxsmm_dnn_tensor *conv_input = 0, *conv_output = 0, *conv_filter = 0;
  libxsmm_mhd_elemtype type_in = LIBXSMM_MHD_ELEMTYPE_UNKNOWN;
  libxsmm_dnn_datatype type_dnn = LIBXSMM_DNN_DATATYPE_F32;
  libxsmm_dnn_conv_desc descriptor = { 0 };
  libxsmm_dnn_layer* handle = 0;
  libxsmm_dnn_err_t status;
  size_t size1 = 0, typesize_dnn = 0;
  size_t conv_output_size1 = 0, i, j;
  unsigned long long start;
  char filename[1024];
  double duration = 0;
  void *filter = 0;
  void *image = 0;
#if !defined(NDEBUG)
  static int error_once = 0;
#endif

  const char *const env_mult = getenv("MULT"), *const env_orig = getenv("ORIG");
  /* extents of result image become multiples of block-size */
  const int mult = ((0 == env_mult || 0 == *env_mult) ? 64/*default*/ : LIBXSMM_MAX(atoi(env_mult), 0));
  /* save result image with original compute-type (type_dnn), and not with pixel-type of input (type_in) */
  const int orig = ((0 == env_orig || 0 == *env_orig) ? 0/*disabled*/ : atoi(env_orig));

  /* Generate an input file if a pseudo filename (resolution) is given. */
  if (0 != FEXIST(filename_in) && 0 < atoi(filename_in)) {
    const char* split = strchr(filename_in, 'x');
    if (0 == split) split = strchr(filename_in, 'X');
    size_in[0] = atoi(filename_in);
    size_in[1] = (0 != split ? atoi(split + 1) : 0);
    if (0 == size_in[1]) size_in[1] = size_in[0];
    image = MALLOC(size_in[0] * size_in[1]);
    if (0 < sprintf(filename, "%s.mhd", filename_in) && 0 != image) {
      const int c0 = 0, c1 = 255, r = LIBXSMM_MAX(kw, kh);
      for (i = 0; i < size_in[1]; ++i) {
        for (j = 0; j < size_in[0]; ++j) {
          ((unsigned char*)image)[i*size_in[0]+j] = (unsigned char)(0 == (i + j) % r ? c1 : c0);
        }
      }
      result = libxsmm_mhd_write(filename, 0/*offset*/, size_in, size_in,
        2/*ndims*/, 1/*ncomponents*/, LIBXSMM_MHD_ELEMTYPE_U8, image,
        0/*header_size*/, 0/*extension_header*/, 0/*extension*/, 0/*extension_size*/);
      if (EXIT_SUCCESS == result) filename_in = filename;
    }
    else {
      result = EXIT_FAILURE;
    }
    FREE(image);
  }

  /* Read MHD-header information; function includes various sanity checks. */
  if (EXIT_SUCCESS == result) {
    result = libxsmm_mhd_read_header(filename_in, sizeof(filename),
      filename, &ndims, size_in, &ncomponents, &type_in,
      &header_size, &extension_size);
  }

  /* Only accept 2d-images (maybe a slice of a 3d-image). */
  if (2 == ndims) {
    const int m = LIBXSMM_MAX(mult, 1);
    offset[0] = ((size_in[0] + LIBXSMM_MAX(kw, m) - 1) / m * m - size_in[0] + kw) / 2;
    offset[1] = ((size_in[1] + LIBXSMM_MAX(kh, m) - 1) / m * m - size_in[1] + kh) / 2;
    /* center image inside of (pitched) buffer */
    size_out[0] = size_in[0] + 2 * offset[0];
    size_out[1] = size_in[1] + 2 * offset[1];
#if defined(USE_OUTPUT_PADDING)
    size_out[0] -= (kw / 2) * 2;
    size_out[1] -= (kh / 2) * 2;
    pitch[0] = size_out[0];
    pitch[1] = size_out[1];
#else
    pitch[0] = size_out[0];
    pitch[1] = size_out[1];
    size_out[0] -= (kw / 2) * 2;
    size_out[1] -= (kh / 2) * 2;
#endif
    size1 = pitch[0] * pitch[1] * ncomponents;
  }
  else {
    result = EXIT_FAILURE;
  }

  /* Allocate image data according to the MHD-header information. */
  if (EXIT_SUCCESS == result) {
    const char* ctypename;
    /* DNN type: assume that MHD I/O provides a super-set of types */
    if (0 != libxsmm_mhd_typename((libxsmm_mhd_elemtype)type_dnn, &typesize_dnn, &ctypename)) {
      const size_t filter_size = ncomponents * kh * kw;
      /* print some information about the workload */
      fprintf(stdout, "filename=%s resolution=%ux%u kernel=%ix%i size_in=%.fMB nrepeat=%u (%s)\n",
        filename, (unsigned int)size_in[0], (unsigned int)size_in[1], kw, kh,
        1.0 * (size1 * typesize_dnn) / (1 << 20),
        (unsigned int)nrepeat, ctypename);
      image = MALLOC(size1 * typesize_dnn);
      filter = MALLOC(filter_size * typesize_dnn);
      if (0 != image && 0 != filter) {
        FILE *const file = fopen("iconv_in.txt", "r"); /* convolution-matrix (kh x kw) */
        double weight;
        switch (type_dnn) {
          case LIBXSMM_DNN_DATATYPE_F64: {
            for (i = 0; i < filter_size; ++i) {
              ((double*)filter)[i] = (double)((0 == file || 1 > fscanf(file, "%lf", &weight)) ?
                (0.05 - ((double)rand() / RAND_MAX) * 0.1) : weight);
            }
          } break;
          case LIBXSMM_DNN_DATATYPE_F32: {
            for (i = 0; i < filter_size; ++i) {
              ((float*)filter)[i] = (float)((0 == file || 1 > fscanf(file, "%lf", &weight)) ?
                (0.05 - ((double)rand() / RAND_MAX) * 0.1) : weight);
            }
          } break;
          case LIBXSMM_DNN_DATATYPE_I32: {
            for (i = 0; i < filter_size; ++i) {
              ((int*)filter)[i] = (int)((0 == file || 1 > fscanf(file, "%lf", &weight)) ?
                (255 * (0.05 - ((double)rand() / RAND_MAX) * 0.1)) : weight);
            }
          } break;
          case LIBXSMM_DNN_DATATYPE_I16: {
            for (i = 0; i < filter_size; ++i) {
              ((short*)filter)[i] = (short)((0 == file || 1 > fscanf(file, "%lf", &weight)) ?
                (255 * (0.05 - ((double)rand() / RAND_MAX) * 0.1)) : weight);
            }
          } break;
          case LIBXSMM_DNN_DATATYPE_I8: {
            for (i = 0; i < filter_size; ++i) {
              ((unsigned char*)filter)[i] = (unsigned char)((0 == file || 1 > fscanf(file, "%lf", &weight)) ?
                (255 * (0.05 - ((double)rand() / RAND_MAX) * 0.1)) : weight);
            }
          } break;
          default: result = EXIT_FAILURE;
        }
        if (0 != file && 0 != fclose(file)) result = EXIT_FAILURE;
      }
      else {
        result = EXIT_FAILURE;
      }
    }
    else {
      result = EXIT_FAILURE;
    }
  }

  /* Read the image data according to the header into the allocated buffer. */
  if (EXIT_SUCCESS == result) {
    result = libxsmm_mhd_read(filename,
      offset, size_in, pitch, ndims, ncomponents, header_size, type_in,
      /* eventually perform a type-conversion (type_in != type_dnn) */
      (const libxsmm_mhd_elemtype*)&type_dnn, image,
      0/*handle_element*/, 0/*extension*/, 0/*extension_size*/);
  }

  /* Setup convolution descriptor. */
  if (EXIT_SUCCESS == result) {
#if defined(_OPENMP)
    descriptor.threads = omp_get_max_threads();
#else
    descriptor.threads = 1;
#endif
    descriptor.N = 1; /* number of images */
    descriptor.R = kh; /* kernel height */
    descriptor.S = kw; /* kernel width */
    descriptor.C = (int)ncomponents; /* in */
    descriptor.K = descriptor.C; /* no reduction */
    descriptor.u = 1; /* H-stride */
    descriptor.v = 1; /* W-stride */
    descriptor.H = (int)pitch[1];
    descriptor.W = (int)pitch[0];
    descriptor.pad_h = 0;
    descriptor.pad_w = 0;
    descriptor.pad_h_in = 0;
    descriptor.pad_w_in = 0;
#if defined(USE_OUTPUT_PADDING)
    descriptor.pad_h_out = ((descriptor.u - 1) * descriptor.H + descriptor.R - descriptor.u) / 2;
    descriptor.pad_w_out = ((descriptor.v - 1) * descriptor.W + descriptor.S - descriptor.v) / 2;
#else
    descriptor.pad_h_out = 0;
    descriptor.pad_w_out = 0;
#endif
    descriptor.algo = LIBXSMM_DNN_CONV_ALGO_DIRECT/*LIBXSMM_DNN_CONV_ALGO_AUTO*/;
    descriptor.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
    descriptor.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
    descriptor.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
#if defined(USE_OVERWRITE)
    descriptor.options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;
#else
    descriptor.options = LIBXSMM_DNN_CONV_OPTION_NONE;
#endif
    descriptor.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
    descriptor.datatype = LIBXSMM_DNN_DATATYPE_F32;
    handle = libxsmm_dnn_create_conv_layer(descriptor, &status);
    if (LIBXSMM_DNN_SUCCESS != status) {
      const char *const error_message = libxsmm_dnn_get_error(status);
      fprintf(stderr, "%s\n", error_message);
      if (LIBXSMM_DNN_WARN_FALLBACK != status) result = EXIT_FAILURE;
    }
  }

  /* Link buffers and convert NCHW-image and KCRS-filter to internal format. */
  if (EXIT_SUCCESS == result) {
    libxsmm_dnn_tensor_datalayout* layout;

    /* Input buffer */
    conv_input_buffer = MALLOC(descriptor.N * descriptor.C * (descriptor.H + 2 * descriptor.pad_h_in) * (descriptor.W + 2 * descriptor.pad_w_in) * typesize_dnn);
    if (0 == conv_input_buffer) result = EXIT_FAILURE;
    layout = libxsmm_dnn_create_tensor_datalayout(handle, LIBXSMM_DNN_INPUT, &status);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
    conv_input = libxsmm_dnn_link_tensor(layout, conv_input_buffer, &status);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxsmm_dnn_destroy_tensor_datalayout(layout);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxsmm_dnn_bind_tensor(handle, conv_input, LIBXSMM_DNN_REGULAR_INPUT);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxsmm_dnn_copyin_tensor(conv_input, image, LIBXSMM_DNN_TENSOR_FORMAT_NCHW);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;

    /* Filter buffer */
    conv_filter_buffer = MALLOC(descriptor.K * descriptor.C * descriptor.R * descriptor.S * typesize_dnn);
    if (0 == conv_filter_buffer) result = EXIT_FAILURE;
    layout = libxsmm_dnn_create_tensor_datalayout(handle, LIBXSMM_DNN_FILTER, &status);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
    conv_filter = libxsmm_dnn_link_tensor(layout, conv_filter_buffer, &status);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxsmm_dnn_destroy_tensor_datalayout(layout);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxsmm_dnn_bind_tensor(handle, conv_filter, LIBXSMM_DNN_REGULAR_FILTER);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxsmm_dnn_copyin_tensor(conv_filter, filter, LIBXSMM_DNN_TENSOR_FORMAT_KCRS);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;

    /* Output buffer */
    conv_output_size1 = descriptor.N * descriptor.K * (descriptor.H + 2 * descriptor.pad_h_out) * (descriptor.W + 2 * descriptor.pad_w_out);
    conv_output_buffer = MALLOC(conv_output_size1 * typesize_dnn);
    if (0 == conv_output_buffer) result = EXIT_FAILURE;
    layout = libxsmm_dnn_create_tensor_datalayout(handle, LIBXSMM_DNN_OUTPUT, &status);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
    conv_output = libxsmm_dnn_link_tensor(layout, conv_output_buffer, &status);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxsmm_dnn_destroy_tensor_datalayout(layout);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxsmm_dnn_bind_tensor(handle, conv_output, LIBXSMM_DNN_REGULAR_OUTPUT);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
  }

  /* Attempt to run the convolution. */
  start = libxsmm_timer_tick();
  for (i = 0; i < nrepeat && EXIT_SUCCESS == result; ++i) {
#if defined(_OPENMP)
# pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
#if !defined(USE_OVERWRITE)
      memset(conv_output_buffer, 0, conv_output_size1 * typesize_dnn);
#endif
      {
#if !defined(NDEBUG)
        const libxsmm_dnn_err_t r =
#endif
        libxsmm_dnn_execute_st(handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid);
#if !defined(NDEBUG)
        if (LIBXSMM_DNN_SUCCESS != r && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
          const char *const error_message = libxsmm_dnn_get_error(r);
          fprintf(stderr, "%s\n", error_message);
          result = EXIT_FAILURE;
        }
#endif
      }
    }
  }
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick());

  /* Copy-out image into original format. */
  if (EXIT_SUCCESS == result) {
    status = libxsmm_dnn_copyout_tensor(conv_output, image, LIBXSMM_DNN_TENSOR_FORMAT_NCHW);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
  }

  /* Write the image into a different file. */
  if (EXIT_SUCCESS == result) {
    if (0 == mult) {
      offset[0] = (size_out[0] - size_in[0]) / 2;
      offset[1] = (size_out[1] - size_in[1]) / 2;
    }
    else {
      offset[0] = offset[1] = 0;
      size_in[0] = size_out[0];
      size_in[1] = size_out[1];
    }
    result = libxsmm_mhd_write(filename_out, offset, size_in, size_out, 2/*ndims*/, ncomponents,
      (libxsmm_mhd_elemtype)type_dnn/* assume MHD I/O provides a super-set of DNN types */,
      image, &header_size, 0/*extension_header*/, 0/*extension*/, 0/*extension_size*/);
  }

  /* convert into input pixel-type, and re-write result image. */
  if (EXIT_SUCCESS == result && 0 == orig && (((int)type_in) != ((int)type_dnn))) {
    size_t typesize_in;
    if (0 != libxsmm_mhd_typename(type_in, &typesize_in, 0/*ctypename*/)) {
      /* we do not want to convert to a larger type (typesize can be equal, but type may be different) */
      if (typesize_in <= typesize_dnn) { /* make sure we can reuse image buffer */
        result = libxsmm_mhd_read(filename_out,
          0/*offset*/, size_in, size_in, 2/*ndims*/, ncomponents, header_size,
          (libxsmm_mhd_elemtype)type_dnn, &type_in, /* conversion requested */
          image, 0/*handle_element*/, 0/*extension*/, 0/*extension_size*/);
        if (EXIT_SUCCESS == result) {
          result = libxsmm_mhd_write(filename_out,
            0/*offset*/, size_in, size_in, 2/*ndims*/, ncomponents, type_in, image,
            0/*header_size*/, 0/*extension_header*/, 0/*extension*/, 0/*extension_size*/);
        }
      }
    }
    else {
      result = EXIT_FAILURE;
    }
  }

  /* Release resources. */
  if (LIBXSMM_DNN_SUCCESS != libxsmm_dnn_release_tensor(handle, LIBXSMM_DNN_REGULAR_INPUT)) result = EXIT_FAILURE;
  if (LIBXSMM_DNN_SUCCESS != libxsmm_dnn_release_tensor(handle, LIBXSMM_DNN_REGULAR_FILTER)) result = EXIT_FAILURE;
  if (LIBXSMM_DNN_SUCCESS != libxsmm_dnn_release_tensor(handle, LIBXSMM_DNN_REGULAR_OUTPUT)) result = EXIT_FAILURE;
  if (LIBXSMM_DNN_SUCCESS != libxsmm_dnn_destroy_tensor(conv_filter)) result = EXIT_FAILURE;
  if (LIBXSMM_DNN_SUCCESS != libxsmm_dnn_destroy_tensor(conv_output)) result = EXIT_FAILURE;
  if (LIBXSMM_DNN_SUCCESS != libxsmm_dnn_destroy_tensor(conv_input)) result = EXIT_FAILURE;
  if (LIBXSMM_DNN_SUCCESS != libxsmm_dnn_destroy_conv_layer(handle)) result = EXIT_FAILURE;
  FREE(conv_output_buffer);
  FREE(conv_filter_buffer);
  FREE(conv_input_buffer);
  FREE(filter);
  FREE(image);

  if (EXIT_SUCCESS == result) {
    if (0 < duration) {
      fprintf(stdout, "\tfrequency: %.1f Hz\n", nrepeat / duration);
    }
    assert(0 != nrepeat);
    fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration / nrepeat);
    fprintf(stdout, "Finished.\n");
  }
  else {
    fprintf(stdout, "FAILED.\n");
  }

  assert(EXIT_SUCCESS == result);
  return result;
}

