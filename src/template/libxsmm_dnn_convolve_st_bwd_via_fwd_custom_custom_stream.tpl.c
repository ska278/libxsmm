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
/* Evangelos Georganas (Intel Corp.)
 ******************************************************************************/
#define IMG_LOOP_INIT 0
#define IFM_LOOP_INIT 1
#define IFM_LOOP_CLOSE 2
#define CONVOLUTION_KERNEL 3
#define OFM_LOOP_CLOSE_S 4
#define OFM_LOOP_FIRST_TOUCH 5
#define IMG_LOOP_CLOSE 6

const int ltid = tid-start_thread;

int BLOCKSIFM = handle->blocksifm_lp;
int BLOCKSOFM = handle->blocksofm_lp;
int oKB = handle->desc.K/16;
int iCB = handle->desc.C/16;

/* number of tasks for transpose that could be run in parallel */
int transpose_work;
if (handle->use_lp_kernel == 0) {
  transpose_work = BLOCKSOFM * (BLOCKSIFM * handle->fm_lp_block);
} else {
#if 0
  transpose_work = handle->desc.C * handle->desc.K;
#else
  transpose_work = oKB * iCB;
#endif
}

/* compute chunck size */
const int transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : ((transpose_work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
const int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;
/* Pointer variables  */
element_output_type *input_base;
element_output_type *input_ptr;
element_filter_type *weight_base;
element_input_type *output_base;
element_output_type *copy_ptr;
element_output_type *prefetch_ptr;

/* Padding related variables */
const int padded_h = handle->ofhp + 2 * handle->desc.pad_h;
const int padded_w = handle->ofwp + 2 * handle->desc.pad_w;
LIBXSMM_VLA_DECL(5, element_output_type, output_buffer, ((element_output_type*)handle->scratch5) + ltid * BLOCKSOFM * padded_h * padded_w * handle->ofmblock, padded_h, padded_w, handle->ofmblock_lp, handle->fm_lp_block);

libxsmm_xmcopyfunction jitted_matcopy = handle->matcopy_bwd[0].xmatcopy;
libxsmm_convfunction kernel_bwd = (libxsmm_convfunction)handle->code_bwd[4].xconv.sconv;
libxsmm_convfunction kernel2_bwd = (libxsmm_convfunction)handle->code_bwd[5].xconv.sconv;
libxsmm_convfunction kernel_pool[2];
kernel_pool[0] = kernel_bwd;
kernel_pool[1] = kernel2_bwd;
char *variant = handle->kernel_bwd_variant_ptrs[ltid];

/* Input tensor declaration */
/* regular/high precision */
element_input_type* del_in = 0;
/* select pointer based on precision */
if (handle->datatype_in != handle->datatype_out) {
  del_in = ((element_input_type*)handle->grad_input->data) + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in) * (handle->ifmblock_hp);
} else {
  del_in = ((element_input_type*)handle->grad_input->data) + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in) * (handle->ifmblock);
}

LIBXSMM_ALIGNED(float scale_factor, 64);
if (handle->use_lp_kernel == 1) {
  scale_factor = (float) pow(2.0, -1.0*((double)(handle->reg_filter->scf + handle->grad_output->scf)));
}

LIBXSMM_ALIGNED(float *max_vals, 64);
#ifdef __AVX512F__
__m512 max_abs;
#else
/* won't happen as this code only runs on AVX512 platforms */
#endif
if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
  LIBXSMM_VLA_DECL(2, float, maxstats, (float*)handle->maxstats_bwd->data, handle->ifmblock_hp);
  max_vals = (float*) &LIBXSMM_VLA_ACCESS(2, maxstats, ltid, 0, handle->ifmblock_hp);
#ifdef __AVX512F__
  max_abs = _mm512_setzero_ps();
  _mm512_store_ps(max_vals, max_abs);
#else
/* won't happen as this code only runs on AVX512 platforms */
#endif
}

{ /* open new scope for additional variable declarations (C89) */
  LIBXSMM_VLA_DECL(5, element_input_type, del_input, del_in, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
  /* Ouput tensor declaration */
  element_output_type *const out = ((element_output_type*)handle->grad_output->data) /* + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock * handle->fm_lp_block*/;
  LIBXSMM_VLA_DECL(6, element_output_type, del_out, out, BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);

  /* Weight and transpose_weight tensor declaration */
  LIBXSMM_VLA_DECL(7, element_filter_type, wt, (element_filter_type*)handle->reg_filter->data, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(7, element_filter_type, tr_wt2, (element_filter_type*)handle->scratch1, BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ofmblock_lp, handle->ifmblock_hp, handle->fm_lp_block);

  /* Auxiliary integer variables   */
  int instr, n_segments, offset_i, offset_o, offset_w, pi, po, pw, pc, i,  n_convs, conv_i, ifm1, img = 0, ifm2, ij, ii, ifm1lpblock ;
  int ti, tj, trans_i, n_trans_tasks, trans_offset, trans_offset_dst;
  /* Stream related variables  */
  segment_t *code_stream;
  int *stream = handle->compute_bwd_indices_ptrs[ltid];
  int *trans_indices =  handle->transpose_bwd_indices_ptrs[ltid];
  int pool_index;
  element_filter_type  *mat, *matT;
  int ifm1ofm1, kj, ki, ofm2, ofm1;
  /* Kernel related variables  */
  libxsmm_convfunction kernel = (libxsmm_convfunction)handle->code_bwd[4].xconv.sconv;
  libxsmm_xmcopyfunction jitted_matcopy = handle->matcopy_bwd[0].xmatcopy;
  libxsmm_xmcopyfunction jitted_zero_overwrite = handle->matcopy_bwd[1].xmatcopy;

  /* Initialize base pointers */
  if ( handle->padding_flag == 1  ) {
    input_base = &LIBXSMM_VLA_ACCESS(5, output_buffer, 0, 0, 0, 0, 0,
        padded_h, padded_w, handle->ofmblock_lp, handle->fm_lp_block);
    /* we need to set the scratch to zero */
    /* @TODO: we need to find a better/faster code here, e.g. just setting the rim */
    memset( input_base, 0, BLOCKSOFM * padded_h * padded_w * handle->ofmblock * sizeof(element_output_type) );
  } else {
    input_base = &LIBXSMM_VLA_ACCESS(6, del_out, 0, 0, 0, 0, 0, 0,
        BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);
  }

  output_base = &LIBXSMM_VLA_ACCESS(5, del_input, 0, 0, 0, 0, 0,
      handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
  weight_base = &LIBXSMM_VLA_ACCESS(7, tr_wt2, 0, 0, 0, 0, 0, 0, 0,
      BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ofmblock_lp, handle->ifmblock_hp, handle->fm_lp_block);

  instr = handle->n_entries_bwd[ltid];
  n_segments = handle->n_bwd_code_segments[ltid];
  i = 0;
  code_stream = handle->bwd_code_segments[ltid];
  n_trans_tasks =  handle->n_entries_trans_bwd[ltid];

  /* lazy barrier init */
  libxsmm_barrier_init(handle->barrier, ltid);

  if ( (handle->options & LIBXSMM_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE) > 0 ) {
    weight_base = (element_filter_type*)handle->reg_filter_tr->data;
  } else {
    if (handle->use_lp_kernel == 0) {
      for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
        ofm1 = ifm1ofm1 / BLOCKSIFM;
        ifm1 = ifm1ofm1 % BLOCKSIFM;
        for (kj=0; kj < handle->desc.R; kj++) {
          for (ki=0; ki < handle->desc.S; ki++) {
            for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
              for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                LIBXSMM_VLA_ACCESS(7, tr_wt2, ifm1, ofm1, handle->desc.R-1-kj , handle->desc.S-1-ki, ofm2, ifm2, 0, BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block) =
                  LIBXSMM_VLA_ACCESS(7, wt, ofm1, ifm1, kj, ki, ifm2, ofm2, 0, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
              }
            }
          }
        }
      }
    } else {
      if  (( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32)) && ((handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0))  {
        for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
          ifm1 = ifm1ofm1 / oKB;
          ofm1 = ifm1ofm1 % oKB;
          int fm_lp_ind;
          for (kj=0; kj < handle->desc.R; kj++) {
            for (ki=0; ki < handle->desc.S; ki++) {
              for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                  for (fm_lp_ind = 0; fm_lp_ind < handle->fm_lp_block; fm_lp_ind++) {
                    LIBXSMM_VLA_ACCESS(7, tr_wt2, ifm1, ofm1, handle->desc.R-1-kj , handle->desc.S-1-ki, ofm2/handle->fm_lp_block, ifm2*handle->fm_lp_block+fm_lp_ind, ofm2%handle->fm_lp_block, BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block) =
                      LIBXSMM_VLA_ACCESS(7, wt, ofm1, ifm1, kj, ki, ifm2, ofm2, fm_lp_ind, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
                  }
                }
              }
            }
          }
        }
      } else {
#ifdef __AVX512F__
        int icb, okb, t1, t2, t3;
        const __m512i permute_index = _mm512_set_epi32(15,13,11,9,7,5,3,1,14,12,10,8,6,4,2,0);
        const  __m256i scatter_index = _mm256_set_epi32(7*32, 6*32, 5*32, 4*32,  3*32, 2*32, 1*32, 0*32);
        for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
          icb = ifm1ofm1 / oKB;
          okb = ifm1ofm1 % oKB;
          for (kj=0; kj < handle->desc.R; kj++) {
            for (ki=0; ki < handle->desc.S; ki++) {
              for (t1 = 0; t1 < 8; t1++) {
                __m512i cur_cache_line = _mm512_loadu_si512(&LIBXSMM_VLA_ACCESS(7, wt, okb, icb, kj, ki, t1, 0, 0, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block));
                __m512i permuted_cache_line = LIBXSMM_INTRINSICS_MM512_PERMUTEVAR_EPI32(permute_index, cur_cache_line);
                __m256i lo_half = LIBXSMM_INTRINSICS_MM512_EXTRACTI64x4_EPI64(permuted_cache_line, 0);
                __m256i hi_half = LIBXSMM_INTRINSICS_MM512_EXTRACTI64x4_EPI64(permuted_cache_line, 1);
                __m256i lo_zipped = _mm256_unpacklo_epi16(lo_half, hi_half);
                __m256i hi_zipped = _mm256_unpackhi_epi16(lo_half, hi_half);
                __m128i part0 = _mm256_extractf128_si256(lo_zipped,0);
                __m128i part2 = _mm256_extractf128_si256(lo_zipped,1);
                __m128i part1 = _mm256_extractf128_si256(hi_zipped,0);
                __m128i part3 =  _mm256_extractf128_si256(hi_zipped,1);
                __m512i compact = _mm512_inserti32x4 (compact, part0, 0);
                compact = _mm512_inserti32x4 (compact, part1, 1);
                compact = _mm512_inserti32x4 (compact, part2, 2);
                compact = _mm512_inserti32x4 (compact, part3, 3);
                _mm512_i32scatter_epi64(&LIBXSMM_VLA_ACCESS(7, tr_wt2, icb, okb, handle->desc.R-1-kj , handle->desc.S-1-ki, 0, 2*t1, 0, BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block) , scatter_index, compact, 2);
              }
            }
          }
        }
#else
/* won't happen as this code only runs on AVX512 platforms */
#endif
      }
    }
    weight_base = &LIBXSMM_VLA_ACCESS(7, tr_wt2, 0, 0, 0, 0, 0, 0, 0,
        BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
    libxsmm_barrier_wait(handle->barrier, ltid);
  }
  pool_index = 0;
  i = 0;

  if (n_segments) {
    /* We have segmented the stream of convolutions since we need to inject different functionalities...  */
    code_stream = handle->bwd_code_segments[ltid];
    if (handle->perform_relu_in_kernel == 1) {/* do RELU stuff in the kernel  */
      LIBXSMM_VLA_DECL(5, element_input_type, original_input, ((element_input_type*)handle->reg_input->data) + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in * handle->ifmblock), handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      element_input_type *regular_input_base;
      regular_input_base = &LIBXSMM_VLA_ACCESS(5, original_input, 0, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);

      if (handle->n_variants == 2) {
        for (pc = 0; pc < n_segments; pc++) {
          instr = code_stream[pc].segment_type;
          n_convs = code_stream[pc].n_convs;

          if (instr == IMG_LOOP_INIT) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
            if (handle->padding_flag == 1) {
#include "libxsmm_dnn_bwd_custom_custom_padding.tpl.c"
            }
          }

          if (instr == IMG_LOOP_CLOSE) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
            if ((handle->padding_flag == 1) && ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_RELU) > 0) ) {
#include "libxsmm_dnn_bwd_custom_custom_padding_back.tpl.c"
            }
          }

	  if (instr == OFM_LOOP_FIRST_TOUCH ) {
	    ofm1 = code_stream[pc].aux_index;
#include "libxsmm_dnn_bwd_custom_custom_apply_bn2.tpl.c"
	  }
          if ( instr == IFM_LOOP_INIT ) {
            ifm1 = code_stream[pc].aux_index;
            /* Apply bias if requested  */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
              /*#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"*/
            }
            /* Overwrite output with zeros if requested */
            if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_bwd == 0) ) {
              jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
            }
          }

          if ( instr == IFM_LOOP_CLOSE) {
            ifm1 = code_stream[pc].aux_index;
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
              ifm1 =  code_stream[pc].aux_index;
              element_input_type* cur_vec = &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 0, 0, 0,
                  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              for ( ij = 0; ij < handle->desc.H; ij++ ) {
                for ( ii = 0; ii < handle->desc.W*handle->ifmblock; ii+=16 ) {
#ifdef __AVX512F__
                  max_abs = _mm512_max_ps(max_abs, LIBXSMM_INTRINSICS_MM512_ABS_PS(_mm512_load_ps(cur_vec+ii)));
#else
                  /* won't happen as this code only runs on AVX512 platforms */
#endif
                }
                cur_vec += handle->ifwp*handle->ifmblock;
              }
            }
          }

          /* Run the stream of convolutions for this segment */
          for (conv_i = 0; conv_i < n_convs; conv_i++) {
            offset_i = stream[i];
            offset_w = stream[i+1];
            offset_o = stream[i+2];
            pi = stream[i+3];
            pw = stream[i+4];
            po = stream[i+5];
            kernel_pool[variant[pool_index]]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, regular_input_base + offset_o, &scale_factor, max_vals);
            pool_index++;
            i+=3;
          }

          if ( instr == IFM_LOOP_CLOSE) {
            ifm1 = code_stream[pc].aux_index;
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_RELU) > 0) {     
#include "libxsmm_dnn_bwd_custom_custom_apply_bn.tpl.c"
	    }
	  }
        }
      } else {
        for (pc = 0; pc < n_segments; pc++) {
          instr = code_stream[pc].segment_type;
          n_convs = code_stream[pc].n_convs;
          if (instr == IMG_LOOP_INIT) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
            if (handle->padding_flag == 1) {
#include "libxsmm_dnn_bwd_custom_custom_padding.tpl.c"
            }
          }

          if (instr == IMG_LOOP_CLOSE) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
            if ((handle->padding_flag == 1) && ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_RELU) > 0) ) {
#include "libxsmm_dnn_bwd_custom_custom_padding_back.tpl.c"
            }
          }

	  if (instr == OFM_LOOP_FIRST_TOUCH ) {
	    ofm1 = code_stream[pc].aux_index;
#include "libxsmm_dnn_bwd_custom_custom_apply_bn2.tpl.c"
	  }
          if ( instr == IFM_LOOP_INIT ) {
            ifm1 = code_stream[pc].aux_index;
            /* Apply bias if requested  */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
              /*#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"*/
            }
            /* Overwrite output with zeros if requested */
            if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_bwd == 0) ) {
              jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
            }
          }

          if ( instr == IFM_LOOP_CLOSE) {
            ifm1 = code_stream[pc].aux_index;
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
              ifm1 =  code_stream[pc].aux_index;
              element_input_type* cur_vec = &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 0, 0, 0,
                  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              for ( ij = 0; ij < handle->desc.H; ij++ ) {
                for ( ii = 0; ii < handle->desc.W*handle->ifmblock; ii+=16 ) {
#ifdef __AVX512F__
                  max_abs = _mm512_max_ps(max_abs, LIBXSMM_INTRINSICS_MM512_ABS_PS(_mm512_load_ps(cur_vec+ii)));
#else
                  /* won't happen as this code only runs on AVX512 platforms */
#endif
                }
                cur_vec += handle->ifwp*handle->ifmblock;
              }
            }
          }

          /* Run the stream of convolutions for this segment */
          for (conv_i = 0; conv_i < n_convs; conv_i++) {
            offset_i = stream[i];
            offset_w = stream[i+1];
            offset_o = stream[i+2];
            pi = stream[i+3];
            pw = stream[i+4];
            po = stream[i+5];
            kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, regular_input_base + offset_o, &scale_factor, max_vals);
            i+=3;
          }
          if ( instr == IFM_LOOP_CLOSE) {
            ifm1 = code_stream[pc].aux_index;
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_RELU) > 0) {     
#include "libxsmm_dnn_bwd_custom_custom_apply_bn.tpl.c"
	    }
	  }
        }
      }
    } else { /* We don't do RELU stuff in the kernel  */
      if (handle->n_variants == 2) {
        for (pc = 0; pc < n_segments; pc++) {
          instr = code_stream[pc].segment_type;
          n_convs = code_stream[pc].n_convs;

          if (instr == IMG_LOOP_INIT) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
            if (handle->padding_flag == 1) {
#include "libxsmm_dnn_bwd_custom_custom_padding.tpl.c"
            }
          }

          if (instr == IMG_LOOP_CLOSE) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
            if ((handle->padding_flag == 1) && ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_RELU) > 0) ) {
#include "libxsmm_dnn_bwd_custom_custom_padding_back.tpl.c"
            }
          }

	  if (instr == OFM_LOOP_FIRST_TOUCH ) {
	    ofm1 = code_stream[pc].aux_index;
#include "libxsmm_dnn_bwd_custom_custom_apply_bn2.tpl.c"
	  }
          if ( instr == IFM_LOOP_INIT ) {
            ifm1 = code_stream[pc].aux_index;
            /* Apply bias if requested  */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
              /*#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"*/
            }
            /* Overwrite output with zeros if requested */
            if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_bwd == 0) ) {
              jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
            }
          }

          if ( instr == IFM_LOOP_CLOSE ) {
            ifm1 = code_stream[pc].aux_index;
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) > 0) {
#ifdef __AVX512F__
              LIBXSMM_VLA_DECL(5, element_input_type, input, (element_input_type*) handle->reg_input->data,  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              LIBXSMM_VLA_DECL(5, element_input_type, del_input_2, (element_input_type*) handle->grad_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              element_input_type *orig_input_ptr;
              element_input_type *del_input_ptr;
              __m512 zero_reg  = _mm512_setzero_ps();
              __m512 orig_reg;
              __mmask16 mask;
              orig_input_ptr = &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              del_input_ptr = &LIBXSMM_VLA_ACCESS(5, del_input_2, img, ifm1, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              for (ij = 0; ij < handle->desc.H; ij++) {
                for (ii = 0; ii < handle->desc.W * 16; ii += 16) {
                  orig_reg  = _mm512_load_ps(orig_input_ptr + ii);
                  mask = _mm512_cmp_ps_mask(zero_reg, orig_reg, _CMP_EQ_OQ);
                  _mm512_mask_storeu_ps(del_input_ptr + ii, mask, zero_reg);
                }
                orig_input_ptr += handle->ifwp * 16;
                del_input_ptr += handle->ifwp *16;
              }
#else
              /* won't happen as this code only runs on AVX512 platforms */
#endif
            }

            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
              ifm1 =  code_stream[pc].aux_index;
              element_input_type* cur_vec = &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 0, 0, 0,
                  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
              for ( ij = 0; ij < handle->desc.H; ij++ ) {
                for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
#ifdef __AVX512F__
                  max_abs = _mm512_max_ps(max_abs, LIBXSMM_INTRINSICS_MM512_ABS_PS(_mm512_load_ps(cur_vec+ii)));
#else
                  /* won't happen as this code only runs on AVX512 platforms */
#endif
                }
                cur_vec += handle->ifwp*handle->ifmblock_hp;
              }
            }
          }

          /* Run the stream of convolutions for this segment */
          for (conv_i = 0; conv_i < n_convs; conv_i++) {
            offset_i = stream[i];
            offset_w = stream[i+1];
            offset_o = stream[i+2];
            pi = stream[i+3];
            pw = stream[i+4];
            po = stream[i+5];
            kernel_pool[variant[pool_index]]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
            pool_index++;
            i+=3;
          }
	if ( instr == IFM_LOOP_CLOSE) {
            ifm1 = code_stream[pc].aux_index;
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_RELU) > 0) {     
#include "libxsmm_dnn_bwd_custom_custom_apply_bn.tpl.c"
	    }
	  }
        }
      } else {
        for (pc = 0; pc < n_segments; pc++) {
          instr = code_stream[pc].segment_type;
          n_convs = code_stream[pc].n_convs;
          if (instr == IMG_LOOP_INIT) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
            if (handle->padding_flag == 1) {
#include "libxsmm_dnn_bwd_custom_custom_padding.tpl.c"
            }
          }

          if (instr == IMG_LOOP_CLOSE) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
            if ((handle->padding_flag == 1) && ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_RELU) > 0) ) {
#include "libxsmm_dnn_bwd_custom_custom_padding_back.tpl.c"
            }
          }

	  if (instr == OFM_LOOP_FIRST_TOUCH ) {
	    ofm1 = code_stream[pc].aux_index;
#include "libxsmm_dnn_bwd_custom_custom_apply_bn2.tpl.c"
	  }

          if ( instr == IFM_LOOP_INIT ) {
            /* Apply bias if requested  */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
              /*#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"*/
            }
            /* Overwrite output with zeros if requested */
            if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_bwd == 0) ) {
              jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
            }
          }

          if ( instr == IFM_LOOP_CLOSE ) {
            ifm1 = code_stream[pc].aux_index;
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) > 0) {
#ifdef __AVX512F__
              LIBXSMM_VLA_DECL(5, element_input_type, input, (element_input_type*) handle->reg_input->data,  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              LIBXSMM_VLA_DECL(5, element_input_type, del_input_2, (element_input_type*) handle->grad_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              element_input_type *orig_input_ptr;
              element_input_type *del_input_ptr;
              __m512 zero_reg  = _mm512_setzero_ps();
              __m512 orig_reg;
              __mmask16 mask;
              orig_input_ptr = &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              del_input_ptr = &LIBXSMM_VLA_ACCESS(5, del_input_2, img, ifm1, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              for (ij = 0; ij < handle->desc.H; ij++) {
                for (ii = 0; ii < handle->desc.W * 16; ii += 16) {
                  orig_reg  = _mm512_load_ps(orig_input_ptr + ii);
                  mask = _mm512_cmp_ps_mask(zero_reg, orig_reg, _CMP_EQ_OQ);
                  _mm512_mask_storeu_ps(del_input_ptr + ii, mask, zero_reg);
                }
                orig_input_ptr += handle->ifwp * 16;
                del_input_ptr += handle->ifwp *16;
              }
#else
              /* won't happen as this code only runs on AVX512 platforms */
#endif
            }

            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
              ifm1 =  code_stream[pc].aux_index;
              element_input_type* cur_vec = &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 0, 0, 0,
                  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
              for ( ij = 0; ij < handle->desc.H; ij++ ) {
                for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
#ifdef __AVX512F__
                  max_abs = _mm512_max_ps(max_abs, LIBXSMM_INTRINSICS_MM512_ABS_PS(_mm512_load_ps(cur_vec+ii)));
#else
                  /* won't happen as this code only runs on AVX512 platforms */
#endif
                }
                cur_vec += handle->ifwp*handle->ifmblock_hp;
              }
            }
          }

          /* Run the stream of convolutions for this segment */
          for (conv_i = 0; conv_i < n_convs; conv_i++) {
            offset_i = stream[i];
            offset_w = stream[i+1];
            offset_o = stream[i+2];
            pi = stream[i+3];
            pw = stream[i+4];
            po = stream[i+5];
            kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
            i+=3;
          }

	  if ( instr == IFM_LOOP_CLOSE) {
            ifm1 = code_stream[pc].aux_index;
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_RELU) > 0) {     
#include "libxsmm_dnn_bwd_custom_custom_apply_bn.tpl.c"
	    }
	  }
        }
      }
    }
  } else {
    /* Run the stream of convolutions, no extra operations are required... */
    if (handle->perform_relu_in_kernel == 1) {/* do RELU stuff in the kernel  */
      LIBXSMM_VLA_DECL(5, element_input_type, original_input, ((element_input_type*)handle->reg_input->data) + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in * handle->ifmblock), BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
      element_input_type *regular_input_base;
      regular_input_base = &LIBXSMM_VLA_ACCESS(5, original_input, 0, 0, 0, 0, 0, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);

      if (handle->n_variants == 2) {
        for (pc = 0; pc < instr; pc+=1) {
          offset_i = stream[i];
          offset_w = stream[i+1];
          offset_o = stream[i+2];
          pi = stream[i+3];
          pw = stream[i+4];
          po = stream[i+5];
          kernel_pool[variant[pc]]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, regular_input_base + offset_o, &scale_factor, max_vals);
          i+=3;
        }
      } else {
        for (pc = 0; pc < instr; pc++) {
          offset_i = stream[i];
          offset_w = stream[i+1];
          offset_o = stream[i+2];
          pi = stream[i+3];
          pw = stream[i+4];
          po = stream[i+5];
          kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, regular_input_base + offset_o, &scale_factor, max_vals);
          i+=3;
        }
      }
    } else {
      if (handle->n_variants == 2) {
        for (pc = 0; pc < instr; pc+=1) {
          offset_i = stream[i];
          offset_w = stream[i+1];
          offset_o = stream[i+2];
          pi = stream[i+3];
          pw = stream[i+4];
          po = stream[i+5];
          kernel_pool[variant[pc]]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
          i+=3;
        }
      } else {
        for (pc = 0; pc < instr; pc++) {
          offset_i = stream[i];
          offset_w = stream[i+1];
          offset_o = stream[i+2];
          pi = stream[i+3];
          pw = stream[i+4];
          po = stream[i+5];
          kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
          i+=3;
        }
      }
    }
  }

  if ( ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) && (handle->use_lp_kernel == 1) && (handle->compute_max_in_kernel_bwd == 0) ) {
#ifdef __AVX512F__
    _mm512_store_ps(max_vals, max_abs);
#else 
    /* won't happen as this code only runs on AVX512 platforms */
#endif
  }
  libxsmm_barrier_wait(handle->barrier, ltid);

#if 0
  /* Fuse ReLu here*/
  if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) > 0) {
    int ii, ij, ifm1, ifm2, img;
    img = ltid;
    LIBXSMM_VLA_DECL(5, element_input_type, input, (element_input_type*) handle->reg_input->data,  BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
    LIBXSMM_VLA_DECL(5, element_input_type, del_input_2, (element_input_type*) handle->grad_input->data, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
    element_input_type *orig_input_ptr;
    element_input_type *del_input_ptr;
    __m512 zero_reg  = _mm512_setzero_ps();
    __m512 orig_reg;
    __mmask16 mask;
    for (ifm1 = 0; ifm1 < BLOCKSIFM; ifm1++ ) {
      orig_input_ptr = &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
      del_input_ptr = &LIBXSMM_VLA_ACCESS(5, del_input_2, img, ifm1, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
      for (ij = 0; ij < handle->desc.H; ij++) {
        for (ii = 0; ii < handle->desc.W * 16; ii += 16) {
          orig_reg  = _mm512_load_ps(orig_input_ptr + ii);
          mask = _mm512_cmp_ps_mask(zero_reg, orig_reg, _CMP_EQ_OQ);
          _mm512_mask_storeu_ps(del_input_ptr + ii, mask, zero_reg);
        }
        orig_input_ptr += handle->ifwp * 16;
        del_input_ptr += handle->ifwp *16;
      }
    }
    libxsmm_barrier_wait(handle->barrier, ltid);
  }
#endif

}
