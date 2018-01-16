/******************************************************************************
 ** Copyright (c) 2016-2017, Intel Corporation                                **
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
/* Evangelos Georganas, John Pennycook, Jason Sewall (Intel Corp.)
******************************************************************************/
#define WEIGHT_INIT 0
#define UPDATE_KERNEL 1
#define WEIGHT_COPY 2

/* FIXME assignemnts here  */
int BLOCKSIFM = handle->blocksifm;
int BLOCKSOFM = handle->blocksofm;
int OFWP = handle->ofwp+handle->output_lp_padding;

/* computing first logical thread */
const int ltid = tid-start_thread;

/* Auxiliary integer variables   */
int img, ifm1, ifm2, imgifm1,ii, ij, i;
int j, k;
int ifmb;

int imgpt = (handle->desc.N + handle->desc.threads - 1)/handle->desc.threads;
int my_img_start = LIBXSMM_MIN( ltid * imgpt, handle->desc.N);
int my_img_end = LIBXSMM_MIN( (ltid+1) * imgpt, handle->desc.N);

/* traspose, copy and reduce work-related variables  */
const int reduce_work = BLOCKSOFM*BLOCKSIFM*handle->desc.R*handle->desc.S*handle->ofmblock;
const int reduce_chunksize = (reduce_work % handle->desc.threads == 0) ? (reduce_work / handle->desc.threads) : (reduce_work / handle->desc.threads) + 1;
const int reduce_thr_begin = (ltid * reduce_chunksize < reduce_work) ? (ltid * reduce_chunksize) : reduce_work;
const int reduce_thr_end = ((ltid + 1) * reduce_chunksize < reduce_work) ? ((ltid + 1) * reduce_chunksize) : reduce_work;
const int copywork = handle->desc.N*BLOCKSIFM;
const int copychunksize = (copywork % handle->desc.threads == 0) ? (copywork / handle->desc.threads) : (copywork / handle->desc.threads) + 1;
const int copy_thr_begin = (ltid * copychunksize < copywork) ? (ltid * copychunksize) : copywork;
const int copy_thr_end = ((ltid + 1) * copychunksize < copywork) ? ((ltid + 1) * copychunksize) : copywork;

/* Pointer related variables for output and weight */
element_output_type *out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock_lp * handle->fm_lp_block;
LIBXSMM_VLA_DECL(6, element_output_type, tr_output,  (element_output_type*)handle->scratch6 , BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);
LIBXSMM_VLA_DECL(6, element_output_type, output, out, handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);
element_filter_type* weight_ptr = (element_filter_type*)handle->grad_filter->data;
element_filter_type* per_thread_weight_ptr = ((element_filter_type*)handle->scratch4) + (ltid*LIBXSMM_MIN(handle->block_upd_ofm,BLOCKSOFM)*LIBXSMM_MIN(handle->block_upd_ifm,BLOCKSIFM)*handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock*handle->fm_lp_block);
LIBXSMM_VLA_DECL(2, element_filter_type, per_thread_weight, per_thread_weight_ptr, handle->ofmblock);
element_filter_type* reduction_weight_ptr = ((element_filter_type*)handle->scratch4) + (handle->desc.threads*LIBXSMM_MIN(handle->block_upd_ofm,BLOCKSOFM)*LIBXSMM_MIN(handle->block_upd_ifm,BLOCKSIFM)*handle->desc.R*handle->desc.S*handle->ifmblock*handle->fm_lp_block*handle->ofmblock);
LIBXSMM_VLA_DECL(3, element_filter_type, reduction_weight, reduction_weight_ptr, handle->desc.threads, handle->ofmblock);

/* Pointer related variables for input */
element_input_type (* LIBXSMM_RESTRICT input_ptr);
element_input_type (* LIBXSMM_RESTRICT copy_ptr);
element_input_type *prefetch_ptr;
int padded_h = (handle->padding_flag == 1) ? handle->ifhp + 2 * handle->desc.pad_h : handle->ifhp;
int padded_w = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
int ifwp_extended = padded_w + handle->qfma_input_pad;
int dst_ifhp;

if (handle->resize_input == 1) {
  ifwp_extended = handle->ifwp_resized + handle->qfma_input_pad;
  dst_ifhp = handle->ifhp_resized;
} else {
  dst_ifhp = handle->ifhp;
}

LIBXSMM_VLA_DECL(6, element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
LIBXSMM_VLA_DECL(5, element_input_type, tr_input_padded, (element_input_type*)handle->scratch5, BLOCKSIFM, padded_h, handle->ifmblock_hp, ifwp_extended);
LIBXSMM_VLA_DECL(5, element_input_type, input_padded, (element_input_type*)handle->scratch5, BLOCKSIFM, padded_h, padded_w, handle->ifmblock_hp);
LIBXSMM_VLA_DECL(5, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, BLOCKSIFM, dst_ifhp, handle->ifmblock_hp, ifwp_extended);

/* Stream related variables  */
segment_t *code_stream;
int *stream = handle->compute_upd_indices_ptrs[ltid];
int instr, n_segments, n_convs, conv_i, offset_i, offset_t, offset_o, offset_w, offset_s, pi, po, pw, pc;

/* Base pointers  */
element_input_type *input_base;
element_input_type *input_zero;
element_filter_type *weight_base;
element_output_type *output_base;

/* Kernel related variables  */
libxsmm_xmatcopyfunction jitted_matcopy = handle->matcopy_upd[0].xmatcopy;
libxsmm_xmatcopyfunction jitted_matzero = handle->matcopy_upd[1].xmatcopy;
libxsmm_xmatcopyfunction jitted_matzero_weights = handle->matcopy_upd[2].xmatcopy;
libxsmm_convfunction kernel = ( handle->trans_ofw_ifm == 0 || handle->ifmblock == 1 ) ? (libxsmm_convfunction)handle->code_upd[1].xconv.sconv : (libxsmm_convfunction)handle->code_upd[4].xconv.sconv;

transposer tp_func;
if ( handle->trans_ofw_ifm > 0 ) {
  if (handle->padding_flag == 1) {
    tp_func = get_transposer(handle->ifmblock, handle->ifwp, ifwp_extended, handle->ifmblock);
  }
  else
    tp_func = get_transposer(handle->ifmblock, handle->ifwp, ifwp_extended, handle->ifmblock);
}

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

/* Initialize base pointers */
if (handle->padding_flag == 1) {
  input_base = &LIBXSMM_VLA_ACCESS(5, tr_input_padded, 0, 0, 0, 0, 0, BLOCKSIFM, padded_h, handle->ifmblock_hp, ifwp_extended);
  input_zero = &LIBXSMM_VLA_ACCESS(5, tr_input_padded, ltid, 0, 0, 0, 0, BLOCKSIFM, padded_h, handle->ifmblock_hp, ifwp_extended);
  memset( input_zero, 0, BLOCKSIFM * padded_h * ifwp_extended * handle->ifmblock_hp * sizeof(element_input_type) );
} else {
  input_base = &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, 0, 0, 0, 0, 0, BLOCKSIFM, dst_ifhp, handle->ifmblock_hp, ifwp_extended);
}

#if 0
if (handle->reduce_weights == 0) {
  int team_div = (int) sqrt(handle->desc.threads);
  while ( handle->desc.threads % team_div != 0  ) {
    team_div--;
  }  
  int n_ifm_teams = ( BLOCKSIFM > BLOCKSOFM ) ? handle->desc.threads/team_div : team_div ;
  int n_ofm_teams = ( BLOCKSIFM > BLOCKSOFM ) ? team_div : handle->desc.threads/team_div ;
  int ifms_per_thread = (BLOCKSIFM+n_ifm_teams-1)/n_ifm_teams;
  int ofms_per_thread = (BLOCKSOFM+n_ofm_teams-1)/n_ofm_teams;
  int my_ifm_id = ltid/n_ofm_teams;
  int my_ofm_id = ltid%n_ofm_teams;
  int my_ifm_start =  LIBXSMM_MIN(my_ifm_id * ifms_per_thread, BLOCKSIFM);
  int my_ifm_end =  LIBXSMM_MIN((my_ifm_id+1) * ifms_per_thread, BLOCKSIFM);
  int my_ofm_start =  LIBXSMM_MIN(my_ofm_id * ofms_per_thread, BLOCKSOFM);
  int my_ofm_end =  LIBXSMM_MIN((my_ofm_id+1) * ofms_per_thread, BLOCKSOFM);

  element_filter_type *zero_ptr;
  int ifm1, ofm1, kj, ki;
  for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++ ) {
    for ( ofm1 = my_ofm_start; ofm1 < my_ofm_end; ofm1++ ) {
      for (kj=0; kj < handle->desc.R; kj++) {
        for (ki=0; ki < handle->desc.S; ki++) {
          zero_ptr = &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, kj, ki, 0, 0, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock_hp, handle->ofmblock);
          memset(zero_ptr, 0, handle->ifmblock_hp*handle->ofmblock*sizeof(element_filter_type));
        }
      }
    }
  }
}
#endif

/* LP transformations */
#if 1
if (handle->padding_flag == 1) {
  int img = ltid, ifm1, ij, ifm2, ii;
  int ofm1, ofm2, k, lp;
  int FM;
  int W;  
  for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ++ifm1) {
    for (ij = 0; ij < handle->ifhp; ++ij) {
      for (ii = 0; ii < handle->ifwp; ++ii) {
        for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
          for (lp = 0; lp < handle->fm_lp_block; ++lp) {
            LIBXSMM_VLA_ACCESS(5, tr_input_padded, img, ifm1, ij+handle->desc.pad_h, ifm2*handle->fm_lp_block+lp, ii+handle->desc.pad_w, BLOCKSIFM, padded_h, handle->ifmblock_hp, ifwp_extended) =
              LIBXSMM_VLA_ACCESS(6, input_nopad, img, ifm1, ij, ii, ifm2, lp, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
          }
        }
      }
    }
  }  
#include "output_lp_transposer.tpl.c"
} else {
  if (handle->resize_input == 0) {
    lp_transpose_input_and_output(ltid, handle);
  } else {
    lp_transpose_and_resize_input_and_output(ltid, handle);
  }
}
#endif

libxsmm_barrier_wait(handle->barrier, ltid);

if (handle->ofh == 28 || handle->ofh == 56 || handle->ofh == 14)
{
  weight_base = &LIBXSMM_VLA_ACCESS(2, per_thread_weight, 0, 0, handle->ofmblock); /* use thread-private scratchpad to accumulate weights */
} else {
  weight_base = &LIBXSMM_VLA_ACCESS(3, reduction_weight, 0, ltid, 0, handle->desc.threads, handle->ofmblock); /* weights are accumulated in registers and can be written straight to memory */
}

/*if (handle->reduce_weights) {
  weight_base = &LIBXSMM_VLA_ACCESS(3, reduction_weight, 0, ltid/(handle->desc.threads/handle->weight_copies), 0, handle->weight_copies, handle->ofmblock); 
} else {
  weight_base = weight_ptr;
}*/

if ( !handle->reduce_weights  ) {
  weight_base = weight_ptr;
}

output_base = &LIBXSMM_VLA_ACCESS(6, tr_output, 0, 0, 0, 0, 0, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);


LIBXSMM_VLA_DECL(6, element_input_type, lp_input, (element_input_type*)handle->reg_input->data, BLOCKSIFM, handle->ifhp, handle->ifwp/2, handle->ifmblock_hp, 2);
LIBXSMM_VLA_DECL(6, element_output_type, lp_output, (element_output_type*)handle->grad_output->data, BLOCKSOFM, handle->ofhp, handle->ofwp/2, handle->ofmblock, 2);

if (handle->trans_ofw_ifm == 1) {
  input_base = &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, 0, 0, 0, 0, 0, BLOCKSIFM, dst_ifhp, handle->ifmblock_hp, ifwp_extended);
} else {
  if (handle->avoid_input_trans == 1) {
    LIBXSMM_VLA_DECL(6, element_input_type, lp_input, (element_input_type*)handle->reg_input->data, BLOCKSIFM, handle->ifhp, handle->ifwp/2, handle->ifmblock_hp, 2);
    input_base = &LIBXSMM_VLA_ACCESS(6, lp_input, 0, 0, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp/2, handle->ifmblock_hp, 2);
  } else {
    LIBXSMM_VLA_DECL(6, element_input_type, lp_input, (element_input_type*)handle->scratch3, handle->blocksifm, handle->ifhp, handle->ifwp/2, handle->ifmblock_hp, 2);
    input_base = &LIBXSMM_VLA_ACCESS(6, lp_input, 0, 0, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp/2, handle->ifmblock_hp, 2);
  }
}

#if 0
if (handle->use_vperm_transposes == 1) {
  element_output_type *out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp * handle->ofmblock_lp * handle->fm_lp_block /*+ handle->desc.pad_w_out*/);
  LIBXSMM_VLA_DECL(6, element_output_type, lp_output, out, BLOCKSOFM, handle->ofhp, handle->ofwp/2, handle->ofmblock, 2);
  output_base = &LIBXSMM_VLA_ACCESS(6, lp_output, 0, 0, 0, 0, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp/2, handle->ofmblock, 2);
} else {
  LIBXSMM_VLA_DECL(6, element_output_type, tr_output,  (element_output_type*)handle->scratch6 , BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);
  output_base = &LIBXSMM_VLA_ACCESS(6, tr_output, 0, 0, 0, 0, 0, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);
}
#endif

if (handle->avoid_output_trans) {
    element_output_type *out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock_lp * handle->fm_lp_block;
    LIBXSMM_VLA_DECL(6, element_output_type, lp_output, out, BLOCKSOFM, handle->ofhp, handle->ofwp/2, handle->ofmblock, 2);
    output_base = &LIBXSMM_VLA_ACCESS(6, lp_output, 0, 0, 0, 0, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp/2, handle->ofmblock, 2);
} else {
    LIBXSMM_VLA_DECL(6, element_output_type, tr_output,  (element_output_type*)handle->scratch6 , BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);
    output_base = &LIBXSMM_VLA_ACCESS(6, tr_output, 0, 0, 0, 0, 0, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);
}

i = 0;
instr = handle->n_entries_upd[ltid];
n_segments = handle->n_upd_code_segments[ltid];

float scale_factor __attribute__((aligned(64)));
if (handle->use_lp_kernel == 1) {
  scale_factor = (float) pow(2.0, -1.0*((double)(handle->reg_input->scf + handle->grad_output->scf)));
}

float vnni_scratch[32] __attribute__((aligned(64)));

float *max_vals __attribute__((aligned(64)));
__m512 max_abs = _mm512_setzero_ps();
if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
  LIBXSMM_VLA_DECL(2, float, maxstats, handle->maxstats_upd->data, 16);
  max_vals = (float*) &LIBXSMM_VLA_ACCESS(2, maxstats, ltid, 0, 16);
}

if (n_segments) {
  /* We have segmented the stream of convolutions since we need to inject different functionalities... */
  code_stream = handle->upd_code_segments[ltid];
  for (pc = 0; pc < n_segments; pc++) {
    instr = code_stream[pc].segment_type;
    n_convs = code_stream[pc].n_convs;

    if (instr == WEIGHT_INIT) {
      offset_w = code_stream[pc].aux_index;
      for ( j = offset_w; j < offset_w + handle->desc.R*handle->desc.S*handle->ofmblock*handle->ifmblock_hp; j += 16) {
        LIBXSMM_PRAGMA_VALIGNED
          LIBXSMM_PRAGMA_SIMD
          for ( k = 0; k < 16; ++k ) {
            weight_base[j + k] = (element_filter_type) 0;
          }
      }
    }

    if (instr == WEIGHT_COPY) {
      offset_w /= handle->desc.R * handle->desc.S * handle->ifmblock_hp * handle->ofmblock;
      offset_w *= handle->desc.R * handle->desc.S * handle->ifmblock_hp;
      offset_s = code_stream[pc].aux_index;
      for ( j = 0; j < handle->desc.R*handle->desc.S*handle->ifmblock_hp; j++ ) {
        LIBXSMM_PRAGMA_NONTEMPORAL
          LIBXSMM_PRAGMA_VALIGNED
          LIBXSMM_PRAGMA_SIMD
          for ( k = 0; k < 16; k++ ) {
            LIBXSMM_VLA_ACCESS(3, reduction_weight, offset_s + j, ltid, k, handle->desc.threads, 16) = LIBXSMM_VLA_ACCESS(2, per_thread_weight, offset_w + j, k, 16);
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
      kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, vnni_scratch);
      i+=3;
    }
  }
} else {
  /* Run the stream of convolutions, no extra operations are required...  */
  for (pc = 0; pc < instr; pc++)
  {
    offset_i = stream[i];
    offset_w = stream[i+1];
    offset_o = stream[i+2];
    pi = stream[i+3];
    pw = stream[i+4];
    po = stream[i+5];
    kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, vnni_scratch);
    i+=3;
  }
}

if (handle->reduce_weights) {
  if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
#define __AVX512F__
    /* Perform reduction because we used thread private filters... */
    if (handle->upd_use_external_reduce == 0) {
      libxsmm_barrier_wait(handle->barrier, ltid);
      for ( j = reduce_thr_begin; j < reduce_thr_end; j++ ) {
#ifdef __AVX512F__
        __m512 weight_sum = _mm512_setzero_ps();
        for ( i = 0; i < handle->desc.threads; i++ ) {
          weight_sum = _mm512_add_ps(weight_sum, _mm512_load_ps(&LIBXSMM_VLA_ACCESS(3, reduction_weight, j, i, 0, handle->desc.threads, 16)));
        }
        if ( ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
          _mm512_stream_ps(&weight_ptr[j*16], weight_sum);
          max_abs = _mm512_max_ps(max_abs, _mm512_abs_ps(weight_sum));
        } else {
          __m512 new_result = _mm512_add_ps(weight_sum, _mm512_load_ps(&weight_ptr[j*16]));
          _mm512_store_ps(&weight_ptr[j*16], new_result);
          max_abs = _mm512_max_ps(max_abs, _mm512_abs_ps(new_result));
        }
#else
        element_filter_type weight_sum[16] LIBXSMM_ATTRIBUTE(aligned(64));
        LIBXSMM_PRAGMA_VALIGNED
          LIBXSMM_PRAGMA_SIMD
          for ( k = 0; k < 16; k++ ) {
            weight_sum[k] = (element_filter_type) 0;
          }
        for ( i = 0; i < handle->desc.threads; i++ ) {
          LIBXSMM_PRAGMA_VALIGNED
            LIBXSMM_PRAGMA_SIMD
            for ( k = 0; k < 16; k++ ) {
              weight_sum[k] += LIBXSMM_VLA_ACCESS(3, reduction_weight, j, i, k, handle->desc.threads, 16);
            }
        }
        if ( ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
          LIBXSMM_PRAGMA_NONTEMPORAL
            LIBXSMM_PRAGMA_VALIGNED
            LIBXSMM_PRAGMA_SIMD
            for ( k = 0; k < 16; k++ ) {
              weight_ptr[j*16 + k] = weight_sum[k];
            }
        } else {
          LIBXSMM_PRAGMA_VALIGNED
            LIBXSMM_PRAGMA_SIMD
            for ( k = 0; k < 16; k++ ) {
              weight_ptr[j*16 + k] += weight_sum[k];
            }
        }
#endif
      }
#ifdef __AVX512F__
      _mm512_store_ps(max_vals, max_abs);
#endif
    }
    libxsmm_barrier_wait(handle->barrier, ltid);
#undef __AVX512F__
  } else {
#define __AVX512F__
    /* Perform reduction because we used thread private filters... */
    if (handle->upd_use_external_reduce == 0) {
      libxsmm_barrier_wait(handle->barrier, ltid);
      for ( j = reduce_thr_begin; j < reduce_thr_end; j++ ) {
#ifdef __AVX512F__
        __m512 weight_sum = _mm512_setzero_ps();
        for ( i = 0; i < handle->desc.threads; i++ ) {
          weight_sum = _mm512_add_ps(weight_sum, _mm512_load_ps(&LIBXSMM_VLA_ACCESS(3, reduction_weight, j, i, 0, handle->desc.threads, 16)));
        }
        if ( ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
          _mm512_stream_ps(&weight_ptr[j*16], weight_sum);
        } else {
          __m512 new_result = _mm512_add_ps(weight_sum, _mm512_load_ps(&weight_ptr[j*16]));
          _mm512_store_ps(&weight_ptr[j*16], new_result);
        }
#else
        element_filter_type weight_sum[16] LIBXSMM_ATTRIBUTE(aligned(64));
        LIBXSMM_PRAGMA_VALIGNED
          LIBXSMM_PRAGMA_SIMD
          for ( k = 0; k < 16; k++ ) {
            weight_sum[k] = (element_filter_type) 0;
          }
        for ( i = 0; i < handle->desc.threads; i++ ) {
          LIBXSMM_PRAGMA_VALIGNED
            LIBXSMM_PRAGMA_SIMD
            for ( k = 0; k < 16; k++ ) {
              weight_sum[k] += LIBXSMM_VLA_ACCESS(3, reduction_weight, j, i, k, handle->desc.threads, 16);
            }
        }
        if ( ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
          LIBXSMM_PRAGMA_NONTEMPORAL
            LIBXSMM_PRAGMA_VALIGNED
            LIBXSMM_PRAGMA_SIMD
            for ( k = 0; k < 16; k++ ) {
              weight_ptr[j*16 + k] = weight_sum[k];
            }
        } else {
          LIBXSMM_PRAGMA_VALIGNED
            LIBXSMM_PRAGMA_SIMD
            for ( k = 0; k < 16; k++ ) {
              weight_ptr[j*16 + k] += weight_sum[k];
            }
        }
#endif
      }
    }
    libxsmm_barrier_wait(handle->barrier, ltid);
#undef __AVX512F__
  }
}
libxsmm_barrier_wait(handle->barrier, ltid);
