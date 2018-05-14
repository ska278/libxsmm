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

#define FUSED_BN_CONV_WRAPPER

#ifdef FUSED_BN_CONV_WRAPPER
void do_BN(int _my_h, int _my_w, element_input_type * input1_st, element_input_type * input1, int ifm_idx, int my_ldh, int my_ldw, element_input_type * myexpect, element_input_type * mystddev, element_input_type * mygamma, element_input_type * mybeta, int ltid, libxsmm_dnn_layer* handle, int my_w,  int my_h)
{
  int my_c;
  for(my_c = 0 ; my_c < handle->ifmblock ; my_c++)
  {
    input1_st[ifm_idx * handle->ifhp * handle->ifwp * handle->ifmblock + my_c + (my_w + handle->desc.pad_w_in) * handle->ifmblock + (my_h + handle->desc.pad_h_in) * handle->ifmblock * handle->ifwp] = input1[ifm_idx * my_ldh * my_ldw * handle->ifmblock + my_c + _my_w * handle->ifmblock + _my_h * handle->ifmblock * my_ldw];
    element_input_type after = (input1[ifm_idx * my_ldh * my_ldw * handle->ifmblock + my_c + _my_w * handle->ifmblock + _my_h * handle->ifmblock * my_ldw] - myexpect[my_c]) * mystddev[my_c] * mygamma[my_c] + mybeta[my_c];
    input1[ifm_idx * my_ldh * my_ldw * handle->ifmblock + my_c + _my_w * handle->ifmblock + _my_h * handle->ifmblock * my_ldw] = (after > 0) ? after : 0.;
  }
}

void wrapper_kernel(libxsmm_convfunction k, element_input_type * input1, const element_filter_type * weight1, element_output_type * output1, element_input_type * input2, const element_filter_type * weight2, element_output_type* output2, float * sf, float * mv, libxsmm_dnn_layer* handle, int ifm1, int padded_w, int padded_h, int img, int BLOCKSIFM, int ltid, int offset_i, int pi, element_input_type * input1_st, int oi, int oj)
{
  if(ltid==0) printf("offset_i, R, S, oj, oi, ofh_rb, ofw_rb, %d %d %d %d %d %d %d\n", offset_i, handle->desc.R, handle->desc.S, 
                          oj, oi, handle->fwd_ofh_rb, handle->fwd_ofw_rb);

  LIBXSMM_VLA_DECL(2, element_input_type, expect, (element_input_type*)handle->reg_expect->data, handle->ifmblock);
  LIBXSMM_VLA_DECL(2, element_input_type, stddev, (element_input_type*)handle->reg_stddev->data, handle->ifmblock);
  LIBXSMM_VLA_DECL(2, element_input_type, gamma, (element_input_type*)handle->reg_gamma->data, handle->ifmblock);
  LIBXSMM_VLA_DECL(2, element_input_type, beta, (element_input_type*)handle->reg_beta->data, handle->ifmblock);
  int my_h, my_w, my_c, ifm_idx, my_ldw, my_pad_h, my_pad_w, my_ldh;
  if (handle->padding_flag == 1) {
    my_ldw = padded_w;
    my_ldh = padded_h;
    my_pad_h = handle->desc.pad_h;
    my_pad_w = handle->desc.pad_w;
  } else {
    my_ldw = handle->ifwp;
    my_ldh = handle->ifhp;
    my_pad_h = handle->desc.pad_h_in;
    my_pad_w = handle->desc.pad_w_in;
  }

  if(handle->desc.R == 1 && handle->desc.S == 1)
  {
    for(ifm_idx = 0 ; ifm_idx < handle->blocksifm_blocking ; ifm_idx++ ) 
    {
      element_input_type * myexpect = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, expect, ifm_idx+ifm1, 0, handle->ifmblock));
      element_input_type * mystddev = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, stddev, ifm_idx+ifm1, 0, handle->ifmblock));
      element_input_type * mygamma = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, gamma, ifm_idx+ifm1, 0, handle->ifmblock));
      element_input_type * mybeta = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, beta, ifm_idx+ifm1, 0, handle->ifmblock));
      for(my_h = 0 ; my_h < handle->fwd_ofh_rb * handle->desc.u; my_h++)
      {
        for(my_w = 0 ; my_w < handle->fwd_ofw_rb * handle->desc.v ; my_w++)
        {
          int _my_h = my_h + my_pad_h;
          int _my_w = my_w + my_pad_w;
          int _my_h_st = my_h + handle->desc.pad_h_in;
          int _my_w_st = my_w + handle->desc.pad_w_in;
          do_BN(_my_h, _my_w, input1_st, input1, ifm_idx, my_ldh, my_ldw, myexpect, mystddev, mygamma, mybeta, ltid, handle, my_w, my_h);
        }
      }
    }
    k( input1, weight1, output1, input2, weight2, output2, sf, mv);
  }
  else if(handle->desc.R == 3 && handle->desc.S == 3)
  {
    for(ifm_idx = 0 ; ifm_idx < handle->blocksifm_blocking ; ifm_idx++ ) 
    {
      element_input_type * myexpect = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, expect, ifm_idx+ifm1, 0, handle->ifmblock));
      element_input_type * mystddev = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, stddev, ifm_idx+ifm1, 0, handle->ifmblock));
      element_input_type * mygamma = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, gamma, ifm_idx+ifm1, 0, handle->ifmblock));
      element_input_type * mybeta = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, beta, ifm_idx+ifm1, 0, handle->ifmblock));

      // Do BN of left corner first
      if((oj == 0) && (oi == 0))
      {
        my_w = 0;
        my_h = 0;
        if((oi + my_w < handle->desc.W) && (oj + my_h < handle->desc.H))
        {
          do_BN(my_h+1, my_w+1, input1_st, input1, ifm_idx, my_ldh, my_ldw, myexpect, mystddev, mygamma, mybeta, ltid, handle, my_w, my_h);
        }
      }

      // Do BN of row 0 for this segment, plus extra column to the right if possible
      if(oj == 0)
      {
        my_h = 0;
        for(my_w = 1 ; my_w < (handle->fwd_ofw_rb + 1) * handle->desc.v ; my_w++)
	{
	  if(oi + my_w < handle->desc.W)
	  {
            do_BN(my_h+1, my_w+1, input1_st, input1, ifm_idx, my_ldh, my_ldw, myexpect, mystddev, mygamma, mybeta, ltid, handle, my_w, my_h);
	  }
	}
      }

      // Do BN of left corner in row below current
      if(oi == 0)
      {
        my_w = 0;
	for(my_h = 1 ; my_h < (handle->fwd_ofh_rb + 1) * handle->desc.u ; my_h++)
	{
          if((oi + my_w < handle->desc.W) && (oj + my_h < handle->desc.H))
          {
            do_BN(my_h+1, my_w+1, input1_st, input1, ifm_idx, my_ldh, my_ldw, myexpect, mystddev, mygamma, mybeta, ltid, handle, my_w, my_h);
          }
	}
      }

      // Do BN of right corner in row below current
      my_w = (handle->fwd_ofw_rb) * handle->desc.v;
      for(my_h = 1 ; my_h < (handle->fwd_ofh_rb + 1) * handle->desc.v ; my_h++)
      {
        if((oi + my_w < handle->desc.W) && (oj + my_h < handle->desc.H))
        {
          do_BN(my_h+1, my_w+1, input1_st, input1, ifm_idx, my_ldh, my_ldw, myexpect, mystddev, mygamma, mybeta, ltid, handle, my_w, my_h);
        }
      }
    }


    for(ifm_idx = 0 ; ifm_idx < handle->blocksifm_blocking ; ifm_idx++ ) 
    {
      element_input_type * myexpect = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, expect, ifm_idx+ifm1, 0, handle->ifmblock));
      element_input_type * mystddev = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, stddev, ifm_idx+ifm1, 0, handle->ifmblock));
      element_input_type * mygamma = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, gamma, ifm_idx+ifm1, 0, handle->ifmblock));
      element_input_type * mybeta = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, beta, ifm_idx+ifm1, 0, handle->ifmblock));

      // Do BN of inside part 
      for(my_w = 1 ; my_w < (handle->fwd_ofw_rb) * handle->desc.v ; my_w++)
      {
        for(my_h = 1 ; my_h < (handle->fwd_ofh_rb + 1) * handle->desc.v ; my_h++)
	{
          if((oi + my_w < handle->desc.W) && (oj + my_h < handle->desc.H))
  	  {
            do_BN(my_h+1, my_w+1, input1_st, input1, ifm_idx, my_ldh, my_ldw, myexpect, mystddev, mygamma, mybeta, ltid, handle, my_w, my_h);
  	  }
	}
      }
    }
    k( input1, weight1, output1, input2, weight2, output2, sf, mv);
  }
}

#endif

#define IMG_LOOP_INIT 0
#define OFM_LOOP_INIT 1
#define OFM_LOOP_CLOSE 2
#define CONVOLUTION_KERNEL 3
#define IFM_LOOP_CLOSE_S 4
#define IFM_LOOP_FIRST_TOUCH 5
#define IMG_LOOP_CLOSE 6

#define LOCAL_ENTRIES_PER_CONV 7

int BLOCKSIFM = handle->blocksifm_lp;
int BLOCKSOFM = handle->blocksofm;

const int ltid = tid-start_thread;
int gs = handle->desc.threads; /*atoi(getenv("GSIZE"));*/
const int tile_id = ltid/gs;
/* Pointer variables  */
element_input_type *input_base, *input_ptr, *input_st_base;
const element_filter_type *weight_base;
element_input_type *input_zero;
element_output_type *output_base;
element_input_type *copy_ptr, *prefetch_ptr;
element_output_type *out = ((element_output_type*)handle->reg_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * (handle->ofmblock);
LIBXSMM_VLA_DECL(5, element_output_type, output, out, BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(6, element_input_type, input, (element_input_type*)handle->reg_input->data, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
/* LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);*/
LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data + tile_id * BLOCKSIFM * BLOCKSOFM * handle->ifmblock * handle->ofmblock * handle->fm_lp_block *  handle->desc.R * handle->desc.S, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);

/* Auxiliary integer variables   */
int instr, n_segments, offset_bn, offset_i, offset_o, offset_w, pi, po, pw, pc, i, ih, n_convs, conv_i, ifm1, ofm1, ofm2, oj, img, input_h_start, input_h_end, my_h_out, oi, offset_i_st;
/* Stream related variables  */
segment_t *code_stream;
int *stream = handle->compute_fwd_indices_ptrs[ltid];
int *bn_stream = handle->bn_indices_ptrs[ltid];

/* Padding related variables */
const int padded_h = handle->ifhp + 2 * handle->desc.pad_h;
const int padded_w = handle->ifwp + 2 * handle->desc.pad_w;
LIBXSMM_VLA_DECL(5, element_input_type, input_buffer, ((element_input_type*)handle->scratch5) + ltid * BLOCKSIFM * padded_h * padded_w * handle->ifmblock * handle->fm_lp_block, padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);

LIBXSMM_VLA_DECL(6, element_input_type, input_st_buffer, (element_input_type*)handle->reg_input_st->data, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
/* Kernel related variables  */
libxsmm_xmcopyfunction jitted_matcopy = handle->matcopy_fwd[0].xmatcopy;
libxsmm_xmcopyfunction jitted_zero_overwrite = handle->matcopy_fwd[1].xmatcopy;
libxsmm_convfunction kernel = (libxsmm_convfunction)handle->code_fwd[0].xconv.sconv;
libxsmm_convfunction kernel2 = (libxsmm_convfunction)handle->code_fwd[1].xconv.sconv;
libxsmm_convfunction kernel_pool[4];
kernel_pool[0] = kernel;
kernel_pool[1] = kernel2;
kernel_pool[2] = kernel;
kernel_pool[3] = kernel2;
LIBXSMM_ALIGNED(float scale_factor, 64);
LIBXSMM_ALIGNED(float *max_vals, 64);
char *variant = handle->kernel_fwd_variant_ptrs[ltid];
int pool_index = 0;
/* Stream for BN offsets */
int bn_i = 0;
#ifndef FP64_BN_STATS
element_output_type *bn_sum_base;
element_output_type *bn_sum_base2;
#else
double *bn_sum_base;
double *bn_sum_base2;
#endif
#ifdef __AVX512F__
__m512 max_abs;
#else
/* won't happen as this code only runs on AVX512 platforms */
#endif

kernel_pool[0] = kernel;
kernel_pool[1] = kernel2;

/* Initialize base pointers */
if (handle->padding_flag == 1) {
  input_base = &LIBXSMM_VLA_ACCESS(5, input_buffer, 0, 0, 0, 0, 0,
      padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
  input_zero = &LIBXSMM_VLA_ACCESS(5, input_buffer, 0, 0, 0, 0, 0,
      padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
  input_st_base = &LIBXSMM_VLA_ACCESS(6, input_st_buffer, 0, 0, 0, 0, 0, 0,
      BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
  /* we need to set the scratch to zero */
  /* @TODO: we need to find a better/faster code here */
  memset( input_zero, 0, BLOCKSIFM * padded_h * padded_w * handle->ifmblock * handle->fm_lp_block * sizeof(element_input_type) );
} else {
  input_base = &LIBXSMM_VLA_ACCESS(6, input, 0, 0, 0, 0, 0, 0,
      BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
  input_st_base = &LIBXSMM_VLA_ACCESS(6, input_st_buffer, 0, 0, 0, 0, 0, 0,
      BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
}
weight_base = &LIBXSMM_VLA_ACCESS(7, weight, 0, 0, 0, 0, 0, 0, 0,
    BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
output_base = &LIBXSMM_VLA_ACCESS(5, output, 0, 0, 0, 0, 0,
    BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);

instr = handle->n_entries_fwd[ltid];
n_segments = handle->n_fwd_code_segments[ltid];

if (handle->use_lp_kernel == 1) {
  scale_factor = libxsmm_sexp2(-1.f*((float)(handle->reg_filter->scf + handle->reg_input->scf)));
}

if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
  LIBXSMM_VLA_DECL(2, element_output_type, maxstats, (element_output_type*)handle->maxstats_fwd->data, handle->ofmblock);
  max_vals = (float*) &LIBXSMM_VLA_ACCESS(2, maxstats, ltid, 0, handle->ofmblock);
#ifdef __AVX512F__
  max_abs = _mm512_setzero_ps();
  _mm512_store_ps(max_vals, max_abs);
#else
/* won't happen as this code only runs on AVX512 platforms */
#endif
}

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

i = 0;
if (n_segments) {
  /* We have segmented the stream of convolutions since we need to inject different functionalities...  */
  code_stream = handle->fwd_code_segments[ltid];
  /* If we are in the img_par execution then avoid fine-grained copy in case of padding...  */
  /* TODO: Second condition guarantees we run the img_par code when we have MB=1 -- and hopefully HUGE images */
  if (handle->desc.N*BLOCKSOFM  >= handle->desc.threads && !((handle->desc.N == 1) && (handle->fwd_ofh_rb == 1))) {
    if (handle->compute_batch_stats_in_kernel == 1) { /* We  do BN stuff in the kernel  */
#ifndef FP64_BN_STATS
      LIBXSMM_VLA_DECL(4, element_output_type, kernel_stats, (element_output_type*)handle->batch_stats->data, BLOCKSOFM, handle->desc.N, handle->ofmblock);
#else
      LIBXSMM_VLA_DECL(4, double, kernel_stats, (double*)handle->batch_stats->data, BLOCKSOFM, handle->desc.N, handle->ofmblock);
#endif
      bn_sum_base =  &LIBXSMM_VLA_ACCESS(4, kernel_stats, 0, 0, 0, 0, BLOCKSOFM, handle->desc.N, handle->ofmblock);
      bn_sum_base2 =  &LIBXSMM_VLA_ACCESS(4, kernel_stats, 1, 0, 0, 0, BLOCKSOFM, handle->desc.N, handle->ofmblock);

      if (handle->n_variants == 2) {
        for (pc = 0; pc < n_segments; pc++) {
          instr = code_stream[pc].segment_type;
          n_convs = code_stream[pc].n_convs;

          if (instr == IMG_LOOP_INIT) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
            if (handle->padding_flag == 1) {
#include "libxsmm_dnn_fwd_custom_custom_padding.tpl.c"
            }
          }

          if (instr == IMG_LOOP_CLOSE) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
	    if ((handle->padding_flag == 1) && (((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_FWD) > 0) ) || ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_RELU_FWD))) {
#include "libxsmm_dnn_fwd_custom_custom_padding_back.tpl.c"
            }
          }
          if ( instr == OFM_LOOP_INIT ) {
            /* Apply bias if requested  */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"
            }
            /* Overwrite output with zeros if requested */
            if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_fwd != 1) ) {
              jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
            }
          }
          if ( instr == IFM_LOOP_FIRST_TOUCH ) {
             ifm1 = code_stream[pc].aux_index;
#include "libxsmm_dnn_fwd_custom_custom_apply_bn.tpl.c"
	  }

          /* Run the stream of convolutions for this segment */
          for (conv_i = 0; conv_i < n_convs; conv_i++) {
            const int vi = variant[pool_index]; /* avoid warning about char used as array index */
            offset_i = stream[i];
            offset_w = stream[i+1];
            offset_o = stream[i+2];
            pi = stream[i+LOCAL_ENTRIES_PER_CONV+0];
            pw = stream[i+LOCAL_ENTRIES_PER_CONV+1];
            po = stream[i+LOCAL_ENTRIES_PER_CONV+2];
            offset_bn = bn_stream[bn_i];
            kernel_pool[vi]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, bn_sum_base + offset_bn, bn_sum_base2 + offset_bn, &scale_factor, max_vals);
            pool_index++;
            i+=LOCAL_ENTRIES_PER_CONV;
            bn_i++;
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
#include "libxsmm_dnn_fwd_custom_custom_padding.tpl.c"
            }
          }

          if (instr == IMG_LOOP_CLOSE) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
	    if ((handle->padding_flag == 1) && (((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_FWD) > 0) ) || ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_RELU_FWD))) {
#include "libxsmm_dnn_fwd_custom_custom_padding_back.tpl.c"
            }
          }
          if ( instr == OFM_LOOP_INIT ) {
            /* Apply bias if requested  */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"
            }
            /* Overwrite output with zeros if requested */
            if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_fwd != 1) ) {
              jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
            }
          }

          if ( instr == IFM_LOOP_FIRST_TOUCH ) {
             ifm1 = code_stream[pc].aux_index;
#include "libxsmm_dnn_fwd_custom_custom_apply_bn.tpl.c"
	  }
          /* Run the stream of convolutions for this segment */
          for (conv_i = 0; conv_i < n_convs; conv_i++) {
            offset_i = stream[i];
            offset_w = stream[i+1];
            offset_o = stream[i+2];
            pi = stream[i+LOCAL_ENTRIES_PER_CONV+0];
            pw = stream[i+LOCAL_ENTRIES_PER_CONV+1];
            po = stream[i+LOCAL_ENTRIES_PER_CONV+2];
            offset_bn = bn_stream[bn_i];
	    if(variant[pool_index] < 2)
	    {
              kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, bn_sum_base + offset_bn, bn_sum_base2 + offset_bn, &scale_factor, max_vals);
	    }
	    else
	    {
              kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, bn_sum_base + offset_bn, bn_sum_base2 + offset_bn, &scale_factor, max_vals);
	    }
	    pool_index++;
            i+=LOCAL_ENTRIES_PER_CONV;
            bn_i++;
          }
        }
      }
    } else { /* We don't do BN stuff in the kernel  */
      if (handle->n_variants == 2) {
        for (pc = 0; pc < n_segments; pc++) {
          instr = code_stream[pc].segment_type;
          n_convs = code_stream[pc].n_convs;

          if (instr == IMG_LOOP_INIT) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
            if (handle->padding_flag == 1) {
#include "libxsmm_dnn_fwd_custom_custom_padding.tpl.c"
            }
          }

          if (instr == IMG_LOOP_CLOSE) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
	    if ((handle->padding_flag == 1) && (((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_FWD) > 0) ) || ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_RELU_FWD))) {
#include "libxsmm_dnn_fwd_custom_custom_padding_back.tpl.c"
            }
          }
          if ( instr == OFM_LOOP_INIT ) {
            /* Apply bias if requested  */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"
            }
            /* Overwrite output with zeros if requested */
            if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_fwd != 1) ) {
              jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
            }
          }

          if ( instr == IFM_LOOP_FIRST_TOUCH ) {
             ifm1 = code_stream[pc].aux_index;
#include "libxsmm_dnn_fwd_custom_custom_apply_bn.tpl.c"
	  }


          if (instr == OFM_LOOP_CLOSE) {
            /* Compute batch norm statistics... */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS) > 0) {
#ifdef __AVX512F__
#ifndef FP64_BN_STATS
              ofm1 =  code_stream[pc].aux_index;
              LIBXSMM_VLA_DECL(4, element_output_type, stats, (element_output_type*)handle->batch_stats->data,  BLOCKSOFM, handle->desc.N, handle->ofmblock);
              element_output_type* red = &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, 0,
                  BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);
              __m512 bsum  = _mm512_setzero_ps();
              __m512 bsum2 = _mm512_setzero_ps();

              for ( oj = 0; oj < handle->ofh; oj++ ) {
                for ( oi = 0; oi < handle->ofw*handle->ofmblock; oi+=16 ) {
                  __m512 btmp = LIBXSMM_INTRINSICS_MM512_LOAD_PS( red+oi );
                  bsum = _mm512_add_ps( bsum, btmp );
                  bsum2 = _mm512_add_ps( bsum2, _mm512_mul_ps( btmp, btmp ) );
                }
                red += handle->ofwp*handle->ofmblock;
              }

              _mm512_store_ps( &LIBXSMM_VLA_ACCESS(4, stats, 0, ofm1, img, 0,
                    BLOCKSOFM, handle->desc.N,  handle->ofmblock), bsum );
              _mm512_store_ps( &LIBXSMM_VLA_ACCESS(4, stats, 1, ofm1, img, 0,
                    BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum2 );
#else
              ofm1 =  code_stream[pc].aux_index;
              {
                LIBXSMM_VLA_DECL(4, double, stats, (double*)handle->batch_stats->data,  BLOCKSOFM, handle->desc.N, handle->ofmblock);
                element_output_type* red = &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, 0,
                    BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);
                __m512d bsum1a = _mm512_setzero_pd();
                __m512d bsum1b = _mm512_setzero_pd();
                __m512d bsum2a = _mm512_setzero_pd();
                __m512d bsum2b = _mm512_setzero_pd();

                for ( oj = 0; oj < handle->ofh; oj++ ) {
                  for ( oi = 0; oi < handle->ofw*handle->ofmblock; oi+=16 ) {
                    __m512d btmpa = _mm512_cvtps_pd ( _mm256_load_ps((const float*) (red+oi+0)) );
                    __m512d btmpb = _mm512_cvtps_pd ( _mm256_load_ps((const float*) (red+oi+8)) );
                    bsum1a = _mm512_add_pd( bsum1a, btmpa);
                    bsum1b = _mm512_add_pd( bsum1b, btmpb);
                    bsum2a = _mm512_add_pd( bsum2a, _mm512_mul_pd( btmpa, btmpa ) );
                    bsum2b = _mm512_add_pd( bsum2b, _mm512_mul_pd( btmpb, btmpb ) );
                  }
                  red += handle->ofwp*handle->ofmblock;
                }

                _mm512_store_pd( &LIBXSMM_VLA_ACCESS(4, stats, 0, ofm1, img, 0,
                      BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum1a );
                _mm512_store_pd( &LIBXSMM_VLA_ACCESS(4, stats, 0, ofm1, img, 8,
                      BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum1b );
                _mm512_store_pd( &LIBXSMM_VLA_ACCESS(4, stats, 1, ofm1, img, 0,
                      BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum2a );
                _mm512_store_pd( &LIBXSMM_VLA_ACCESS(4, stats, 1, ofm1, img, 8,
                      BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum2b );
              }
#endif
#else
              /* won't happen as this code only runs on AVX512 platforms */
#endif
            }

            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
              element_output_type* cur_vec = &LIBXSMM_VLA_ACCESS(5, output, img, /*ofm1*/code_stream[pc].aux_index, 0, 0, 0,
                  BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);
              for ( oj = 0; oj < handle->ofh; oj++ ) {
                for ( oi = 0; oi < handle->ofw*handle->ofmblock; oi+=16 ) {
#ifdef __AVX512F__
                  max_abs = _mm512_max_ps(max_abs, LIBXSMM_INTRINSICS_MM512_ABS_PS(LIBXSMM_INTRINSICS_MM512_LOAD_PS(cur_vec+oi)));
#else
                  /* Won't happen as this code only runs on AVX512 systems */
#endif
                }
                cur_vec += handle->ofwp*handle->ofmblock;
              }
            }
          }

          /* Run the stream of convolutions for this segment */
          for (conv_i = 0; conv_i < n_convs; conv_i++) {
            const int vi = variant[pool_index]; /* avoid warning about char used as array index */
            offset_i = stream[i];
            offset_w = stream[i+1];
            offset_o = stream[i+2];
            pi = stream[i+LOCAL_ENTRIES_PER_CONV+0];
            pw = stream[i+LOCAL_ENTRIES_PER_CONV+1];
            po = stream[i+LOCAL_ENTRIES_PER_CONV+2];
            kernel_pool[vi]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
            pool_index++;
            i+=LOCAL_ENTRIES_PER_CONV;
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
#include "libxsmm_dnn_fwd_custom_custom_padding.tpl.c"
            }
          }

          if (instr == IMG_LOOP_CLOSE) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
	    if ((handle->padding_flag == 1) && (((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_FWD) > 0) ) || ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_RELU_FWD))) {
#include "libxsmm_dnn_fwd_custom_custom_padding_back.tpl.c"
            }
          }
          if ( instr == OFM_LOOP_INIT ) {
            /* Apply bias if requested  */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"
            }
            /* Overwrite output with zeros if requested */
            if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_fwd != 1) ) {
              jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
            }
          }

          if ( instr == IFM_LOOP_FIRST_TOUCH ) {
             ifm1 = code_stream[pc].aux_index;
#ifndef FUSED_BN_CONV_WRAPPER
#include "libxsmm_dnn_fwd_custom_custom_apply_bn.tpl.c"
#endif
	  }
          if ( instr == OFM_LOOP_CLOSE ) {
            /* Compute batch norm statistics... */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS) > 0) {
#ifdef __AVX512F__
#ifndef FP64_BN_STATS
              ofm1 =  code_stream[pc].aux_index;
              LIBXSMM_VLA_DECL(4, element_output_type, stats, (element_output_type*)handle->batch_stats->data,  BLOCKSOFM, handle->desc.N, handle->ofmblock);
              element_output_type* red = &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, 0,
                  BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);
              __m512 bsum  = _mm512_setzero_ps();
              __m512 bsum2 = _mm512_setzero_ps();

              for ( oj = 0; oj < handle->ofh; oj++ ) {
                for ( oi = 0; oi < handle->ofw*handle->ofmblock; oi+=16 ) {
                  __m512 btmp = LIBXSMM_INTRINSICS_MM512_LOAD_PS( red+oi );
                  bsum = _mm512_add_ps( bsum, btmp );
                  bsum2 = _mm512_add_ps( bsum2, _mm512_mul_ps( btmp, btmp ) );
                }
                red += handle->ofwp*handle->ofmblock;
              }

              _mm512_store_ps( &LIBXSMM_VLA_ACCESS(4, stats, 0, ofm1, img, 0,
                    BLOCKSOFM, handle->desc.N,  handle->ofmblock), bsum );
              _mm512_store_ps( &LIBXSMM_VLA_ACCESS(4, stats, 1, ofm1, img, 0,
                    BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum2 );
#else
              ofm1 =  code_stream[pc].aux_index;
              {
                LIBXSMM_VLA_DECL(4, double, stats, (double*)handle->batch_stats->data,  BLOCKSOFM, handle->desc.N, handle->ofmblock);
                element_output_type* red = &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, 0,
                    BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);
                __m512d bsum1a = _mm512_setzero_pd();
                __m512d bsum1b = _mm512_setzero_pd();
                __m512d bsum2a = _mm512_setzero_pd();
                __m512d bsum2b = _mm512_setzero_pd();

                for ( oj = 0; oj < handle->ofh; oj++ ) {
                  for ( oi = 0; oi < handle->ofw*handle->ofmblock; oi+=16 ) {
                    __m512d btmpa = _mm512_cvtps_pd ( _mm256_load_ps((const float*) (red+oi+0)) );
                    __m512d btmpb = _mm512_cvtps_pd ( _mm256_load_ps((const float*) (red+oi+8)) );
                    bsum1a = _mm512_add_pd( bsum1a, btmpa);
                    bsum1b = _mm512_add_pd( bsum1b, btmpb);
                    bsum2a = _mm512_add_pd( bsum2a, _mm512_mul_pd( btmpa, btmpa ) );
                    bsum2b = _mm512_add_pd( bsum2b, _mm512_mul_pd( btmpb, btmpb ) );
                  }
                  red += handle->ofwp*handle->ofmblock;
                }

                _mm512_store_pd( &LIBXSMM_VLA_ACCESS(4, stats, 0, ofm1, img, 0,
                      BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum1a );
                _mm512_store_pd( &LIBXSMM_VLA_ACCESS(4, stats, 0, ofm1, img, 8,
                      BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum1b );
                _mm512_store_pd( &LIBXSMM_VLA_ACCESS(4, stats, 1, ofm1, img, 0,
                      BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum2a );
                _mm512_store_pd( &LIBXSMM_VLA_ACCESS(4, stats, 1, ofm1, img, 8,
                      BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum2b );
              }
#endif
#else
              /* won't happen as this code only runs on AVX512 platforms */
#endif
            }

            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
              element_output_type* cur_vec = &LIBXSMM_VLA_ACCESS(5, output, img, /*ofm1*/code_stream[pc].aux_index, 0, 0, 0,
                  BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);
              for ( oj = 0; oj < handle->ofh; oj++ ) {
                for ( oi = 0; oi < handle->ofw*handle->ofmblock; oi+=16 ) {
#ifdef __AVX512F__
                  max_abs = _mm512_max_ps(max_abs, LIBXSMM_INTRINSICS_MM512_ABS_PS(LIBXSMM_INTRINSICS_MM512_LOAD_PS(cur_vec+oi)));
#else
                  /* won't happen as this code only runs on AVX512 platforms */
#endif
                }
                cur_vec += handle->ofwp*handle->ofmblock;
              }
            }
          }

          /* Run the stream of convolutions for this segment */
          for (conv_i = 0; conv_i < n_convs; conv_i++) {
            offset_i = stream[i];
            offset_w = stream[i+1];
            offset_o = stream[i+2];
            offset_i_st = stream[i+3];
            oi = stream[i+4];
            oj = stream[i+5];
            ifm1 = stream[i+6];
            pi = stream[i+LOCAL_ENTRIES_PER_CONV+0];
            pw = stream[i+LOCAL_ENTRIES_PER_CONV+1];
            po = stream[i+LOCAL_ENTRIES_PER_CONV+2];
	    if(variant[pool_index] < 2)
	    {
              kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
	    }
	    else
	    {
#ifdef FUSED_BN_CONV_WRAPPER
              wrapper_kernel(kernel, input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals, handle, ifm1, padded_w, padded_h, img, BLOCKSIFM, ltid, offset_i, pi, input_st_base + offset_i_st, oi, oj);
#else
              kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
#endif
	    }
	    pool_index++;
            i+=LOCAL_ENTRIES_PER_CONV;
          }
        }
      }
    }
  } else {
    /* Use fine-grained operations since we are in the img_par path, so update relevant kernel pointers... */
    jitted_matcopy = handle->matcopy_fwd[2].xmatcopy;
    jitted_zero_overwrite = handle->matcopy_fwd[3].xmatcopy;
    input_h_start = LIBXSMM_MAX(0,  handle->ofh_fwd_start[ltid] - handle->desc.R + 1);
    input_h_end = LIBXSMM_MIN( handle->ifhp, (handle->ofh_fwd_end[ltid] + handle->desc.R -1) * handle->desc.u );
    my_h_out = handle->ofh_fwd_end[ltid]-handle->ofh_fwd_start[ltid];
    for (pc = 0; pc < n_segments; pc++) {
      instr = code_stream[pc].segment_type;
      n_convs = code_stream[pc].n_convs;
      if (instr == IMG_LOOP_INIT) {
        /* Padding code via jitted matcopy kernel */
#include "libxsmm_dnn_fwd_custom_custom_padding_img_par.tpl.c"
      }

      if (instr == IMG_LOOP_CLOSE) {
        img = code_stream[pc].aux_index;
        /* Apply padding  */
        if ((handle->padding_flag == 1) && (((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_FWD) > 0) ) || ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_RELU_FWD))) {
#include "libxsmm_dnn_fwd_custom_custom_padding_back.tpl.c"
        }
      }

      if ( instr == OFM_LOOP_INIT ) {
        /* Apply bias if requested  */
        if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
#include "libxsmm_dnn_fwd_custom_custom_bias_img_par.tpl.c"
        }
        /* Overwrite output with zeros if requested */
        if ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) {
#if 0
          for ( ih = 0; ih < my_h_out * handle->ofmblock * handle->ofwp; ih += handle->ofmblock * handle->ofwp) {
            jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2] + ih, NULL, NULL);
          }
#endif
          int h,w;
          __m512 zero_reg  = _mm512_setzero_ps();
          for (h = 0; h<handle->fwd_ofh_rb; h++) {
            for (w = 0; w<handle->fwd_ofw_rb; w++) {
              _mm512_store_ps(output_base+stream[i+2]+w*handle->ofmblock+h*handle->ofwp*handle->ofmblock, zero_reg);
            }
          }
        }
      }
      if ( instr == IFM_LOOP_FIRST_TOUCH ) {
           ifm1 = code_stream[pc].aux_index;
#include "libxsmm_dnn_fwd_custom_custom_apply_bn.tpl.c"
      }

      /* Run the stream of convolutions for this segment */
      for (conv_i = 0; conv_i < n_convs; conv_i++) {
        offset_i = stream[i];
        offset_w = stream[i+1];
        offset_o = stream[i+2];
        oi = stream[i+4];
        oj = stream[i+5];
        ifm1 = stream[i+6];
        pi = stream[i+LOCAL_ENTRIES_PER_CONV+0];
        pw = stream[i+LOCAL_ENTRIES_PER_CONV+1];
        po = stream[i+LOCAL_ENTRIES_PER_CONV+2];
	if(variant[pool_index] < 2)
        {
	  kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
	}
	else
	{
	  kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
	}
	pool_index++;
        i+=LOCAL_ENTRIES_PER_CONV;
      }
    }
  }
} else {
  /* Run the stream of convolutions, no extra operations are required... */
  if ( handle->compute_batch_stats_in_kernel == 1 ) { /* We  do BN stuff in the kernel  */
#ifndef FP64_BN_STATS
    LIBXSMM_VLA_DECL(4, element_output_type, kernel_stats, (element_output_type*)handle->batch_stats->data, BLOCKSOFM, handle->desc.N, handle->ofmblock);
#else
    LIBXSMM_VLA_DECL(4, double, kernel_stats, (double*)handle->batch_stats->data, BLOCKSOFM, handle->desc.N, handle->ofmblock);
#endif
    bn_sum_base =  &LIBXSMM_VLA_ACCESS(4, kernel_stats, 0, 0, 0, 0, BLOCKSOFM, handle->desc.N, handle->ofmblock);
    bn_sum_base2 =  &LIBXSMM_VLA_ACCESS(4, kernel_stats, 1, 0, 0, 0, BLOCKSOFM, handle->desc.N, handle->ofmblock);
    if (handle->n_variants  == 2) {
      for (pc = 0; pc < instr; pc += 1) {
        const int vi = variant[pc]; /* avoid warning about char used as array index */
        offset_i = stream[i];
        offset_w = stream[i+1];
        offset_o = stream[i+2];
        pi = stream[i+LOCAL_ENTRIES_PER_CONV+0];
        pw = stream[i+LOCAL_ENTRIES_PER_CONV+1];
        po = stream[i+LOCAL_ENTRIES_PER_CONV+2];
        offset_bn = bn_stream[bn_i];
        kernel_pool[vi]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, bn_sum_base + offset_bn, bn_sum_base2 + offset_bn, &scale_factor, max_vals);
        i+=LOCAL_ENTRIES_PER_CONV;
        bn_i++;
      }
    } else {
      for (pc = 0; pc < instr; pc++) {
        offset_i = stream[i];
        offset_w = stream[i+1];
        offset_o = stream[i+2];
        pi = stream[i+LOCAL_ENTRIES_PER_CONV+0];
        pw = stream[i+LOCAL_ENTRIES_PER_CONV+1];
        po = stream[i+LOCAL_ENTRIES_PER_CONV+2];
        offset_bn = bn_stream[bn_i];
	if(variant[pool_index] < 2)
	{
          kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po,  bn_sum_base + offset_bn, bn_sum_base2 + offset_bn, &scale_factor, max_vals);
	}
	else
	{
          kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po,  bn_sum_base + offset_bn, bn_sum_base2 + offset_bn, &scale_factor, max_vals);
	}
	pool_index++;
        i+=LOCAL_ENTRIES_PER_CONV;
        bn_i++;
      }
    }
  } else { /* We do not  do BN stuff in the kernel  */
    if (handle->n_variants == 2) {
      for (pc = 0; pc < instr; pc += 1) {
        const int vi = variant[pc]; /* avoid warning about char used as array index */
        offset_i = stream[i];
        offset_w = stream[i+1];
        offset_o = stream[i+2];
        pi = stream[i+LOCAL_ENTRIES_PER_CONV+0];
        pw = stream[i+LOCAL_ENTRIES_PER_CONV+1];
        po = stream[i+LOCAL_ENTRIES_PER_CONV+2];
        kernel_pool[variant[pc]]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
        i+=LOCAL_ENTRIES_PER_CONV;
      }
    } else {
      for (pc = 0; pc < instr; pc++) {
        offset_i = stream[i];
        offset_w = stream[i+1];
        offset_o = stream[i+2];
        pi = stream[i+LOCAL_ENTRIES_PER_CONV+0];
        pw = stream[i+LOCAL_ENTRIES_PER_CONV+1];
        po = stream[i+LOCAL_ENTRIES_PER_CONV+2];
	if(variant[pool_index] < 2)
	{
          kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
	}
	else
	{
          kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
	}
	pool_index++;
        i+=LOCAL_ENTRIES_PER_CONV;
      }
    }
  }
}

if ( ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) && (handle->use_lp_kernel == 1) && (handle->compute_max_in_kernel_fwd == 0) ) {
#ifdef __AVX512F__
  _mm512_store_ps(max_vals, max_abs);
#else
  /* won't happen as this code only runs on AVX512 platforms */
#endif
}

libxsmm_barrier_wait(handle->barrier, ltid);

#undef IMG_LOOP_INIT
#undef OFM_LOOP_INIT
#undef OFM_LOOP_CLOSE
#undef CONVOLUTION_KERNEL
#undef IFM_LOOP_CLOSE_S

