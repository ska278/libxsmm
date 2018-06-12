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
/* Evangelos Georganas (Intel Corp.) Michael Anderson (Intel Corp.)
 ******************************************************************************/

if((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_BWD) > 0)
{

LIBXSMM_VLA_DECL(2, element_input_type, bmean1, (element_input_type*)handle->reg_bmean1->data, handle->ofmblock);
LIBXSMM_VLA_DECL(2, element_input_type, brstd1, (element_input_type*)handle->reg_brstd1->data, handle->ofmblock);
LIBXSMM_VLA_DECL(2, element_input_type, gamma, (element_input_type*)handle->reg_gamma_bwd->data, handle->ofmblock);
LIBXSMM_VLA_DECL(2, element_input_type, dbeta, (element_input_type*)handle->grad_beta->data, handle->ofmblock);
LIBXSMM_VLA_DECL(2, element_input_type, dgamma, (element_input_type*)handle->grad_gamma->data, handle->ofmblock);

int ofm_idx, my_h, my_w, my_c;
int my_ldw, my_pad_h, my_pad_w;
const int padded_h = handle->ofhp + 2 * handle->desc.pad_h;
const int padded_w = handle->ofwp + 2 * handle->desc.pad_w;
for(ofm_idx = ofm1 ; ofm_idx < ofm1 + handle->blocksofm_blocking ; ofm_idx++ ) 
{
  element_input_type * myoutput;
  element_input_type * myinput_r;
  if (handle->padding_flag == 1) {
    LIBXSMM_VLA_DECL(6, element_input_type, input_r, (element_input_type*)handle->reg_input_st_bwd->data, BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
    myinput_r = (element_input_type*) &LIBXSMM_VLA_ACCESS(6, input_r, img, ofm_idx, 0, 0, 0, 0,
        BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
    LIBXSMM_VLA_DECL(6, element_input_type, output, (element_input_type*)handle->grad_output->data, BLOCKSOFM, padded_h, padded_w, handle->ofmblock, handle->fm_lp_block);
    myoutput = (element_input_type*) &LIBXSMM_VLA_ACCESS(5, output_buffer, ofm_idx, 0, 0, 0, 0,
               padded_h, padded_w, handle->ofmblock, handle->fm_lp_block);

    // Switch to scratch
    my_ldw = padded_w;
    my_pad_h = handle->desc.pad_h;
    my_pad_w = handle->desc.pad_w;
  } else {
    LIBXSMM_VLA_DECL(6, element_input_type, input_r, (element_input_type*)handle->reg_input_st_bwd->data, BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
    myinput_r = (element_input_type*) &LIBXSMM_VLA_ACCESS(6, input_r, img, ofm_idx, 0, 0, 0, 0,
        BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
    LIBXSMM_VLA_DECL(6, element_input_type, output, (element_input_type*)handle->grad_output->data, BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
    myoutput = (element_input_type*) &LIBXSMM_VLA_ACCESS(6, output, img, ofm_idx, 0, 0, 0, 0,
        BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
    my_ldw = handle->ofwp;
    my_pad_h = handle->desc.pad_h_out;
    my_pad_w = handle->desc.pad_w_out;
  }

  element_input_type * mygamma = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, gamma, ofm_idx, 0, handle->ofmblock));
  element_input_type * mydgamma = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, dgamma, ofm_idx, 0, handle->ofmblock));
  element_input_type * mydbeta = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, dbeta, ofm_idx, 0, handle->ofmblock));
  element_input_type * mybmean1 = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, bmean1, ofm_idx, 0, handle->ofmblock));
  element_input_type * mybrstd1 = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, brstd1, ofm_idx, 0, handle->ofmblock));

  element_input_type nhw = handle->desc.N * handle->ofw * handle->ofh;
  element_input_type recp_nhw = 1.0f/nhw; 

  for(my_h = 0 ; my_h < handle->ofh ; my_h+=1)
  {
    for(my_w = 0 ; my_w < handle->ofw ; my_w+=1)
    {
      #pragma omp simd
      #pragma vector aligned
      for(my_c = 0 ; my_c < handle->ofmblock ; my_c++)
      {
        int _my_h = (my_h) + my_pad_h;
        int _my_w = (my_w) + my_pad_w;
	  myoutput[my_c + _my_w * handle->ofmblock + _my_h * handle->ofmblock * my_ldw] = 
            mygamma[my_c] * mybrstd1[my_c] * recp_nhw * (nhw * myoutput[my_c + _my_w * handle->ofmblock + _my_h * handle->ofmblock * my_ldw] - (mydbeta[my_c] + (myinput_r[my_c + (_my_w + handle->desc.pad_w_out) * handle->ofmblock + (_my_h + handle->desc.pad_h_out) * handle->ofmblock * handle->ofwp]  - mybmean1[my_c]) * mydgamma[my_c] * mybrstd1[my_c]));
      }
    }
  }
}
}

