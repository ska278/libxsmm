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

LIBXSMM_VLA_DECL(4, element_input_type, lcl_gamma_beta, (element_input_type*)handle->reg_lcl_gamma_beta->data, BLOCKSIFM, handle->desc.N, handle->ifmblock);
LIBXSMM_VLA_DECL(2, element_input_type, bmean2, (element_input_type*)handle->reg_bmean2->data, handle->ifmblock);
LIBXSMM_VLA_DECL(2, element_input_type, brstd2, (element_input_type*)handle->reg_brstd2->data, handle->ifmblock);

int my_ldw, my_pad_h, my_pad_w;
int ifm_idx, my_h, my_w, my_c;
const int padded_h = handle->ifhp + 2 * handle->desc.pad_h;
const int padded_w = handle->ifwp + 2 * handle->desc.pad_w;
//for(ifm_idx = ifm1 ; ifm_idx < ifm1 + handle->blocksifm_blocking ; ifm_idx++ ) 
ifm_idx = ifm1;
{
  element_input_type * myinput;
  if (handle->padding_flag == 1) {
    LIBXSMM_VLA_DECL(6, element_input_type, input, (element_input_type*)handle->reg_input_st_bwd2->data, BLOCKSIFM, padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
    myinput = (element_input_type*) &LIBXSMM_VLA_ACCESS(6, input, img, ifm_idx, 0, 0, 0, 0,
      BLOCKSIFM, padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
    assert(myinput);
    my_ldw = padded_w;
    my_pad_h = handle->desc.pad_h;
    my_pad_w = handle->desc.pad_w;
  } else {
    LIBXSMM_VLA_DECL(6, element_input_type, input, (element_input_type*)handle->reg_input_st_bwd2->data, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
    myinput = (element_input_type*) &LIBXSMM_VLA_ACCESS(6, input, img, ifm_idx, 0, 0, 0, 0,
        BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
    assert(myinput);
    my_ldw = handle->ifwp;
    my_pad_h = handle->desc.pad_h_in;
    my_pad_w = handle->desc.pad_w_in;
  }
  element_input_type * mybmean2 = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, bmean2, ifm_idx, 0, handle->ifmblock));
  element_input_type * mybrstd2 = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, brstd2, ifm_idx, 0, handle->ifmblock));
  assert(mybmean2);
  assert(mybrstd2);

  for(my_h = 0 ; my_h < handle->desc.H ; my_h++)
  {
    for(my_w = 0 ; my_w < handle->desc.W ; my_w++)
    {
      for(my_c = 0 ; my_c < handle->ifmblock ; my_c++)
      {
        int _my_h = my_h + my_pad_h;
        int _my_w = my_w + my_pad_w;
        (LIBXSMM_VLA_ACCESS(  4, lcl_gamma_beta, 0, ifm_idx, img, my_c, BLOCKSIFM, handle->desc.N, handle->ifmblock)) += 
	    (myinput[my_c + _my_w * handle->ifmblock + _my_h * handle->ifmblock * my_ldw] - mybmean2[my_c]) * 
	    (myinput[my_c + _my_w * handle->ifmblock + _my_h * handle->ifmblock * my_ldw]) * mybrstd2[my_c];
        (LIBXSMM_VLA_ACCESS(  4, lcl_gamma_beta, 1, ifm_idx, img, my_c, BLOCKSIFM, handle->desc.N, handle->ifmblock)) +=
	    (myinput[my_c + _my_w * handle->ifmblock + _my_h * handle->ifmblock * my_ldw]);
      }
    }
  }
}




