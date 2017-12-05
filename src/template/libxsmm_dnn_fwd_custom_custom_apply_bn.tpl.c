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

LIBXSMM_VLA_DECL(2, element_input_type, expect, (element_input_type*)handle->reg_expect->data, handle->ifmblock);
LIBXSMM_VLA_DECL(2, element_input_type, stddev, (element_input_type*)handle->reg_stddev->data, handle->ifmblock);
LIBXSMM_VLA_DECL(2, element_input_type, gamma, (element_input_type*)handle->reg_gamma->data, handle->ifmblock);
LIBXSMM_VLA_DECL(2, element_input_type, beta, (element_input_type*)handle->reg_beta->data, handle->ifmblock);

int my_h, my_w, my_c, ifm_idx;
for(ifm_idx = ifm1 ; ifm_idx < ifm1 + handle->blocksifm_blocking ; ifm_idx++ ) 
{
  float * myinput;
  if (handle->padding_flag == 1) {
    myinput = (float*) &LIBXSMM_VLA_ACCESS(5, input_buffer, img, 0, 0, 0, 0,
      padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
  } else {
    myinput = (float*) &LIBXSMM_VLA_ACCESS(6, input, img, ifm_idx, 0, 0, 0, 0,
        BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
  }
  float * myexpect = (float*) &(LIBXSMM_VLA_ACCESS(  2, expect, ifm_idx, 0, handle->ifmblock));
  float * mystddev = (float*) &(LIBXSMM_VLA_ACCESS(  2, stddev, ifm_idx, 0, handle->ifmblock));
  float * mygamma = (float*) &(LIBXSMM_VLA_ACCESS(  2, gamma, ifm_idx, 0, handle->ifmblock));
  float * mybeta = (float*) &(LIBXSMM_VLA_ACCESS(  2, beta, ifm_idx, 0, handle->ifmblock));
  for(my_h = 0 ; my_h < handle->ifhp ; my_h++) 
  {
    for(my_w = 0 ; my_w < handle->ifwp ; my_w++)
    {
      for(my_c = 0 ; my_c < handle->ifmblock ; my_c++)
      {
        int _my_h = my_h + handle->desc.pad_h;
        int _my_w = my_w + handle->desc.pad_w;
        float after = (myinput[my_c + _my_w * handle->ifmblock + _my_h * handle->ifmblock * handle->ifwp] - myexpect[my_c]) / mystddev[my_c] * mygamma[my_c] + mybeta[my_c];
        myinput[my_c + _my_w * handle->ifmblock + _my_h * handle->ifmblock * handle->ifwp] = after;
      }
    }
  }
}
