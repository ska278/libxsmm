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
/* Evangelos Georganas (Intel Corp.)
 ******************************************************************************/

/* Padding code via jitted matcopy kernel */

int my_h, my_w, my_c;
for (ifm1 = handle->blocksifm-1; ifm1 >= 0; ifm1--) {
  input_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(6, input, img, ifm1, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
  copy_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_buffer, ifm1, handle->desc.pad_h, handle->desc.pad_w, 0, 0, padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
  for(my_h = 0 ; my_h < handle->desc.H ; my_h++)
  {
    for(my_w = 0 ; my_w < handle->desc.W ; my_w++)
    {
      #pragma omp simd
      for(my_c = 0 ; my_c < 16 ; my_c++)
      {
        input_ptr[my_c + (my_w + handle->desc.pad_w_in) * handle->ifmblock + (my_h + handle->desc.pad_h_in) * handle->ifmblock * handle->ifwp] =  
          copy_ptr[my_c + (my_w) * handle->ifmblock + (my_h) * handle->ifmblock * padded_w];
      }
    }
  }
}

