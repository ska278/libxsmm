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
for (ofm1 = handle->blocksofm-1; ofm1 >= 0; ofm1--) {
  input_ptr = (element_output_type*)&LIBXSMM_VLA_ACCESS(6, del_out, img, ofm1, 0, 0, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);
  copy_ptr = (element_output_type*)&LIBXSMM_VLA_ACCESS(5, output_buffer, ofm1, handle->desc.pad_h, handle->desc.pad_w, 0, 0, padded_h, padded_w, handle->ofmblock_lp, handle->fm_lp_block);

  for(my_h = 0 ; my_h < handle->desc.H ; my_h+=handle->desc.u)
  {
    for(my_w = 0 ; my_w < handle->desc.W ; my_w+=handle->desc.v)
    {
      #pragma omp simd
      for(my_c = 0 ; my_c < 16 ; my_c++)
      {
        input_ptr[my_c + (my_w/handle->desc.v + handle->desc.pad_w_out) * handle->ofmblock_lp + (my_h/handle->desc.u + handle->desc.pad_h_out) * handle->ofmblock_lp * handle->ofwp] =
          copy_ptr[my_c + (my_w/handle->desc.v) * handle->ofmblock_lp + (my_h/handle->desc.u) * handle->ofmblock_lp * padded_w];
      }
    }
  }
}



