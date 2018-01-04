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

int my_h, my_w, my_c, ifm_idx, my_ldw, my_pad_h, my_pad_w;
for(ifm_idx = ifm1 ; ifm_idx < ifm1 + handle->blocksifm_blocking ; ifm_idx++ ) 
{
  element_input_type * myinput;
  element_input_type * myinput_st;
  if (handle->padding_flag == 1) {
    LIBXSMM_VLA_DECL(6, element_input_type, input_st, (element_input_type*)handle->reg_input_st->data, BLOCKSIFM, padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
    myinput = (element_input_type*) &LIBXSMM_VLA_ACCESS(5, input_buffer, ifm_idx, 0, 0, 0, 0,
      padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
    myinput_st = (element_input_type*) &LIBXSMM_VLA_ACCESS(6, input_st, img, ifm_idx, 0, 0, 0, 0,
        BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
    my_ldw = padded_w;
    my_pad_h = handle->desc.pad_h;
    my_pad_w = handle->desc.pad_w;
  } else {
    LIBXSMM_VLA_DECL(6, element_input_type, input_st, (element_input_type*)handle->reg_input_st->data, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
    myinput = (element_input_type*) &LIBXSMM_VLA_ACCESS(6, input, img, ifm_idx, 0, 0, 0, 0,
        BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
    myinput_st = (element_input_type*) &LIBXSMM_VLA_ACCESS(6, input_st, img, ifm_idx, 0, 0, 0, 0,
        BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
    my_ldw = handle->ifwp;
    my_pad_h = handle->desc.pad_h_in;
    my_pad_w = handle->desc.pad_w_in;
  }
  element_input_type * myexpect = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, expect, ifm_idx, 0, handle->ifmblock));
  element_input_type * mystddev = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, stddev, ifm_idx, 0, handle->ifmblock));
  element_input_type * mygamma = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, gamma, ifm_idx, 0, handle->ifmblock));
  element_input_type * mybeta = (element_input_type*) &(LIBXSMM_VLA_ACCESS(  2, beta, ifm_idx, 0, handle->ifmblock));
  if(handle->ifmblock == 16)
  {
#ifdef __AVX512F__

    // load batch norm parameters
    __m512 _expect = _mm512_load_ps(myexpect);
    __m512 _stddev = _mm512_load_ps(mystddev);
    __m512 _gamma = _mm512_load_ps(mygamma);
    __m512 _beta = _mm512_load_ps(mybeta);
    for(my_h = 0 ; my_h < handle->desc.H ; my_h++) 
    {
      for(my_w = 0 ; my_w < handle->desc.W ; my_w++)
      {
        int _my_h = my_h + my_pad_h;
        int _my_w = my_w + my_pad_w;
        int _my_h_st = my_h + handle->desc.pad_h_in;
        int _my_w_st = my_w + handle->desc.pad_w_in;

	// load input
	__m512 _input = _mm512_load_ps(&myinput[_my_w * handle->ifmblock + _my_h * handle->ifmblock * my_ldw]);

	// Streaming store input to other buffer
	//_mm512_stream_ps(&myinput_st[_my_w * handle->ifmblock + _my_h * handle->ifmblock * my_ldw], _input);
	_mm512_stream_ps(&myinput_st[_my_w_st * handle->ifmblock + _my_h_st * handle->ifmblock * handle->ifwp], _input);

	// Apply bn
	_input = _mm512_add_ps( _mm512_mul_ps( _mm512_mul_ps( _mm512_sub_ps(_input, _expect) , _stddev), _gamma), _beta);

	// Apply relu
	__m512 _zero = _mm512_set1_ps(0.f);
	__mmask16 msk = _mm512_cmp_ps_mask(_zero, _input, 1);
	_input = _mm512_maskz_add_ps(msk, _zero, _input);
	_mm512_store_ps(&myinput[_my_w * handle->ifmblock + _my_h * handle->ifmblock * my_ldw], _input);
      }
    }
#else
    for(my_h = 0 ; my_h < handle->desc.H ; my_h++) 
    {
      for(my_w = 0 ; my_w < handle->desc.W ; my_w++)
      {
        #pragma omp simd
        #pragma vector aligned nontemporal(myinput_st)
        for(my_c = 0 ; my_c < 16 ; my_c++)
        {
          int _my_h = my_h + my_pad_h;
          int _my_w = my_w + my_pad_w;

	  // Streaming store input to other buffer
	  myinput_st[my_c + _my_w * handle->ifmblock + _my_h * handle->ifmblock * my_ldw] = myinput[my_c + _my_w * handle->ifmblock + _my_h * handle->ifmblock * my_ldw];

          element_input_type after = (myinput[my_c + _my_w * handle->ifmblock + _my_h * handle->ifmblock * my_ldw] - myexpect[my_c]) * mystddev[my_c] * mygamma[my_c] + mybeta[my_c];
          myinput[my_c + _my_w * handle->ifmblock + _my_h * handle->ifmblock * my_ldw] = (after > 0.f) ? after : 0.f;
        }
      }
    }
#endif
  } else {
    for(my_h = 0 ; my_h < handle->desc.H ; my_h++) 
    {
      for(my_w = 0 ; my_w < handle->desc.W ; my_w++)
      {
        #pragma omp simd
        #pragma vector aligned nontemporal(myinput_st)
        for(my_c = 0 ; my_c < handle->ifmblock ; my_c++)
        {
          int _my_h = my_h + my_pad_h;
          int _my_w = my_w + my_pad_w;

	  // Streaming store input to other buffer
	  myinput_st[my_c + _my_w * handle->ifmblock + _my_h * handle->ifmblock * my_ldw] = myinput[my_c + _my_w * handle->ifmblock + _my_h * handle->ifmblock * my_ldw];

          element_input_type after = (myinput[my_c + _my_w * handle->ifmblock + _my_h * handle->ifmblock * my_ldw] - myexpect[my_c]) * mystddev[my_c] * mygamma[my_c] + mybeta[my_c];
          myinput[my_c + _my_w * handle->ifmblock + _my_h * handle->ifmblock * my_ldw] = (after > 0) ? after : 0.;
        }
      }
    }
  }
}
