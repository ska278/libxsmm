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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

/* use for-loops to potentially leverage NUMA in the future */
int i1, i2, i3, i4;
int difmblock = tensor->layout->dim_size[0];
int ddescn = tensor->layout->dim_size[1];
int dblocksifm = tensor->layout->dim_size[2];
int d2 = tensor->layout->dim_size[3];

element_type* user_data = (element_type*)data;
LIBXSMM_VLA_DECL(4, const element_type, handle_data, (const element_type*)tensor->data, dblocksifm, ddescn, difmblock);

for (i1 = 0; i1 < d2; ++i1) {
  for (i2 = 0; i2 < dblocksifm; ++i2) {
    for (i3 = 0; i3 < ddescn; ++i3) {
      for (i4 = 0; i4 < difmblock; ++i4) {
        user_data[i1*dblocksifm*ddescn*difmblock + i2*difmblock + i3*difmblock*dblocksifm + i4] = LIBXSMM_VLA_ACCESS(4, handle_data, i1, i2, i3, i4, dblocksifm, ddescn, difmblock);
//        user_data[(i1*dblocksifm*ddescn*difmblock) + (i2*ddescn*difmblock) + i3*difmblock + i4] = LIBXSMM_VLA_ACCESS(4, handle_data, i1, i2, i3, i4, dblocksifm, ddescn, difmblock);
      }
    }
  }
}

