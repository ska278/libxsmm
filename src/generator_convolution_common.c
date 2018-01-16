/******************************************************************************
** Copyright (c) 2015-2017, Intel Corporation                                **
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
/* Alexander Heinecke, Rajkishore Barik (Intel Corp.)
******************************************************************************/

#include "generator_x86_instructions.h"
#include "generator_convolution_common.h"
#include "generator_common.h"

#include <libxsmm_intrinsics_x86.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FP64_BN_STATS

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_m_loop( libxsmm_generated_code*                   io_generated_code,
                                                  libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                  const libxsmm_matcopy_kernel_config*      i_kernel_config,
                                                  const unsigned int                        i_gp_reg_m_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_add_instruction, i_gp_reg_m_loop, 1);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_m_loop( libxsmm_generated_code*                       io_generated_code,
                                                  libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                  const libxsmm_matcopy_kernel_config*          i_kernel_config,
                                                  const unsigned int                            i_gp_reg_m_loop,
                                                  const unsigned int                            i_m ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_cmp_instruction, i_gp_reg_m_loop, i_m );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_n_loop( libxsmm_generated_code*                  io_generated_code,
                                                 libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                 const libxsmm_matcopy_kernel_config*      i_kernel_config,
                                                 const unsigned int                        i_gp_reg_n_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_add_instruction, i_gp_reg_n_loop, 1);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_n_loop( libxsmm_generated_code*                       io_generated_code,
                                                 libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                 const libxsmm_matcopy_kernel_config*          i_kernel_config,
                                                 const unsigned int                            i_gp_reg_n_loop,
                                                 const unsigned int                            i_n ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_cmp_instruction, i_gp_reg_n_loop, i_n );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_oi_loop( libxsmm_generated_code*                   io_generated_code,
                                                   libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                   const unsigned int                        i_gp_reg_oi_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_oi_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_oi_loop, 1);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_oi_loop( libxsmm_generated_code*                       io_generated_code,
                                                   libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                   const unsigned int                            i_gp_reg_oi_loop,
                                                   const unsigned int                            i_oi ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_oi_loop, i_oi );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_oj_loop( libxsmm_generated_code*                   io_generated_code,
                                                   libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                   const unsigned int                        i_gp_reg_oj_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_oj_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_oj_loop, 1);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_oj_loop( libxsmm_generated_code*                       io_generated_code,
                                                   libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                   const unsigned int                            i_gp_reg_oj_loop,
                                                   const unsigned int                            i_oj ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_oj_loop, i_oj );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_ofw_loop( libxsmm_generated_code*                   io_generated_code,
                                                   libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                   const unsigned int                        i_gp_reg_oi_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_oi_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_oi_loop, 1);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_ofw_loop( libxsmm_generated_code*                       io_generated_code,
                                                   libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                   const unsigned int                            i_gp_reg_oi_loop,
                                                   const unsigned int                            i_ofw ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_oi_loop, i_ofw );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_ofh_loop( libxsmm_generated_code*                   io_generated_code,
                                                   libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                   const unsigned int                        i_gp_reg_ofh_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_ofh_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_ofh_loop, 1);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_ofh_loop( libxsmm_generated_code*                       io_generated_code,
                                                   libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                   const unsigned int                            i_gp_reg_ofh_loop,
                                                   const unsigned int                            i_ofh ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_ofh_loop, i_ofh );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_kh_loop( libxsmm_generated_code*                   io_generated_code,
                                                   libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                   const unsigned int                        i_gp_reg_kh_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_kh_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_kh_loop, 1);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_kh_loop( libxsmm_generated_code*                       io_generated_code,
                                                   libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                   const unsigned int                            i_gp_reg_kh_loop,
                                                   const unsigned int                            i_kh ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_kh_loop, i_kh );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_kw_loop( libxsmm_generated_code*                   io_generated_code,
                                                   libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                   const unsigned int                        i_gp_reg_kw_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_kw_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_kw_loop, 1);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_kw_loop( libxsmm_generated_code*                       io_generated_code,
                                                   libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                   const unsigned int                            i_gp_reg_kw_loop,
                                                   const unsigned int                            i_kw ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_kw_loop, i_kw );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_ifmOuter_loop( libxsmm_generated_code*                   io_generated_code,
                                                         libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                         const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                         const unsigned int                        i_gp_reg_ifmOuter_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_ifmOuter_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_ifmOuter_loop, 1);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_ifmOuter_loop( libxsmm_generated_code*                       io_generated_code,
                                                         libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                         const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                         const unsigned int                            i_gp_reg_ifmOuter_loop,
                                                         const unsigned int                            i_ifmOuter_blocking ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_ifmOuter_loop, i_ifmOuter_blocking );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}


LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_h_block_loop( libxsmm_generated_code*                   io_generated_code,
                                                         libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                         const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                         const unsigned int                        i_gp_reg_h_block_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_h_block_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_h_block_loop, 1);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_h_block_loop( libxsmm_generated_code*                       io_generated_code,
                                                         libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                         const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                         const unsigned int                            i_gp_reg_h_block_loop,
                                                         const unsigned int                            i_h_blocking ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_h_block_loop, i_h_blocking );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_image_block_loop( libxsmm_generated_code*                   io_generated_code,
                                                         libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                         const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                         const unsigned int                        i_gp_reg_img_block_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_img_block_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_img_block_loop, 1);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_image_block_loop( libxsmm_generated_code*                       io_generated_code,
                                                         libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                         const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                         const unsigned int                            i_gp_reg_img_block_loop,
                                                         const unsigned int                            i_img_blocking ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_img_block_loop, i_img_blocking );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}


LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_ifm_loop( libxsmm_generated_code*                   io_generated_code,
                                                    libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                    const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                    const unsigned int                        i_gp_reg_ifmInner_loop,
                                                    const unsigned int                        i_unrolled_trips ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_ifmInner_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_ifmInner_loop, i_unrolled_trips);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_ifm_loop( libxsmm_generated_code*                   io_generated_code,
                                                    libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                    const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                    const unsigned int                        i_gp_reg_ifmInner_loop,
                                                    const unsigned int                        i_trip_count ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_ifmInner_loop, i_trip_count );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_ofm_loop( libxsmm_generated_code*                   io_generated_code,
                                                    libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                    const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                    const unsigned int                        i_gp_reg_ofmInner_loop,
                                                    const unsigned int                        i_unrolled_trips ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_ofmInner_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_ofmInner_loop, i_unrolled_trips);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_ofm_loop( libxsmm_generated_code*                   io_generated_code,
                                                    libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                    const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                    const unsigned int                        i_gp_reg_ofmInner_loop,
                                                    const unsigned int                        i_trip_count ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_ofmInner_loop, i_trip_count );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_forward_load_output( libxsmm_generated_code*                           io_generated_code,
                                                        const libxsmm_convolution_forward_gp_reg_mapping* i_gp_reg_mapping,
                                                        const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                        const libxsmm_convolution_forward_descriptor*     i_conv_desc ) {
  /* determine the number of registers needed for an ofm block */
  const unsigned int l_reg_per_block = i_conv_desc->ofm_block / i_conv_kernel_config->vector_length_out;
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofh_rb * i_conv_desc->ofw_rb * l_reg_per_block);
  /* register blocking counter  */
  unsigned int l_i, l_j, l_k, l_accs;
  /* block-feature map offset, leading dimension */
  unsigned int l_lead_dim = 0;

  int use_lp_kernel;
  if (i_conv_desc->datatype_itm != i_conv_desc->datatype) {
    use_lp_kernel = 1;
  } else {
    use_lp_kernel = 0;
  }

#if !defined(NDEBUG)
  /* Do some test if it's possible to generated the requested code.
     This is not done in release mode and therefore bad
     things might happen.... HUAAH */
#endif /*NDEBUG*/
  if ( i_conv_desc->ofh_rb*i_conv_desc->ofw_rb*l_reg_per_block > i_conv_kernel_config->vector_reg_count-4 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_INVALID_CONV_ACC );
    return;
  }
  if ( i_conv_desc->ofm_block % i_conv_kernel_config->vector_length_out != 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CONV_OFM_VEC );
    return;
  }

  /* calculate leading dimension depending on format */
  if ( (i_conv_desc->format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC) > 0 ) {
    l_lead_dim = i_conv_desc->ofm_block * i_conv_desc->blocks_ofm;
  } else if ( (i_conv_desc->format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0 ) {
    l_lead_dim = i_conv_desc->ofm_block;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_CONV_FORMAT );
    return;
  }

  if ( ((i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_KNM && i_conv_desc->ofw_rb <= 14 && i_conv_desc->ofh_rb == 1 && l_reg_per_block == 1) || (i_conv_desc->ofw_rb < 12 && i_conv_desc->ofh_rb == 1 && l_reg_per_block == 1)) && (i_conv_kernel_config->instruction_set != LIBXSMM_X86_AVX2) ) {
    /* determining the number of accumulators */
    l_accs = (i_conv_desc->ofw_rb < 9) ? 3 : 2;

    /* adding to C, so let's load C and init additional accumulators */
    for ( l_j = 0; l_j < i_conv_desc->ofw_rb; l_j++ ) {
      for ( l_i = l_accs; l_i > 1; l_i-- ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_conv_kernel_config->instruction_set,
                                                 i_conv_kernel_config->vxor_instruction,
                                                 i_conv_kernel_config->vector_name,
                                                 i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*l_i) + l_j,
                                                 i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*l_i) + l_j,
                                                 i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*l_i) + l_j );
      }
      if ( i_conv_desc->use_nts == 0 ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_conv_kernel_config->instruction_set,
                                          i_conv_kernel_config->vmove_instruction,
                                          i_gp_reg_mapping->gp_reg_output,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          l_j * l_lead_dim * i_conv_kernel_config->datatype_size_out,
                                          i_conv_kernel_config->vector_name,
                                          i_conv_kernel_config->vector_reg_count - i_conv_desc->ofw_rb + l_j , 0, 0 );
#if 1
        if ( (i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1 ) {
           libxsmm_x86_instruction_prefetch( io_generated_code,
                                             LIBXSMM_X86_INSTR_PREFETCHT0 /*i_conv_kernel_config->prefetch_instruction*/,
                                             i_gp_reg_mapping->gp_reg_output_pf,
                                             LIBXSMM_X86_GP_REG_UNDEF, 0,
                                             l_j * l_lead_dim * i_conv_kernel_config->datatype_size_out );
        }
#endif
      } else {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_conv_kernel_config->instruction_set,
                                                 i_conv_kernel_config->vxor_instruction,
                                                 i_conv_kernel_config->vector_name,
                                                 i_conv_kernel_config->vector_reg_count - i_conv_desc->ofw_rb + l_j,
                                                 i_conv_kernel_config->vector_reg_count - i_conv_desc->ofw_rb + l_j,
                                                 i_conv_kernel_config->vector_reg_count - i_conv_desc->ofw_rb + l_j );
      }
    }
  } else {
    /* adding to C, so let's load C */
    for ( l_i = 0; l_i < i_conv_desc->ofh_rb; l_i++ ) {
      for ( l_j = 0; l_j < i_conv_desc->ofw_rb; l_j++ ) {
        for ( l_k = 0; l_k < l_reg_per_block; l_k++ ) {

          if ((i_conv_desc->use_nts == 0) && (use_lp_kernel == 0)) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_conv_kernel_config->instruction_set,
                                              i_conv_kernel_config->vmove_instruction,
                                              i_gp_reg_mapping->gp_reg_output,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              ( l_i * i_conv_desc->ofw_padded * i_conv_desc->stride_h_store * l_lead_dim * i_conv_kernel_config->datatype_size_out) +
                                              ( l_j * l_lead_dim * i_conv_desc->stride_w_store * i_conv_kernel_config->datatype_size_out ) +
                                              ( l_k * i_conv_kernel_config->vector_length_out * i_conv_kernel_config->datatype_size_out ),
                                              i_conv_kernel_config->vector_name,
                                              l_vec_reg_acc_start + l_k + (l_j * l_reg_per_block) + (i_conv_desc->ofw_rb * l_reg_per_block * l_i), 0, 0 );
#if 0
            if ( (i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1 ) {
              libxsmm_x86_instruction_prefetch( io_generated_code,
                                                LIBXSMM_X86_INSTR_PREFETCHT1 /*i_conv_kernel_config->prefetch_instruction*/,
                                                i_gp_reg_mapping->gp_reg_output_pf,
                                                LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                ( l_i * i_conv_desc->ofw_padded *  i_conv_desc->stride_h_store * l_lead_dim * i_conv_kernel_config->datatype_size_out) +
                                                ( l_j * l_lead_dim * i_conv_desc->stride_w_store *  i_conv_kernel_config->datatype_size_out ) +
                                                ( l_k * i_conv_kernel_config->vector_length_out * i_conv_kernel_config->datatype_size_out ) );
            }
#endif
          } else {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                     i_conv_kernel_config->instruction_set,
                                                     i_conv_kernel_config->vxor_instruction,
                                                     i_conv_kernel_config->vector_name,
                                                     l_vec_reg_acc_start + l_k + (l_j * l_reg_per_block) + (i_conv_desc->ofw_rb * l_reg_per_block * l_i),
                                                     l_vec_reg_acc_start + l_k + (l_j * l_reg_per_block) + (i_conv_desc->ofw_rb * l_reg_per_block * l_i),
                                                     l_vec_reg_acc_start + l_k + (l_j * l_reg_per_block) + (i_conv_desc->ofw_rb * l_reg_per_block * l_i) );
          }
        }
      }
    }
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_forward_store_output( libxsmm_generated_code*                           io_generated_code,
                                                         const libxsmm_convolution_forward_gp_reg_mapping* i_gp_reg_mapping,
                                                         const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                         const libxsmm_convolution_forward_descriptor*     i_conv_desc ) {
  /* determine the number of registers needed for an ofm block */
  const unsigned int l_reg_per_block = i_conv_desc->ofm_block / i_conv_kernel_config->vector_length_out;
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofh_rb * i_conv_desc->ofw_rb * l_reg_per_block);
  /* register blocking counter  */
  unsigned int l_i, l_j, l_k, l_accs, i, j;
  /* block-feature map offset, leading dimension */
  unsigned int l_lead_dim = 0;
  /* store instruction to use */
  unsigned int l_intr_store = i_conv_kernel_config->vmove_instruction;

  int use_lp_kernel;
  if (i_conv_desc->datatype_itm != i_conv_desc->datatype) {
    use_lp_kernel = 1;
  } else {
    use_lp_kernel = 0;
  }

#if !defined(NDEBUG)
  /* Do some test if it's possible to generated the requested code.
     This is not done in release mode and therefore bad
     things might happen.... HUAAH */
#endif /*NDEBUG*/
  if ( i_conv_desc->ofh_rb*i_conv_desc->ofw_rb*l_reg_per_block > i_conv_kernel_config->vector_reg_count-4 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_INVALID_CONV_ACC );
    return;
  }
  if ( i_conv_desc->ofm_block % i_conv_kernel_config->vector_length_out != 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CONV_OFM_VEC );
    return;
  }

  /* calculate leading dimension depending on format */
  if ( (i_conv_desc->format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC) > 0 ) {
    l_lead_dim = i_conv_desc->ofm_block * i_conv_desc->blocks_ofm;
  } else if ( (i_conv_desc->format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0 ) {
    l_lead_dim = i_conv_desc->ofm_block;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_CONV_FORMAT );
    return;
  }

  /* if requested let's switch to streaming stores */
  if ( i_conv_desc->use_nts == 1 ) {
    l_intr_store = LIBXSMM_X86_INSTR_VMOVNTPS;
  } 

  if ( (i_conv_desc->compute_batch_stats > 0) /*&& (i_conv_desc->ifm_block != 3)*/ ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, i_gp_reg_mapping->gp_reg_help_3);
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_conv_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_help_3,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        48,
        i_gp_reg_mapping->gp_reg_help_2,
        0 );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_conv_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_help_3,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        56,
        i_gp_reg_mapping->gp_reg_help_3,
        0 );


    libxsmm_x86_instruction_prefetch( io_generated_code,
        LIBXSMM_X86_INSTR_PREFETCHT0, 
        i_gp_reg_mapping->gp_reg_help_2,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        0);

    libxsmm_x86_instruction_prefetch( io_generated_code,
        LIBXSMM_X86_INSTR_PREFETCHT0, 
        i_gp_reg_mapping->gp_reg_help_3,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        0);      

#ifdef FP64_BN_STATS
    libxsmm_x86_instruction_prefetch( io_generated_code,
        LIBXSMM_X86_INSTR_PREFETCHT0, 
        i_gp_reg_mapping->gp_reg_help_2,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        64);  

    libxsmm_x86_instruction_prefetch( io_generated_code,
        LIBXSMM_X86_INSTR_PREFETCHT0, 
        i_gp_reg_mapping->gp_reg_help_3,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        64);     
#endif
  }

  if ( ((i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_KNM && i_conv_desc->ofw_rb <= 14 && i_conv_desc->ofh_rb == 1 && l_reg_per_block == 1) || (i_conv_desc->ofw_rb < 12 && i_conv_desc->ofh_rb == 1 && l_reg_per_block == 1)) && (i_conv_kernel_config->instruction_set != LIBXSMM_X86_AVX2) ) {
    /* determining the number of accumulators */
    l_accs = (i_conv_desc->ofw_rb < 9) ? 3 : 2;

    /* adding up accumulators, adding different order to avoid stalls to some extent.... */
    for ( l_i = l_accs; l_i > 1; l_i-- ) {
      for ( l_j = 0; l_j < i_conv_desc->ofw_rb; l_j++ ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vadd_instruction,
            i_conv_kernel_config->vector_name,
            i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*l_i) + l_j,
            i_conv_kernel_config->vector_reg_count - i_conv_desc->ofw_rb + l_j,
            i_conv_kernel_config->vector_reg_count - i_conv_desc->ofw_rb + l_j );
      }
    }

    for ( l_j = 0; l_j < i_conv_desc->ofw_rb; l_j++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_conv_kernel_config->instruction_set,
          l_intr_store,
          i_gp_reg_mapping->gp_reg_output,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          l_j * l_lead_dim * i_conv_kernel_config->datatype_size_out,
          i_conv_kernel_config->vector_name,
          i_conv_kernel_config->vector_reg_count - i_conv_desc->ofw_rb + l_j, 0, 1 );
    }
  } else {
    /* adding to C, so let's store C */
    if ( (i_conv_desc->use_fwd_generator_for_bwd == 0) || (i_conv_desc->stride_w_store == 1 && i_conv_desc->stride_h_store == 1) ) {
      /* In case of LP kernel convert the kernels to F32  */
      if (use_lp_kernel == 1) {
        unsigned int regX, mem_offset;
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, i_gp_reg_mapping->gp_reg_help_5);
        unsigned int rsp_offset;
        /* Scale factor offset in rsp */
        rsp_offset = 48;
        if (i_conv_desc->compute_batch_stats == 1) {
          rsp_offset = 64;
        }
        if (i_conv_desc->perform_relu_in_kernel == 1) {
          rsp_offset = 56;
        }
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_conv_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_help_5,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            rsp_offset,
            i_gp_reg_mapping->gp_reg_help_4,
            0 );
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VBROADCASTSS,
            i_gp_reg_mapping->gp_reg_help_4,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_conv_kernel_config->vector_name, 0,
            0, 0 );

        if (i_conv_desc->compute_max == 1) {
          /* Load  address of "max_vals" -- max vals is always next to scale factor in RSP, thus +8 in RSP offset */
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_conv_kernel_config->alu_mov_instruction,
              i_gp_reg_mapping->gp_reg_help_5,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              rsp_offset+8,
              i_gp_reg_mapping->gp_reg_help_4,
              0 );

          if (i_conv_desc->perform_relu_in_kernel == 1) {
            /* Load in reg_help_2 the offset for the "input" to determine RELU  */
            libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, i_gp_reg_mapping->gp_reg_help_0);
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                i_conv_kernel_config->alu_mov_instruction,
                i_gp_reg_mapping->gp_reg_help_0,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                48,
                i_gp_reg_mapping->gp_reg_help_2,
                0 ); 
          }

          /* Initialize zmm1 with zeros */
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vxor_instruction,
              i_conv_kernel_config->vector_name, 1, 1, 1);

          /* Initialize "zmm mask" in zmm2 */
          int mask_array[64];
          int ind_mask;
          for (ind_mask = 0; ind_mask < 16; ind_mask++) {
            mask_array[ind_mask] = 0x7FFFFFFF;
          }         

          libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
              (const unsigned char*) mask_array,
              "abs_mask",
              i_conv_kernel_config->vector_name,
              2);
        }

        for ( l_i = 0; l_i < i_conv_desc->ofh_rb; l_i++ ) {
          for ( l_j = 0; l_j < i_conv_desc->ofw_rb; l_j++ ) {
            for ( l_k = 0; l_k < l_reg_per_block; l_k++ ) {
              regX = l_vec_reg_acc_start + l_k + (l_j * l_reg_per_block) + (i_conv_desc->ofw_rb * l_reg_per_block * l_i);
              mem_offset =  (l_k + (l_j * l_reg_per_block) + (i_conv_desc->ofw_padded * l_reg_per_block * l_i)) *  l_lead_dim * i_conv_kernel_config->datatype_size_out;
              /* Convert result to F32  */
              libxsmm_x86_instruction_vec_compute_reg(  io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VCVTDQ2PS,
                  i_conv_kernel_config->vector_name,
                  regX,
                  regX,
                  LIBXSMM_X86_VEC_REG_UNDEF);

              if ( i_conv_desc->use_nts == 0 ) {                  
                /* Fused multiply add  */
                libxsmm_x86_instruction_vec_compute_mem(  io_generated_code,
                    i_conv_kernel_config->instruction_set,
                    LIBXSMM_X86_INSTR_VFMADD213PS,
                    0,
                    i_gp_reg_mapping->gp_reg_output,
                    LIBXSMM_X86_GP_REG_UNDEF,
                    0,
                    mem_offset,
                    i_conv_kernel_config->vector_name, 
                    0, 
                    regX);
              } else {
                libxsmm_x86_instruction_vec_compute_reg(  io_generated_code,
                    i_conv_kernel_config->instruction_set,
                    LIBXSMM_X86_INSTR_VMULPS,
                    i_conv_kernel_config->vector_name,
                    regX,
                    0,
                    regX);

                if ( (i_conv_desc->compute_max == 1) && (i_conv_desc->perform_relu_in_kernel == 0)) {
                  /* Compute ABS(regX) in zmm3  */
                  libxsmm_x86_instruction_vec_compute_reg(  io_generated_code,
                      i_conv_kernel_config->instruction_set,
                      LIBXSMM_X86_INSTR_VPANDD,
                      i_conv_kernel_config->vector_name,
                      regX,
                      2,
                      3);

                  libxsmm_x86_instruction_vec_compute_reg(  io_generated_code,
                      i_conv_kernel_config->instruction_set,
                      LIBXSMM_X86_INSTR_VMAXPS,
                      i_conv_kernel_config->vector_name,
                      1,
                      3,
                      1);
                }
              }

              /* Store the result to output  */
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  l_intr_store,
                  i_gp_reg_mapping->gp_reg_output,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  mem_offset,
                  i_conv_kernel_config->vector_name,
                  regX, 0, 1 );  

            }
          }
        }  

        if ( (i_conv_desc->compute_max == 1) && (i_conv_desc->perform_relu_in_kernel == 0)) {
          /* Store "max" register (zmm1) to max_vals address  */
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vmove_instruction,
              i_gp_reg_mapping->gp_reg_help_4,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              0,
              i_conv_kernel_config->vector_name,
              2, 0, 0 );

          libxsmm_x86_instruction_vec_compute_reg(  io_generated_code,
              i_conv_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMAXPS,
              i_conv_kernel_config->vector_name,
              2,
              1,
              1);

          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vmove_instruction,
              i_gp_reg_mapping->gp_reg_help_4,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              0,
              i_conv_kernel_config->vector_name,
              1, 0, 1 );  
        }

      } else {
        for ( l_i = 0; l_i < i_conv_desc->ofh_rb; l_i++ ) {
          for ( l_j = 0; l_j < i_conv_desc->ofw_rb; l_j++ ) {
            for ( l_k = 0; l_k < l_reg_per_block; l_k++ ) {
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  l_intr_store,
                  i_gp_reg_mapping->gp_reg_output,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  ( l_i * i_conv_desc->ofw_padded * l_lead_dim * i_conv_kernel_config->datatype_size_out) +
                  ( l_j * l_lead_dim * i_conv_kernel_config->datatype_size_out ) +
                  ( l_k * i_conv_kernel_config->vector_length_out * i_conv_kernel_config->datatype_size_out ),
                  i_conv_kernel_config->vector_name,
                  l_vec_reg_acc_start + l_k + (l_j * l_reg_per_block) + (i_conv_desc->ofw_rb * l_reg_per_block * l_i), 0, 1 );
            }
          }
        }
      }

      /* check if we need to calculate batch stats */
      if ( i_conv_desc->compute_batch_stats > 0 ) {
        /* reset zmm0 and zmm1 */
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vxor_instruction,
            i_conv_kernel_config->vector_name, 0, 0, 0);
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vxor_instruction,
            i_conv_kernel_config->vector_name, 1, 1, 1);
#ifdef FP64_BN_STATS   
        /* reset zmm2 and zmm3 */
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vxor_instruction,
            i_conv_kernel_config->vector_name, 2, 2, 2);
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vxor_instruction,
            i_conv_kernel_config->vector_name, 3, 3, 3);
#endif
      }

      if (i_conv_desc->perform_relu_in_kernel == 1) {
        /* Do the ReLu stuff here  */
        unsigned int reg_X;
        unsigned int store_offset;

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vxor_instruction,
            i_conv_kernel_config->vector_name, 0, 0, 0);

        if (i_conv_desc->compute_max == 1) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vxor_instruction,
              i_conv_kernel_config->vector_name, 1, 1, 1);      
        }

        for ( l_i = 0; l_i < i_conv_desc->ofh_rb; l_i++ ) {
          for ( l_j = 0; l_j < i_conv_desc->ofw_rb; l_j++ ) {
            for ( l_k = 0; l_k < l_reg_per_block; l_k++ ) {
              reg_X =  l_vec_reg_acc_start + l_k + (l_j * l_reg_per_block) + (i_conv_desc->ofw_rb * l_reg_per_block * l_i);
              store_offset = ( l_i * i_conv_desc->ofw_padded * l_lead_dim * i_conv_kernel_config->datatype_size_out) +
                ( l_j * l_lead_dim * i_conv_kernel_config->datatype_size_out ) +
                ( l_k * i_conv_kernel_config->vector_length_out * i_conv_kernel_config->datatype_size_out );

              /* VCMP  */
              libxsmm_x86_instruction_vec_compute_mem_mask ( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VCMPPS,
                  0,        
                  i_gp_reg_mapping->gp_reg_help_2,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  store_offset,
                  i_conv_kernel_config->vector_name,
                  0,
                  reg_X,
                  0,
                  1); 

              /* BLEND  */
              libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VBLENDMPS,
                  i_conv_kernel_config->vector_name,
                  0,
                  reg_X,
                  reg_X,
                  1); 

              /* STORE */
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  l_intr_store,
                  i_gp_reg_mapping->gp_reg_output,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  store_offset,
                  i_conv_kernel_config->vector_name,
                  reg_X, 0, 1 );

              if (i_conv_desc->compute_max == 1) {
                /* Compute ABS(regX) in zmm3  */
                libxsmm_x86_instruction_vec_compute_reg(  io_generated_code,
                    i_conv_kernel_config->instruction_set,
                    LIBXSMM_X86_INSTR_VPANDD,
                    i_conv_kernel_config->vector_name,
                    reg_X,
                    2,
                    3);

                libxsmm_x86_instruction_vec_compute_reg(  io_generated_code,
                    i_conv_kernel_config->instruction_set,
                    LIBXSMM_X86_INSTR_VMAXPS,
                    i_conv_kernel_config->vector_name,
                    1,
                    3,
                    1);
              }
            }
          }
        }

        if (i_conv_desc->compute_max == 1){ 
          /* Store "max" register (zmm1) to max_vals address  */
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vmove_instruction,
              i_gp_reg_mapping->gp_reg_help_4,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              0,
              i_conv_kernel_config->vector_name,
              2, 0, 0 );

          libxsmm_x86_instruction_vec_compute_reg(  io_generated_code,
              i_conv_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMAXPS,
              i_conv_kernel_config->vector_name,
              2,
              1,
              1);

          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vmove_instruction,
              i_gp_reg_mapping->gp_reg_help_4,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              0,
              i_conv_kernel_config->vector_name,
              1, 0, 1 );  
        }
      } 

      if ( i_conv_desc->compute_batch_stats > 0 ) {
#ifdef FP32_BN_STATS
        for ( l_i = 0; l_i < i_conv_desc->ofh_rb; l_i++ ) {
          for ( l_j = 0; l_j < i_conv_desc->ofw_rb; l_j++ ) {
            for ( l_k = 0; l_k < l_reg_per_block; l_k++ ) {
              /* compute sum of channels */
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  i_conv_kernel_config->vadd_instruction,
                  i_conv_kernel_config->vector_name, 
                  l_vec_reg_acc_start + l_k + (l_j * l_reg_per_block) + (i_conv_desc->ofw_rb * l_reg_per_block * l_i), 0, 0);
              /* compute sum_2 of channels */
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  i_conv_kernel_config->vfma_instruction,
                  i_conv_kernel_config->vector_name, 
                  l_vec_reg_acc_start + l_k + (l_j * l_reg_per_block) + (i_conv_desc->ofw_rb * l_reg_per_block * l_i), 
                  l_vec_reg_acc_start + l_k + (l_j * l_reg_per_block) + (i_conv_desc->ofw_rb * l_reg_per_block * l_i), 1);
            }
          }
        }

#if 0
        if (i_conv_desc->ifm_block == 3) {
          libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, i_gp_reg_mapping->gp_reg_help_0);
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_conv_kernel_config->alu_mov_instruction,
              i_gp_reg_mapping->gp_reg_help_0,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              48,
              i_gp_reg_mapping->gp_reg_help_2,
              0 );
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_conv_kernel_config->alu_mov_instruction,
              i_gp_reg_mapping->gp_reg_help_0,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              56,
              i_gp_reg_mapping->gp_reg_help_3,
              0 );
        }
#endif

        /* Load running sums to registers (zmm2, zmm3) */

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vmove_instruction,
            i_gp_reg_mapping->gp_reg_help_2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_conv_kernel_config->vector_name,
            2, 0, 0);

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vmove_instruction,
            i_gp_reg_mapping->gp_reg_help_3,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_conv_kernel_config->vector_name,
            3, 0, 0);

        /* Add (zmm2,zmm3) to (zmm0,zmm1) */
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vadd_instruction,
            i_conv_kernel_config->vector_name,
            0,
            2,
            0);

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vadd_instruction,
            i_conv_kernel_config->vector_name,
            1,
            3,
            1);


        /* store both registers (zmm0, zmm1) to external buffers */
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vmove_instruction,
            i_gp_reg_mapping->gp_reg_help_2,
            LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
            i_conv_kernel_config->vector_name,
            0, 0, 1 );

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vmove_instruction,
            i_gp_reg_mapping->gp_reg_help_3,
            LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
            i_conv_kernel_config->vector_name,
            1, 0, 1 );
#endif
#ifdef FP64_BN_STATS
        unsigned int X;
        for ( l_i = 0; l_i < i_conv_desc->ofh_rb; l_i++ ) {
          for ( l_j = 0; l_j < i_conv_desc->ofw_rb; l_j++ ) {
            for ( l_k = 0; l_k < l_reg_per_block; l_k++ ) {
              X = l_vec_reg_acc_start + l_k + (l_j * l_reg_per_block) + (i_conv_desc->ofw_rb * l_reg_per_block * l_i);
              if (X == 4) {
                /* sum1 == zmm0  */
                libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                    i_conv_kernel_config->instruction_set,
                    LIBXSMM_X86_INSTR_VCVTPS2PD,
                    i_conv_kernel_config->vector_name,
                    X,
                    0,
                    LIBXSMM_X86_VEC_REG_UNDEF);

                /* sum2_1 == zmm2 = zmm0^2  */
                libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                    i_conv_kernel_config->instruction_set,
                    LIBXSMM_X86_INSTR_VFMADD231PD,
                    i_conv_kernel_config->vector_name, 
                    0, 
                    0, 2);

                libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                    i_conv_kernel_config->instruction_set,
                    LIBXSMM_X86_INSTR_VEXTRACTF64X4,
                    i_conv_kernel_config->vector_name,
                    X,
                    X,
                    LIBXSMM_X86_VEC_REG_UNDEF,
                    1);

                /* sum2 == zmm1  */
                libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                    i_conv_kernel_config->instruction_set,
                    LIBXSMM_X86_INSTR_VCVTPS2PD,
                    i_conv_kernel_config->vector_name,
                    X,
                    1,
                    LIBXSMM_X86_VEC_REG_UNDEF);

                /* sum2_2 == zmm3 = zmm1^2  */
                libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                    i_conv_kernel_config->instruction_set,
                    LIBXSMM_X86_INSTR_VFMADD231PD,
                    i_conv_kernel_config->vector_name, 
                    1, 
                    1, 3);
              } else {
                /* Use zmm4 as tmp register  */
                libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                    i_conv_kernel_config->instruction_set,
                    LIBXSMM_X86_INSTR_VCVTPS2PD,
                    i_conv_kernel_config->vector_name,
                    X,
                    4,
                    LIBXSMM_X86_VEC_REG_UNDEF);


                libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                    i_conv_kernel_config->instruction_set,
                    LIBXSMM_X86_INSTR_VADDPD,
                    i_conv_kernel_config->vector_name,
                    0,
                    4,
                    0);

                libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                    i_conv_kernel_config->instruction_set,
                    LIBXSMM_X86_INSTR_VFMADD231PD,
                    i_conv_kernel_config->vector_name, 
                    4, 
                    4, 2);

                libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                    i_conv_kernel_config->instruction_set,
                    LIBXSMM_X86_INSTR_VEXTRACTF64X4,
                    i_conv_kernel_config->vector_name,
                    X,
                    X,
                    LIBXSMM_X86_VEC_REG_UNDEF,
                    1);

                libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                    i_conv_kernel_config->instruction_set,
                    LIBXSMM_X86_INSTR_VCVTPS2PD,
                    i_conv_kernel_config->vector_name,
                    X,
                    4,
                    LIBXSMM_X86_VEC_REG_UNDEF);


                libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                    i_conv_kernel_config->instruction_set,
                    LIBXSMM_X86_INSTR_VADDPD,
                    i_conv_kernel_config->vector_name,
                    1,
                    4,
                    1);

                libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                    i_conv_kernel_config->instruction_set,
                    LIBXSMM_X86_INSTR_VFMADD231PD,
                    i_conv_kernel_config->vector_name, 
                    4, 
                    4, 3);
              }
            }
          }
        }

#if 0
        if (i_conv_desc->ifm_block == 3) {
          libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, i_gp_reg_mapping->gp_reg_help_0);
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_conv_kernel_config->alu_mov_instruction,
              i_gp_reg_mapping->gp_reg_help_0,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              48,
              i_gp_reg_mapping->gp_reg_help_2,
              0 );
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_conv_kernel_config->alu_mov_instruction,
              i_gp_reg_mapping->gp_reg_help_0,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              56,
              i_gp_reg_mapping->gp_reg_help_3,
              0 );
        }
#endif


        /* Load running sums (zmm4, zmm5) and (zmm6,zmm7) */

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVAPD,
            i_gp_reg_mapping->gp_reg_help_2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_conv_kernel_config->vector_name,
            4, 0, 0);

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVAPD,
            i_gp_reg_mapping->gp_reg_help_2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            64,
            i_conv_kernel_config->vector_name,
            5, 0, 0);


        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVAPD,
            i_gp_reg_mapping->gp_reg_help_3,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_conv_kernel_config->vector_name,
            6, 0, 0);

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVAPD,
            i_gp_reg_mapping->gp_reg_help_3,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            64,
            i_conv_kernel_config->vector_name,
            7, 0, 0);

        /* Add (zmm4, zmm5, zmm6, zmm7) to (zmm0, zmm1, zmm2, zmm3) */
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
            i_conv_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VADDPD,
            i_conv_kernel_config->vector_name,
            0,
            4,
            0);

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
            i_conv_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VADDPD,
            i_conv_kernel_config->vector_name,
            1,
            5,
            1);

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
            i_conv_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VADDPD,
            i_conv_kernel_config->vector_name,
            2,
            6,
            2);

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
            i_conv_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VADDPD,
            i_conv_kernel_config->vector_name,
            3,
            7,
            3);


        /* store registers (zmm0, zmm1, zmm2, zmm3) to external buffers */
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVAPD,
            i_gp_reg_mapping->gp_reg_help_2,
            LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
            i_conv_kernel_config->vector_name,
            0, 0, 1 );

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVAPD,
            i_gp_reg_mapping->gp_reg_help_2,
            LIBXSMM_X86_GP_REG_UNDEF, 0, 64,
            i_conv_kernel_config->vector_name,
            1, 0, 1 );

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVAPD,
            i_gp_reg_mapping->gp_reg_help_3,
            LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
            i_conv_kernel_config->vector_name,
            2, 0, 1 );

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVAPD,
            i_gp_reg_mapping->gp_reg_help_3,
            LIBXSMM_X86_GP_REG_UNDEF, 0, 64,
            i_conv_kernel_config->vector_name,
            3, 0, 1 );
#endif
      }
    } else { /* "Store" branch for backward with stride */
      unsigned int n_zero_regs = i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofh_rb * i_conv_desc->ofw_rb);
      unsigned int reg_to_use = 0;
      unsigned int index_zero;

      unsigned int reg_X;
      unsigned int store_offset;

      if ( use_lp_kernel == 1 ) {
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, i_gp_reg_mapping->gp_reg_help_5);
        unsigned int rsp_offset;
        /* Scale factor offset in rsp */
        rsp_offset = 48;
        if (i_conv_desc->compute_batch_stats == 1) {
          rsp_offset = 64;
        }
        if (i_conv_desc->perform_relu_in_kernel == 1) {
          rsp_offset = 56;
        }

        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_conv_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_help_5,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            rsp_offset,
            i_gp_reg_mapping->gp_reg_help_4,
            0 );
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VBROADCASTSS,
            i_gp_reg_mapping->gp_reg_help_4,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_conv_kernel_config->vector_name, 0,
            0, 0 );

        if (i_conv_desc->compute_max == 1) {
          /* Load  address of "max_vals" -- max vals is always next to scale factor in RSP, thus +8 in RSP offset */
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_conv_kernel_config->alu_mov_instruction,
              i_gp_reg_mapping->gp_reg_help_5,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              rsp_offset+8,
              i_gp_reg_mapping->gp_reg_help_4,
              0 );

          if (i_conv_desc->perform_relu_in_kernel == 1) {
            /* Load in reg_help_2 the offset for the "input" to determine RELU  */
            libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, i_gp_reg_mapping->gp_reg_help_0);
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                i_conv_kernel_config->alu_mov_instruction,
                i_gp_reg_mapping->gp_reg_help_0,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                48,
                i_gp_reg_mapping->gp_reg_help_2,
                0 ); 
          }

          /* Initialize zmm1 with zeros */
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vxor_instruction,
              i_conv_kernel_config->vector_name, 1, 1, 1);

          /* Initialize "zmm mask" in zmm2 */
          int mask_array[64];
          int ind_mask;
          for (ind_mask = 0; ind_mask < 16; ind_mask++) {
            mask_array[ind_mask] = 0x7FFFFFFF;
          }         

          libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
              (const unsigned char*) mask_array,
              "abs_mask",
              i_conv_kernel_config->vector_name,
              2);
        }

        for (i = 0; i < i_conv_desc->ofh_rb; i++) {
          for ( j = 0; j < i_conv_desc->ofw_rb; j++ ) {
            reg_X =  l_vec_reg_acc_start + (i * i_conv_desc->ofw_rb) + j;
            store_offset = ((i * i_conv_desc->stride_h_store) * i_conv_desc->ofw_padded + j * i_conv_desc->stride_w_store) * i_conv_kernel_config->vector_length_out * i_conv_kernel_config->datatype_size_out;
            /* Convert result to F32  */
            libxsmm_x86_instruction_vec_compute_reg(  io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VCVTDQ2PS,
                i_conv_kernel_config->vector_name,
                reg_X,
                reg_X,
                LIBXSMM_X86_VEC_REG_UNDEF);

            if ( i_conv_desc->use_nts == 0 ) {                  
              /* Fused multiply add  */
              libxsmm_x86_instruction_vec_compute_mem(  io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VFMADD213PS,
                  0,
                  i_gp_reg_mapping->gp_reg_output,
                  LIBXSMM_X86_GP_REG_UNDEF,
                  0,
                  store_offset,
                  i_conv_kernel_config->vector_name, 
                  0, 
                  reg_X);
            } else {
              libxsmm_x86_instruction_vec_compute_reg(  io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VMULPS,
                  i_conv_kernel_config->vector_name,
                  reg_X,
                  0,
                  reg_X);               
            }             
          }
        }
      }

      /* Zero registers for NTS  */
#if 0
      if ( i_conv_desc->use_nts == 1 ) {
        for ( i = 0; i < n_zero_regs; i++ ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vxor_instruction,
              i_conv_kernel_config->vector_name,
              i,
              i,
              i);
        }
      }
#endif
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
          i_conv_kernel_config->instruction_set,
          i_conv_kernel_config->vxor_instruction,
          i_conv_kernel_config->vector_name,
          0,
          0,
          0);

      if (i_conv_desc->compute_max == 1) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vxor_instruction,
            i_conv_kernel_config->vector_name,
            1,
            1,
            1);
      }

      for (i = 0; i < i_conv_desc->ofh_rb; i++) {
        for ( j = 0; j < i_conv_desc->ofw_rb; j++ ) {
          reg_X =  l_vec_reg_acc_start + (i * i_conv_desc->ofw_rb) + j;
          store_offset = ((i * i_conv_desc->stride_h_store) * i_conv_desc->ofw_padded + j * i_conv_desc->stride_w_store) * i_conv_kernel_config->vector_length_out * i_conv_kernel_config->datatype_size_out;
          if (i_conv_desc->perform_relu_in_kernel == 0) {          
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_conv_kernel_config->instruction_set,
                l_intr_store,
                i_gp_reg_mapping->gp_reg_output,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                store_offset,
                i_conv_kernel_config->vector_name,
                reg_X, 0, 1 );
          } else {
            /* VCMP  */
            libxsmm_x86_instruction_vec_compute_mem_mask ( io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VCMPPS,
                0,        
                i_gp_reg_mapping->gp_reg_help_2,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                store_offset,
                i_conv_kernel_config->vector_name,
                0,
                reg_X,
                0,
                1); 

            /* BLEND  */
            libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VBLENDMPS,
                i_conv_kernel_config->vector_name,
                0,
                reg_X,
                reg_X,
                1);    

            /* STORE */
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_conv_kernel_config->instruction_set,
                l_intr_store,
                i_gp_reg_mapping->gp_reg_output,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                store_offset,
                i_conv_kernel_config->vector_name,
                reg_X, 0, 1 );
          }

          if (i_conv_desc->compute_max == 1) {
            /* Compute ABS(regX) in zmm3  */
            libxsmm_x86_instruction_vec_compute_reg(  io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPANDD,
                i_conv_kernel_config->vector_name,
                reg_X,
                2,
                3);

            libxsmm_x86_instruction_vec_compute_reg(  io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMAXPS,
                i_conv_kernel_config->vector_name,
                1,
                3,
                1);
          }

          if ( i_conv_desc->use_nts == 1 ) {
            /* Zero out the skipped "W" pixels for the H index we do write */
            for (index_zero = 1; index_zero < i_conv_desc->stride_w_store; index_zero++) {
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  l_intr_store,
                  i_gp_reg_mapping->gp_reg_output,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  ((i * i_conv_desc->stride_h_store) * i_conv_desc->ofw_padded + j * i_conv_desc->stride_w_store + index_zero) * i_conv_kernel_config->vector_length_out * i_conv_kernel_config->datatype_size_out,
                  i_conv_kernel_config->vector_name,
                  0 , 0, 1 );
              reg_to_use++;
            } 
          }
        } /* end of for ofw_rb loop */      
      } /* end of for ofh_rb loop */

      if ( i_conv_desc->use_nts == 1 ) {
        /* Zero out the skipped "H" rows of pixels  */
        for (i = 0; i < i_conv_desc->ofh_rb; i++) {
          for (index_zero = 1; index_zero < i_conv_desc->stride_h_store; index_zero++) {
            for (j = 0; j < i_conv_desc->ofw_rb * i_conv_desc->stride_w_store; j++) {
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  l_intr_store,
                  i_gp_reg_mapping->gp_reg_output,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  ((i * i_conv_desc->stride_h_store + index_zero) * i_conv_desc->ofw_padded + j) * i_conv_kernel_config->vector_length_out * i_conv_kernel_config->datatype_size_out,
                  i_conv_kernel_config->vector_name,
                  0 , 0, 1 );
              reg_to_use++;
            }
          }
        }
      }

      /* Update max...*/
      if (i_conv_desc->compute_max == 1){ 
        /* Store "max" register (zmm1) to max_vals address  */
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vmove_instruction,
            i_gp_reg_mapping->gp_reg_help_4,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_conv_kernel_config->vector_name,
            2, 0, 0 );

        libxsmm_x86_instruction_vec_compute_reg(  io_generated_code,
            i_conv_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMAXPS,
            i_conv_kernel_config->vector_name,
            2,
            1,
            1);

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vmove_instruction,
            i_gp_reg_mapping->gp_reg_help_4,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_conv_kernel_config->vector_name,
            1, 0, 1 );  
      }

    }
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_reset_x86_convolution_forward_gp_reg_mapping( libxsmm_convolution_forward_gp_reg_mapping* io_gp_reg_mapping ) {
  io_gp_reg_mapping->gp_reg_input = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_weight = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_output = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_input_pf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_weight_pf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_output_pf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_kw_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_kh_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_ifmInner_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_0 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_1 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_2 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_3 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_4 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_5 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_6 = LIBXSMM_X86_GP_REG_UNDEF;
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_reset_x86_convolution_backward_gp_reg_mapping( libxsmm_convolution_backward_gp_reg_mapping* io_gp_reg_mapping ) {
  io_gp_reg_mapping->gp_reg_input = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_weight = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_output = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_input_pf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_weight_pf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_output_pf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_kw_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_oi_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_ofmInner_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_0 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_1 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_2 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_3 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_4 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_5 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_6 = LIBXSMM_X86_GP_REG_UNDEF;
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_reset_x86_convolution_weight_update_gp_reg_mapping( libxsmm_convolution_weight_update_gp_reg_mapping* io_gp_reg_mapping ) {
  io_gp_reg_mapping->gp_reg_input = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_weight = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_output = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_input_pf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_weight_pf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_output_pf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_oj_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_oi_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_ifmInner_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_0 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_1 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_2 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_3 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_4 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_5 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_6 = LIBXSMM_X86_GP_REG_UNDEF;
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_init_convolution_kernel_config( libxsmm_convolution_kernel_config* io_conv_kernel_config ) {
  io_conv_kernel_config->instruction_set = LIBXSMM_X86_INSTR_UNDEF;
  io_conv_kernel_config->vector_reg_count = 0;
  io_conv_kernel_config->vector_length_in = 0;
  io_conv_kernel_config->datatype_size_in = 0;
  io_conv_kernel_config->vector_length_out = 0;
  io_conv_kernel_config->datatype_size_out = 0;
  io_conv_kernel_config->vector_length_wt = 0;
  io_conv_kernel_config->datatype_size_wt = 0;
  io_conv_kernel_config->vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
  io_conv_kernel_config->vfma_instruction = LIBXSMM_X86_INSTR_UNDEF;
  io_conv_kernel_config->prefetch_instruction = LIBXSMM_X86_INSTR_UNDEF;
  io_conv_kernel_config->alu_add_instruction = LIBXSMM_X86_INSTR_UNDEF;
  io_conv_kernel_config->alu_sub_instruction = LIBXSMM_X86_INSTR_UNDEF;
  io_conv_kernel_config->alu_cmp_instruction = LIBXSMM_X86_INSTR_UNDEF;
  io_conv_kernel_config->alu_jmp_instruction = LIBXSMM_X86_INSTR_UNDEF;
  io_conv_kernel_config->alu_mov_instruction = LIBXSMM_X86_INSTR_UNDEF;
  io_conv_kernel_config->vector_name = '\0';
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_fma( libxsmm_generated_code*                           io_generated_code,
    const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
    const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
    const libxsmm_convolution_backward_descriptor*     i_conv_desc ) {
  libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
      i_conv_kernel_config->instruction_set,
      i_conv_kernel_config->vfma_instruction,
      1, /* use broadcast*/
      i_gp_reg_mapping->gp_reg_output,
      LIBXSMM_X86_GP_REG_UNDEF /* for not using SIB addressing */,
      0 /* no scale for no SIB addressing */,
      /* disp */(i_gp_reg_mapping->gp_reg_oi_loop * i_conv_desc->ofm_block + i_gp_reg_mapping->gp_reg_ofmInner_loop) * i_conv_kernel_config->datatype_size_out,
      i_conv_kernel_config->vector_name,
      1 /* weight */,
      4 /* input loaded */);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_load_weight( libxsmm_generated_code*                           io_generated_code,
    const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
    const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
    const libxsmm_convolution_backward_descriptor*     i_conv_desc ) {
  /* add 5 and 6 */
  /* get the gp_reg_input to help_0 */
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_weight, i_gp_reg_mapping->gp_reg_help_1 );

  /* add kw's offset */
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_5, i_gp_reg_mapping->gp_reg_help_1);
  /* add oi's offset */
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_6, i_gp_reg_mapping->gp_reg_help_1);
  libxsmm_x86_instruction_vec_move( io_generated_code,
      i_conv_kernel_config->instruction_set,
      i_conv_kernel_config->vmove_instruction,
      i_gp_reg_mapping->gp_reg_help_1,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      0,
      i_conv_kernel_config->vector_name,
      1 , 0, 0 );
  LIBXSMM_UNUSED(i_conv_desc);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_load_input( libxsmm_generated_code*                           io_generated_code,
    const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
    const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
    const libxsmm_convolution_backward_descriptor*     i_conv_desc) {
  /* determine the number of registers needed for an ifm block */
  const unsigned int l_reg_per_block = i_conv_desc->ofm_block / i_conv_kernel_config->vector_length_in;
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - ( i_conv_desc->ofw_rb * l_reg_per_block);
  /* register blocking counter */
  unsigned int reg_count = 0;
  unsigned int l_j, l_k;
  /* adding to C, so let's load C */
  for ( l_j = 0; l_j < i_conv_desc->ofw_rb; l_j++ ) {
    for ( l_k = 0; l_k < l_reg_per_block; l_k++, reg_count++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_conv_kernel_config->instruction_set,
          i_conv_kernel_config->vmove_instruction,
          i_gp_reg_mapping->gp_reg_input,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (reg_count)* i_conv_kernel_config->vector_length_in * i_conv_kernel_config->datatype_size_in,
          i_conv_kernel_config->vector_name,
          l_vec_reg_acc_start + reg_count , 0, 0 );
    }
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_store_input( libxsmm_generated_code*                           io_generated_code,
    const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
    const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
    const libxsmm_convolution_backward_descriptor*     i_conv_desc ) {

  /* determine the number of registers needed for an ofm block */
  const unsigned int l_reg_per_block = i_conv_desc->ofm_block / i_conv_kernel_config->vector_length_in;
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb * l_reg_per_block);
  /* register blocking counter */
  unsigned int reg_count = 0;
  unsigned int l_j, l_k;
  /* adding to C, so let's load C */
  for ( l_j = 0; l_j < i_conv_desc->ofw_rb; l_j++ ) {
    for ( l_k = 0; l_k < l_reg_per_block; l_k++, reg_count++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_conv_kernel_config->instruction_set,
          i_conv_kernel_config->vmove_instruction,
          i_gp_reg_mapping->gp_reg_input,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (reg_count)  *i_conv_kernel_config->vector_length_in * i_conv_kernel_config->datatype_size_in,
          i_conv_kernel_config->vector_name,
          l_vec_reg_acc_start + reg_count , 0, 1 );
    }
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_weight_update_load_weight( libxsmm_generated_code*                           io_generated_code,
    const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
    const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
    const libxsmm_convolution_weight_update_descriptor*     i_conv_desc) {
  /* determine the number of registers needed for an ofm block */
  /* const unsigned int l_reg_per_block = (i_conv_desc->ifm_block == 1) ? (i_conv_desc->kw) : (i_conv_desc->ifm_block / i_conv_kernel_config->vector_length_wt); */
  const unsigned int l_reg_per_block = (i_conv_desc->ifm_block == 1) ? (i_conv_desc->kw) : (i_conv_desc->ifm_block_hp < i_conv_kernel_config->vector_length_wt ? 1 : i_conv_desc->ifm_block_hp / i_conv_kernel_config->vector_length_wt);
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - ( i_conv_desc->ifm_block_hp * l_reg_per_block);
  /* register blocking counter */
  unsigned int reg_count = 0;

  unsigned int l_j, l_k;
  /* adding to C, so let's load C */
  /* choosing offset according to format */
  /* for filter in custom format it's vector length */
  unsigned int offset = i_conv_kernel_config->vector_length_wt;
  int use_lp_kernel;
  if (i_conv_desc->datatype_itm != i_conv_desc->datatype) {
    use_lp_kernel = 1;
  } else {
    use_lp_kernel = 0;
  }
  /* for filter in custom reduction format it's vector length * ncopies */
  if (i_conv_desc->use_nts == 1) {
    offset *= i_conv_desc->ncopies;
  }

  /* for filter in RSCK format it's active ofm leading dimension */
  if ( (i_conv_desc->format & LIBXSMM_DNN_TENSOR_FORMAT_RSCK) > 0 ) {
    offset = i_conv_kernel_config->l_ld_ofm_act;
    if (i_conv_desc->ifm_block == 1) {
      offset *= i_conv_kernel_config->l_ld_ifm_act;
    }
  }
  /* TODO support reductions in RSCK format */

  int eq_index;
  int unrolled_ofms = ( i_conv_desc->ifm_block != 3 ) ? 1 : 4;

  /* adding to C, so let's load C */
  if ( (i_conv_desc->use_nts == 0) && (use_lp_kernel == 0) ) {
    for ( l_j = 0; l_j < i_conv_desc->ifm_block_hp; l_j++ ) {
      for ( l_k = 0; l_k < l_reg_per_block; l_k++, reg_count++ ) {
        for (eq_index = 0; eq_index < unrolled_ofms; eq_index++) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vmove_instruction,
              i_gp_reg_mapping->gp_reg_weight,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (reg_count)*offset * i_conv_kernel_config->datatype_size_wt + 
              eq_index * i_conv_desc->kw * i_conv_desc->kh *  i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->l_ld_ofm_act *  i_conv_kernel_config->datatype_size_wt ,
              i_conv_kernel_config->vector_name,
              l_vec_reg_acc_start + reg_count - eq_index * 3 , 0, 0);

          if ( (i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1 ) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT1 ,
                i_gp_reg_mapping->gp_reg_weight_pf,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                (reg_count)*offset * i_conv_kernel_config->datatype_size_wt  +
                eq_index * i_conv_desc->kw * i_conv_desc->kh *  i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->l_ld_ofm_act *  i_conv_kernel_config->datatype_size_wt  );
          }
        }
      }
    }
  } else {

    for ( l_j = 0; l_j < i_conv_desc->ifm_block_hp; l_j++ ) {
      for ( l_k = 0; l_k < l_reg_per_block; l_k++, reg_count++ ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vxor_instruction,
            i_conv_kernel_config->vector_name,
            l_vec_reg_acc_start + reg_count,
            l_vec_reg_acc_start + reg_count,
            l_vec_reg_acc_start + reg_count);

        if ( (i_conv_desc->use_nts == 0) && (use_lp_kernel == 1) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1 ,
              i_gp_reg_mapping->gp_reg_weight,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (reg_count)*offset * i_conv_kernel_config->datatype_size_wt);
        }
      }
    }

  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_weight_update_transpose_load_weight( libxsmm_generated_code*                           io_generated_code,
    const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
    const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
    const libxsmm_convolution_weight_update_descriptor*     i_conv_desc) {
  /* determine the number of registers needed for an ofm block */
  const unsigned int l_reg_per_block = (i_conv_desc->ofm_block / i_conv_kernel_config->vector_length_wt);
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - ( i_conv_desc->ofm_block * l_reg_per_block);
  /* register blocking counter */
  unsigned int reg_count = 0;
  unsigned int l_j, l_k;
  /* adding to C, so let's load C */
  for ( l_j = 0; l_j < i_conv_desc->ofm_block; l_j++ ) {
    for ( l_k = 0; l_k < l_reg_per_block; l_k++, reg_count++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_conv_kernel_config->instruction_set,
          i_conv_kernel_config->vmove_instruction,
          i_gp_reg_mapping->gp_reg_weight,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (reg_count)*i_conv_kernel_config->vector_length_wt * i_conv_kernel_config->datatype_size_wt,
          i_conv_kernel_config->vector_name,
          l_vec_reg_acc_start + reg_count , 0, 0 );
      if ( (i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1 ) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
            LIBXSMM_X86_INSTR_PREFETCHT0 ,
            i_gp_reg_mapping->gp_reg_weight_pf,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (reg_count)*i_conv_kernel_config->vector_length_wt *i_conv_kernel_config->datatype_size_wt);
      }
    }
  }
}
LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_weight_update_store_weight( libxsmm_generated_code*                           io_generated_code,
    const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
    const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
    const libxsmm_convolution_weight_update_descriptor*     i_conv_desc) {
  /* determine the number of registers needed for an ofm block */
  const unsigned int l_reg_per_block = (i_conv_desc->ifm_block == 1) ? (i_conv_desc->kw) : (i_conv_desc->ifm_block_hp < i_conv_kernel_config->vector_length_wt ? 1 : i_conv_desc->ifm_block_hp / i_conv_kernel_config->vector_length_wt);
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - ( i_conv_desc->ifm_block_hp * l_reg_per_block);
  /* register blocking counter  */
  unsigned int reg_count = 0;

  unsigned int l_j, l_k;
  /* choosing offset according to format */
  /* for filter in custom format it's vector length */
  unsigned int offset = i_conv_kernel_config->vector_length_wt;
  /* for filter in custom reduction format it's vector length * ncopies */
  if (i_conv_desc->use_nts == 1) {
    offset *= i_conv_desc->ncopies;
  }

  int use_lp_kernel;
  if (i_conv_desc->datatype_itm != i_conv_desc->datatype) {
    use_lp_kernel = 1;
  } else {
    use_lp_kernel = 0;
  }

  /* for filter in RSCK format it's active ofm leading dimension */
  if ( (i_conv_desc->format & LIBXSMM_DNN_TENSOR_FORMAT_RSCK) > 0 ) {
    offset = i_conv_kernel_config->l_ld_ofm_act;
    if (i_conv_desc->ifm_block == 1) {
      offset *= i_conv_kernel_config->l_ld_ifm_act;
    }
  }
  /* TODO support reductions in RSCK format */
  int eq_index;
  int unrolled_ofms = ( i_conv_desc->ifm_block != 3 ) ? 1 : 4;

  if ( use_lp_kernel == 1 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, i_gp_reg_mapping->gp_reg_help_5);
    unsigned int rsp_offset = 48 ;
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_conv_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_help_5,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        rsp_offset,
        i_gp_reg_mapping->gp_reg_help_4,
        0 );
    libxsmm_x86_instruction_vec_move( io_generated_code,
        i_conv_kernel_config->instruction_set,
        LIBXSMM_X86_INSTR_VBROADCASTSS,
        i_gp_reg_mapping->gp_reg_help_4,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        0,
        i_conv_kernel_config->vector_name, 0,
        0, 0 );
  }

  if ( i_conv_desc->use_nts == 0  ) {
    /* adding to C, so let's load C */
    for ( l_j = 0; l_j < i_conv_desc->ifm_block_hp; l_j++ ) {
      for ( l_k = 0; l_k < l_reg_per_block; l_k++, reg_count++ ) {
        for (eq_index = 0; eq_index < unrolled_ofms; eq_index++) {
          if (use_lp_kernel == 0) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_conv_kernel_config->instruction_set,
                i_conv_kernel_config->vmove_instruction,
                i_gp_reg_mapping->gp_reg_weight,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                (reg_count)*offset * i_conv_kernel_config->datatype_size_wt +
                eq_index * i_conv_desc->kw * i_conv_desc->kh *  i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->l_ld_ofm_act *  i_conv_kernel_config->datatype_size_wt,
                i_conv_kernel_config->vector_name,
                l_vec_reg_acc_start + reg_count - eq_index * 3 , 0, 1 );
          } else {
            unsigned int reg_X;
            unsigned int mem_offset;
            reg_X =  l_vec_reg_acc_start + reg_count - eq_index * 3;
            mem_offset = (reg_count)*offset * i_conv_kernel_config->datatype_size_wt + eq_index * i_conv_desc->kw * i_conv_desc->kh *  i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->l_ld_ifm_act *  i_conv_kernel_config->datatype_size_wt;

            libxsmm_x86_instruction_vec_compute_reg(  io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VCVTDQ2PS,
                i_conv_kernel_config->vector_name,
                reg_X,
                reg_X,
                LIBXSMM_X86_VEC_REG_UNDEF);

            libxsmm_x86_instruction_vec_compute_mem(  io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VFMADD213PS,
                0,
                i_gp_reg_mapping->gp_reg_weight,
                LIBXSMM_X86_GP_REG_UNDEF,
                0,
                mem_offset,
                i_conv_kernel_config->vector_name, 
                0, 
                reg_X);

            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_conv_kernel_config->instruction_set,
                i_conv_kernel_config->vmove_instruction,
                i_gp_reg_mapping->gp_reg_weight,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                mem_offset,
                i_conv_kernel_config->vector_name,
                reg_X, 0, 1 );          
          }
        }
      }
    }
  } else {
    for ( l_j = 0; l_j < i_conv_desc->ifm_block_hp; l_j++ ) {
      for ( l_k = 0; l_k < l_reg_per_block; l_k++, reg_count++ ) {
        if (use_lp_kernel == 1) {
          /* Convert result to F32  */
          libxsmm_x86_instruction_vec_compute_reg(  io_generated_code,
              i_conv_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VCVTDQ2PS,
              i_conv_kernel_config->vector_name,
              l_vec_reg_acc_start + reg_count ,
              l_vec_reg_acc_start + reg_count ,
              LIBXSMM_X86_VEC_REG_UNDEF);

          libxsmm_x86_instruction_vec_compute_reg(  io_generated_code,
              i_conv_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMULPS,
              i_conv_kernel_config->vector_name,
              l_vec_reg_acc_start + reg_count ,
              0,
              l_vec_reg_acc_start + reg_count);    
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vmove_instruction,
            i_gp_reg_mapping->gp_reg_weight,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (reg_count)*offset * i_conv_kernel_config->datatype_size_wt,
            i_conv_kernel_config->vector_name,
            l_vec_reg_acc_start + reg_count , 0, 1 );
      }
    }
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_weight_update_transpose_store_weight( libxsmm_generated_code*                           io_generated_code,
    const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
    const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
    const libxsmm_convolution_weight_update_descriptor*     i_conv_desc) {
  /* determine the number of registers needed for an ofm block */
  const unsigned int l_reg_per_block = (i_conv_desc->ofm_block / i_conv_kernel_config->vector_length_wt);
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - ( i_conv_desc->ofm_block * l_reg_per_block);
  /* register blocking counter  */
  unsigned int reg_count = 0;
  unsigned int l_j, l_k;
  /* adding to C, so let's load C */
  for ( l_j = 0; l_j < i_conv_desc->ofm_block; l_j++ ) {
    for ( l_k = 0; l_k < l_reg_per_block; l_k++, reg_count++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_conv_kernel_config->instruction_set,
          i_conv_kernel_config->vmove_instruction,
          i_gp_reg_mapping->gp_reg_weight,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (reg_count)*i_conv_kernel_config->vector_length_wt *i_conv_kernel_config->datatype_size_wt,
          i_conv_kernel_config->vector_name,
          l_vec_reg_acc_start + reg_count , 0, 1 );
    }
  }
}
