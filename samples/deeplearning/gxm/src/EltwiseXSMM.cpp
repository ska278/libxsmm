/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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
/* Sasikanth Avancha, Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/


#include <stdio.h>
#include <omp.h>
#include "EltwiseXSMM.hpp"

#define VLEN 16

void EltwiseXSMM::forwardPropagate(vector<TensorBuf*>& inpb, TensorBuf *outpb, int tid)
{
  float *outp = (float*)outpb->getBuffer();

  float *inp_r = (float*)inpb[0]->getBuffer();
  float *inp_nr = (float*)inpb[1]->getBuffer();

  int nImg = gp->batch_size;
  int nOfm = gp->nOutput;
  int ifh = gp->iHeight;
  int ifw = gp->iWidth;
  int ofh = gp->oHeight;
  int ofw = gp->oWidth;
  int threads = gp->num_threads;

  int op = gp->op;

  __assume_aligned(outp, 64);

  int size = nImg * nOfm * ofh *ofw;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<size; i++)
    outp[i] = 0;

  switch(op)
  {
    case ELSUM:
      {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i=0; i < size; i++)
          outp[i] = inp_r[i] + inp_nr[i];
      }
      break;

    case ELPROD:
      break;

    case ELMAX:
      break;
  }

  outpb->setLayoutType(LIBXSMM_CUSTOM_LAYOUT);
}

void EltwiseXSMM::backPropagate(TensorBuf *deloutpb, vector<TensorBuf*>& delinpb, int tid)
{
  float *deloutp = (float*)deloutpb->getBuffer();

  int op = gp->op;

  switch(op)
  {
    case ELSUM:
    {
      for(int i=0; i<delinpb.size(); i++)
      {
        float *delinp = (float*)delinpb[i]->getBuffer();
        Shape *ss = delinpb[i]->getTensor()->getShape();
        int size = ss->dims[0] * ss->dims[1] * ss->dims[2] * ss->dims[3];

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int j=0; j < size; j++)
          delinp[j] = deloutp[j];
      }
    }
    break;

    case ELPROD:
      break;

    case ELMAX:
      break;
  }

  for(int b=0; b<delinpb.size(); b++)
    delinpb[b]->setLayoutType(LIBXSMM_CUSTOM_LAYOUT);
}
