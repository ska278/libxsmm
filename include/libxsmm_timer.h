/******************************************************************************
** Copyright (c) 2009-2018, Intel Corporation                                **
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
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_TIMER_H
#define LIBXSMM_TIMER_H

#include "libxsmm_macros.h"


typedef unsigned long long libxsmm_timer_tickint;

/**
 * Returns the current clock tick of a monotonic timer source with
 * platform-specific resolution (not necessarily CPU cycles).
 */
LIBXSMM_API libxsmm_timer_tickint libxsmm_timer_tick(void);

/** Helper function to receive the difference between two timer ticks; avoids potential side-effects of LIBXSMM_DIFF. */
LIBXSMM_API_INLINE libxsmm_timer_tickint libxsmm_timer_diff(libxsmm_timer_tickint tick0, libxsmm_timer_tickint tick1) {
  return LIBXSMM_DIFF(tick0, tick1);
}

/** Returns the duration (in seconds) between two values received by libxsmm_timer_tick. */
LIBXSMM_API double libxsmm_timer_duration(libxsmm_timer_tickint tick0, libxsmm_timer_tickint tick1);

#endif /*LIBXSMM_TIMER_H*/
