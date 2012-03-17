/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-09-14 15:39:17 +0100 (Tue, 14 Sep 2010) $
 Revision          : $Revision: 3898 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef CUDA_UTILS_KERNELS_CU
#define CUDA_UTILS_KERNELS_CU

/** In this file, we put pure CUDA kernels. */

/** 
 * Test kernel to test that we are compiling things correctly with nvcc etc. Here we add two floats. 
 */
__global__ void TestAddKernel( float a, float b, float *c ) {
  *c = a + b;
}

#endif CUDA_UTILS_KERNELS_CU