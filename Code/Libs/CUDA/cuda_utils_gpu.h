/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-09-14 15:39:17 +0100 (Tue, 14 Sep 2010) $
 Revision          : $Revision: 4625 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef CUDA_UTILS_GPU_H
#define CUDA_UTILS_GPU_H

/**
 * In this file we declare pure C style functions, that are called from other applications.
 */

/**
 * Test function to add two floats.
 */
extern "C++"
float TestAdd(float a, float b);

#endif CUDA_UTILS_GPU_H
