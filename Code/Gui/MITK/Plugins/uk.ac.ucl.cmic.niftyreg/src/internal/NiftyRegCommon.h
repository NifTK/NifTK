/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: $
 Last modified by  : $Author:  $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef NiftyRegCommon_h
#define NiftyRegCommon_h

#include "nifti1_io.h"


/// Codes for interpolation type
typedef enum {
  UNSET_INTERPOLATION = 0,
  NEAREST_INTERPOLATION = 1,
  LINEAR_INTERPOLATION = 2,
  CUBIC_INTERPOLATION = 3
} InterpolationType;

/// Codes for similarity measure type
typedef enum {
  UNSET_SIMILARITY = 0,
  NMI_SIMILARITY = 1,
  SSD_SIMILARITY = 2,
  KLDIV_SIMILARITY = 3
} SimilarityType;

/// Codes for affine registration type
typedef enum {
  UNSET_TRANSFORMATION = 0,
  RIGID_ONLY = 1,
  RIGID_THEN_AFFINE = 2,
  DIRECT_AFFINE = 3
} AffineRegistrationType;


mat44 mat44_transpose(mat44 in);


#endif // NiftyRegCommon_h

