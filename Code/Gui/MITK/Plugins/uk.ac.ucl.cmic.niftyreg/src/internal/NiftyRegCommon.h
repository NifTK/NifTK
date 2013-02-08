/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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

