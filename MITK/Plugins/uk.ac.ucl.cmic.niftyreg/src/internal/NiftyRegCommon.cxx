/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "NiftyRegCommon.h"

// ---------------------------------------------------------------------------
// mat44_transpose()
// ---------------------------------------------------------------------------

mat44 mat44_transpose(mat44 in)
{
  mat44 out;
  for(unsigned int i = 0; i < 4; i++)
    {
    for(unsigned int j = 0; j < 4; j++)
      {
      out.m[i][j] = in.m[j][i];
      }
    }
  return out;
}




