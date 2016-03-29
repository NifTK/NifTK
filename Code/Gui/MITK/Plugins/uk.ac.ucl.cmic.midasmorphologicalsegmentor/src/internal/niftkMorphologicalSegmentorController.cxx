/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMorphologicalSegmentorController.h"

#include "niftkMorphologicalSegmentorView.h"

//-----------------------------------------------------------------------------
niftkMorphologicalSegmentorController::niftkMorphologicalSegmentorController(niftkMorphologicalSegmentorView* segmentorView)
  : niftkBaseSegmentorController(segmentorView),
    m_MorphologicalSegmentorView(segmentorView)
{
}


//-----------------------------------------------------------------------------
niftkMorphologicalSegmentorController::~niftkMorphologicalSegmentorController()
{
}
