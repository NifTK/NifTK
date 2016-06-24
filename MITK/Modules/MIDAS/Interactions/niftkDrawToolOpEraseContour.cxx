/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkDrawToolOpEraseContour.h"

namespace niftk
{

DrawToolOpEraseContour::DrawToolOpEraseContour(
  mitk::OperationType type,
  mitk::ContourModelSet* contourModelSet,
  int dataIndex
  )
: mitk::Operation(type)
, m_ContourModelSet(contourModelSet)
, m_DataIndex(dataIndex)
{
}

DrawToolOpEraseContour::~DrawToolOpEraseContour()
{
}

mitk::ContourModelSet* DrawToolOpEraseContour::GetContourModelSet() const
{
  return m_ContourModelSet;
}

int DrawToolOpEraseContour::GetDataIndex() const
{
  return m_DataIndex;
}

}
