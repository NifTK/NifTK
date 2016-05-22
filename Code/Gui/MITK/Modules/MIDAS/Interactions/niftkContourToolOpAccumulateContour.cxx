/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkContourToolOpAccumulateContour.h"

namespace niftk
{

ContourToolOpAccumulateContour::ContourToolOpAccumulateContour(
  mitk::OperationType type,
  bool redo,
  int dataIndex,
  mitk::ContourModelSet::Pointer contourSet
  )
: mitk::Operation(type)
, m_Redo(redo)
, m_DataIndex(dataIndex)
, m_ContourSet(contourSet)
{
}

ContourToolOpAccumulateContour::~ContourToolOpAccumulateContour()
{
}

bool ContourToolOpAccumulateContour::IsRedo() const
{
  return m_Redo;
}

int ContourToolOpAccumulateContour::GetDataIndex() const
{
  return m_DataIndex;
}

mitk::ContourModelSet::Pointer ContourToolOpAccumulateContour::GetContourSet() const
{
  return m_ContourSet;
}

}
