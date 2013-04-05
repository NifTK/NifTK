/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASContourToolOpAccumulateContour.h"

namespace mitk {

MIDASContourToolOpAccumulateContour::MIDASContourToolOpAccumulateContour(
  mitk::OperationType type,
  bool redo,
  int dataSetNumber,
  mitk::ContourSet::Pointer contourSet
  )
: mitk::Operation(type)
, m_Redo(redo)
, m_DataSetNumber(dataSetNumber)
, m_ContourSet(contourSet)
{

}

} // end namespace
