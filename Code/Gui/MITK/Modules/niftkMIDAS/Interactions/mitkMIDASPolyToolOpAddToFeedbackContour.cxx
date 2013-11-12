/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASPolyToolOpAddToFeedbackContour.h"

namespace mitk {

MIDASPolyToolOpAddToFeedbackContour::MIDASPolyToolOpAddToFeedbackContour(
  mitk::OperationType type,
  mitk::Point3D &point,
  mitk::ContourModel* contour,
  const mitk::PlaneGeometry* planeGeometry
  )
: mitk::Operation(type)
, m_Point(point)
, m_Contour(contour)
, m_PlaneGeometry(planeGeometry)
{

}

} // end namespace
