/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASPolyToolOpUpdateFeedbackContour.h"

namespace mitk {

MIDASPolyToolOpUpdateFeedbackContour::MIDASPolyToolOpUpdateFeedbackContour(
  mitk::OperationType type,
  unsigned int pointId,
  const mitk::Point3D &point,
  mitk::Contour* contour,
  const mitk::PlaneGeometry* planeGeometry
  )
: mitk::Operation(type)
, m_PointId(pointId)
, m_Point(point)
, m_Contour(contour)
, m_PlaneGeometry(planeGeometry)
{

}

} // end namespace
