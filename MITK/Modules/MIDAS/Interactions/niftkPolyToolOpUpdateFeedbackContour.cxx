/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPolyToolOpUpdateFeedbackContour.h"

namespace niftk
{

PolyToolOpUpdateFeedbackContour::PolyToolOpUpdateFeedbackContour(
  mitk::OperationType type,
  unsigned int pointId,
  const mitk::Point3D &point,
  mitk::ContourModel* contour,
  const mitk::PlaneGeometry* planeGeometry
  )
: mitk::Operation(type)
, m_PointId(pointId)
, m_Point(point)
, m_Contour(contour)
, m_PlaneGeometry(planeGeometry)
{
}

PolyToolOpUpdateFeedbackContour::~PolyToolOpUpdateFeedbackContour()
{
}

unsigned int PolyToolOpUpdateFeedbackContour::GetPointId() const
{
  return m_PointId;
}

const mitk::Point3D PolyToolOpUpdateFeedbackContour::GetPoint() const
{
  return m_Point;
}

mitk::ContourModel* PolyToolOpUpdateFeedbackContour::GetContour() const
{
  return m_Contour.GetPointer();
}

const mitk::PlaneGeometry* PolyToolOpUpdateFeedbackContour::GetPlaneGeometry()
{
  return m_PlaneGeometry;
}

}
