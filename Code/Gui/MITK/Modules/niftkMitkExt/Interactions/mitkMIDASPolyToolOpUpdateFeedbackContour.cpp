/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-02-16 21:02:48 +0000 (Thu, 16 Feb 2012) $
 Revision          : $Revision: 8525 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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
