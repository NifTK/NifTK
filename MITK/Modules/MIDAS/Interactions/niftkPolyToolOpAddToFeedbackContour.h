/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPolyToolOpAddToFeedbackContour_h
#define niftkPolyToolOpAddToFeedbackContour_h

#include "niftkMIDASExports.h"

#include <mitkContourModel.h>
#include <mitkOperation.h>
#include <mitkOperationActor.h>
#include <mitkPlaneGeometry.h>
#include <mitkTool.h>
#include <mitkToolManager.h>

namespace niftk
{

/**
 * \class PolyToolOpAddToFeedbackContour
 * \brief Operation class to hold data to pass back to this PolyTool,
 * so that the PolyTool can execute the Undo/Redo command.
 */
class NIFTKMIDAS_EXPORT PolyToolOpAddToFeedbackContour: public mitk::Operation
{
public:

  PolyToolOpAddToFeedbackContour(
      mitk::OperationType type,
      mitk::Point3D &point,
      mitk::ContourModel* contour,
      const mitk::PlaneGeometry* geometry
      );

  ~PolyToolOpAddToFeedbackContour();

  mitk::Point3D GetPoint() const;

  mitk::ContourModel* GetContour() const;

  const mitk::PlaneGeometry* GetPlaneGeometry();

private:

  mitk::Point3D m_Point;
  mitk::ContourModel::Pointer m_Contour;
  const mitk::PlaneGeometry* m_PlaneGeometry;
};

}

#endif
