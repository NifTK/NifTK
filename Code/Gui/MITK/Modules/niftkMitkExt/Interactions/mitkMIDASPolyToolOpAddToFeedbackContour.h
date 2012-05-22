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

#ifndef MITKMIDASPOLYTOOLOPADDTOFEEDBACKCONTOURCONTOUR_H
#define MITKMIDASPOLYTOOLOPADDTOFEEDBACKCONTOURCONTOUR_H

#include "niftkMitkExtExports.h"
#include "mitkOperation.h"
#include "mitkOperationActor.h"
#include "mitkTool.h"
#include "mitkToolManager.h"
#include "mitkContour.h"
#include "mitkPlaneGeometry.h"

namespace mitk
{

/**
 * \class MIDASPolyToolOpAddToFeedbackContour
 * \brief Operation class to hold data to pass back to this MIDASPolyTool,
 * so that the MIDASPolyTool can execute the Undo/Redo command.
 */
class NIFTKMITKEXT_EXPORT MIDASPolyToolOpAddToFeedbackContour: public mitk::Operation
{
public:

  MIDASPolyToolOpAddToFeedbackContour(
      mitk::OperationType type,
      mitk::Point3D &point,
      mitk::Contour* contour,
      const mitk::PlaneGeometry* geometry
      );
  ~MIDASPolyToolOpAddToFeedbackContour() {};
  mitk::Point3D GetPoint() const { return m_Point;}
  mitk::Contour* GetContour() const { return m_Contour.GetPointer();}
  const mitk::PlaneGeometry* GetPlaneGeometry() { return m_PlaneGeometry; }

private:
  mitk::Point3D m_Point;
  mitk::Contour::Pointer m_Contour;
  const mitk::PlaneGeometry* m_PlaneGeometry;
};

} // end namespace

#endif
