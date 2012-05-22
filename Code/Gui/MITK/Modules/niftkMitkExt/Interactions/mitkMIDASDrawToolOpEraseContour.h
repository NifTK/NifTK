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

#ifndef MITKMIDASDRAWTOOLOPERASECONTOUR_H
#define MITKMIDASDRAWTOOLOPERASECONTOUR_H

#include "niftkMitkExtExports.h"
#include "mitkOperation.h"
#include "mitkOperationActor.h"
#include "mitkTool.h"
#include "mitkContourSet.h"

namespace mitk
{

/**
 * \class MIDASDrawToolOpEraseContour
 * \brief Operation class to hold data to pass back to this MIDASDrawTool,
 * so that this MIDASDrawTool can execute the Undo/Redo command.
 */
class NIFTKMITKEXT_EXPORT MIDASDrawToolOpEraseContour: public mitk::Operation
{
public:

  MIDASDrawToolOpEraseContour(
      mitk::OperationType type,
      mitk::ContourSet* contour
      );
  ~MIDASDrawToolOpEraseContour() {};
  mitk::ContourSet* GetContourSet() const { return m_ContourSet;}

private:
  mitk::ContourSet::Pointer m_ContourSet;
};

} // end namespace

#endif
