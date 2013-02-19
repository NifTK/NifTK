/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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
      mitk::ContourSet* contour,
      const int& workingNodeNumber
      );
  ~MIDASDrawToolOpEraseContour() {};
  mitk::ContourSet* GetContourSet() const { return m_ContourSet;}
  int GetWorkingNode() const { return m_WorkingNodeNumber; }

private:
  mitk::ContourSet::Pointer m_ContourSet;
  int m_WorkingNodeNumber;
};

} // end namespace

#endif
