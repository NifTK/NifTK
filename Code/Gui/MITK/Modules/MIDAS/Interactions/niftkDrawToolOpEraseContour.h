/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkDrawToolOpEraseContour_h
#define niftkDrawToolOpEraseContour_h

#include "niftkMIDASExports.h"

#include <mitkContourModelSet.h>
#include <mitkOperation.h>
#include <mitkOperationActor.h>
#include <mitkTool.h>

namespace niftk
{

/**
 * \class MIDASDrawToolOpEraseContour
 * \brief Operation class to hold data to pass back to this MIDASDrawTool,
 * so that this MIDASDrawTool can execute the Undo/Redo command.
 */
class NIFTKMIDAS_EXPORT MIDASDrawToolOpEraseContour: public mitk::Operation
{
public:

  MIDASDrawToolOpEraseContour(
      mitk::OperationType type,
      mitk::ContourModelSet* contour,
      int dataIndex
      );

  ~MIDASDrawToolOpEraseContour();

  mitk::ContourModelSet* GetContourModelSet() const;

  int GetDataIndex() const;

private:

  mitk::ContourModelSet::Pointer m_ContourModelSet;
  int m_DataIndex;

};

}

#endif
