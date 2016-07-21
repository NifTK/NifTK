/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkContourToolOpAccumulateContour_h
#define niftkContourToolOpAccumulateContour_h

#include "niftkMIDASExports.h"

#include <mitkContourModelSet.h>
#include <mitkOperation.h>
#include <mitkOperationActor.h>
#include <mitkTool.h>
#include <mitkToolManager.h>

namespace niftk
{

/**
 * \class ContourToolOpAccumulateContour
 * \brief Operation class to hold data to pass back to this ContourTool,
 * so that this ContourTool can execute the Undo/Redo command.
 */
class NIFTKMIDAS_EXPORT ContourToolOpAccumulateContour: public mitk::Operation
{
public:

  ContourToolOpAccumulateContour(
      mitk::OperationType type,
      bool redo,
      int dataIndex,
      mitk::ContourModelSet::Pointer contourSet
      );

  ~ContourToolOpAccumulateContour();

  bool IsRedo() const;

  int GetDataIndex() const;

  mitk::ContourModelSet::Pointer GetContourSet() const;

private:

  bool m_Redo;
  int  m_DataIndex;
  mitk::ContourModelSet::Pointer m_ContourSet;

};

}

#endif
