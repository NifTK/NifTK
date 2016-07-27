/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkGeneralSegmentorEventInterface.h"

#include "niftkGeneralSegmentorController.h"

namespace niftk
{

//-----------------------------------------------------------------------------
GeneralSegmentorEventInterface::GeneralSegmentorEventInterface()
: m_GeneralSegmentorController(nullptr)
{
}


//-----------------------------------------------------------------------------
GeneralSegmentorEventInterface::~GeneralSegmentorEventInterface()
{
}


//-----------------------------------------------------------------------------
void GeneralSegmentorEventInterface::SetGeneralSegmentorController(GeneralSegmentorController* generalSegmentorController)
{
  m_GeneralSegmentorController = generalSegmentorController;
}


//-----------------------------------------------------------------------------
void GeneralSegmentorEventInterface::ExecuteOperation(mitk::Operation* op)
{
  m_GeneralSegmentorController->ExecuteOperation(op);
}

}
