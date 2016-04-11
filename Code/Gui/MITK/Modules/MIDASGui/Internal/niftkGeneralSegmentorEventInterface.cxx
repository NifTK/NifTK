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

//-----------------------------------------------------------------------------
niftkGeneralSegmentorEventInterface::niftkGeneralSegmentorEventInterface()
: m_GeneralSegmentorController(nullptr)
{
}


//-----------------------------------------------------------------------------
niftkGeneralSegmentorEventInterface::~niftkGeneralSegmentorEventInterface()
{
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorEventInterface::SetGeneralSegmentorController( niftkGeneralSegmentorController* generalSegmentorController)
{
  m_GeneralSegmentorController = generalSegmentorController;
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorEventInterface::ExecuteOperation(mitk::Operation* op)
{
  m_GeneralSegmentorController->ExecuteOperation(op);
}
