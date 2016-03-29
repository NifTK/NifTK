/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseSegmentorController.h"

#include "niftkBaseSegmentorView.h"

//-----------------------------------------------------------------------------
niftkBaseSegmentorController::niftkBaseSegmentorController(niftkBaseSegmentorView* segmentorView)
  : m_SegmentorView(segmentorView)
{
  // Create an own tool manager and connect it to the data storage straight away.
  m_ToolManager = mitk::ToolManager::New(segmentorView->GetDataStorage());

  this->RegisterTools();
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorController::~niftkBaseSegmentorController()
{
}


//-----------------------------------------------------------------------------
mitk::ToolManager* niftkBaseSegmentorController::GetToolManager() const
{
  return m_ToolManager;
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorController::RegisterTools()
{
}
