/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASSeedTool.h"
#include "mitkMIDASSeedTool.xpm"
#include <mitkToolManager.h>
#include <mitkPointSet.h>
#include <mitkProperties.h>
#include <mitkPositionEvent.h>
#include <mitkRenderingManager.h>
#include <mitkGlobalInteraction.h>

namespace mitk
{
  MITK_TOOL_MACRO(NIFTKMIDAS_EXPORT, MIDASSeedTool, "MIDAS Seed Tool");
}

//-----------------------------------------------------------------------------
mitk::MIDASSeedTool::~MIDASSeedTool()
{
}


//-----------------------------------------------------------------------------
mitk::MIDASSeedTool::MIDASSeedTool() : MIDASTool("dummy")
, m_PointSetInteractor(NULL)
{
}


//-----------------------------------------------------------------------------
const char* mitk::MIDASSeedTool::GetName() const
{
  return "Seed";
}


//-----------------------------------------------------------------------------
const char** mitk::MIDASSeedTool::GetXPM() const
{
  return mitkMIDASSeedTool_xpm;
}


//-----------------------------------------------------------------------------
void mitk::MIDASSeedTool::InstallEventFilter(MIDASEventFilter* eventFilter)
{
  Superclass::InstallEventFilter(eventFilter);
  if (m_PointSetInteractor.IsNotNull())
  {
    m_PointSetInteractor->InstallEventFilter(eventFilter);
  }
}


//-----------------------------------------------------------------------------
void mitk::MIDASSeedTool::RemoveEventFilter(MIDASEventFilter* eventFilter)
{
  if (m_PointSetInteractor.IsNotNull())
  {
    m_PointSetInteractor->RemoveEventFilter(eventFilter);
  }
  Superclass::RemoveEventFilter(eventFilter);
}


//-----------------------------------------------------------------------------
void mitk::MIDASSeedTool::Activated()
{
  Superclass::Activated();

  mitk::PointSet* pointSet = NULL;
  mitk::DataNode* pointSetNode = NULL;

  this->FindPointSet(pointSet, pointSetNode);

  /// TODO

//  if (pointSet != NULL && pointSetNode != NULL)
//  {
//    if (m_PointSetInteractor.IsNull())
//    {
////      m_PointSetInteractor = mitk::MIDASPointSetInteractor::New("MIDASSeedTool", pointSetNode);
//      m_PointSetInteractor = mitk::MIDASPointSetInteractor::New();

//      std::vector<mitk::MIDASEventFilter*> eventFilters = this->GetEventFilters();
//      std::vector<mitk::MIDASEventFilter*>::const_iterator it = eventFilters.begin();
//      std::vector<mitk::MIDASEventFilter*>::const_iterator itEnd = eventFilters.end();
//      for ( ; it != itEnd; ++it)
//      {
//        m_PointSetInteractor->InstallEventFilter(*it);
//      }

////      m_PointSetInteractor->SetAccuracy(1.0);
//    }

//    mitk::GlobalInteraction* globalInteraction = mitk::GlobalInteraction::GetInstance();
//    globalInteraction->AddInteractor(m_PointSetInteractor);
//  }
}


//-----------------------------------------------------------------------------
void mitk::MIDASSeedTool::Deactivated()
{
  Superclass::Deactivated();

  /// TODO

//  if (m_PointSetInteractor.IsNotNull())
//  {
//    std::vector<mitk::MIDASEventFilter*> eventFilters = this->GetEventFilters();
//    std::vector<mitk::MIDASEventFilter*>::const_iterator it = eventFilters.begin();
//    std::vector<mitk::MIDASEventFilter*>::const_iterator itEnd = eventFilters.end();
//    for ( ; it != itEnd; ++it)
//    {
//      m_PointSetInteractor->RemoveEventFilter(*it);
//    }
//    mitk::GlobalInteraction::GetInstance()->RemoveInteractor(m_PointSetInteractor);
//  }
  m_PointSetInteractor = NULL;
}
