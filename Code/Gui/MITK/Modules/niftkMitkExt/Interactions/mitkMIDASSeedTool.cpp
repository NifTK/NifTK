/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-19 22:31:43 +0000 (Sat, 19 Nov 2011) $
 Revision          : $Revision: 7815 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkMIDASSeedTool.h"
#include "mitkMIDASSeedTool.xpm"
#include "mitkToolManager.h"
#include "mitkPointSet.h"
#include "mitkProperties.h"
#include "mitkStateEvent.h"
#include "mitkPositionEvent.h"
#include "mitkRenderingManager.h"
#include "mitkGlobalInteraction.h"

namespace mitk{
  MITK_TOOL_MACRO(NIFTKMITKEXT_EXPORT, MIDASSeedTool, "MIDAS Seed Tool");
}

mitk::MIDASSeedTool::~MIDASSeedTool()
{
}

mitk::MIDASSeedTool::MIDASSeedTool() : MIDASTool("dummy")
, m_PointSetInteractor(NULL)
{
}

const char* mitk::MIDASSeedTool::GetName() const
{
  return "Seed";
}

const char** mitk::MIDASSeedTool::GetXPM() const
{
  return mitkMIDASSeedTool_xpm;
}

float mitk::MIDASSeedTool::CanHandleEvent(const StateEvent *event) const
{
  // See StateMachine.xml for event Ids.
  if (event != NULL
      && event->GetEvent() != NULL
      && (   event->GetId() == 1   // left mouse down - see QmitkNiftyViewApplicationPlugin::MIDAS_PAINTBRUSH_TOOL_STATE_MACHINE_XML
          || event->GetId() == 505 // left mouse up
          || event->GetId() == 530 // left mouse down and move
          || event->GetId() == 4   // middle mouse down
          || event->GetId() == 533 // middle mouse down and move
          )
      )
  {
    return 1;
  }
  else
  {
    return mitk::StateMachine::CanHandleEvent(event);
  }
}

void mitk::MIDASSeedTool::Deactivated()
{
  Superclass::Deactivated();

  if (m_PointSetInteractor.IsNotNull())
  {
    mitk::GlobalInteraction::GetInstance()->RemoveInteractor(m_PointSetInteractor);
  }
}

void mitk::MIDASSeedTool::Activated()
{
  Superclass::Activated();

  mitk::PointSet* pointSet = NULL;
  mitk::DataNode* pointSetNode = NULL;

  this->FindPointSet(pointSet, pointSetNode);

  if (pointSet != NULL && pointSetNode != NULL)
  {
    if (m_PointSetInteractor.IsNull())
    {
      m_PointSetInteractor = mitk::MIDASPointSetInteractor::New("MIDASSeedTool", pointSetNode);
    }
    mitk::GlobalInteraction::GetInstance()->AddInteractor( m_PointSetInteractor );
  }
}
