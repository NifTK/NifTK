/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseController.h"

#include <QmitkRenderWindow.h>

#include "niftkBaseGUI.h"
#include "niftkIBaseView.h"

//-----------------------------------------------------------------------------
niftk::BaseController::BaseController(niftkIBaseView* view)
  : m_GUI(nullptr),
    m_View(view)
{
}


//-----------------------------------------------------------------------------
niftk::BaseController::~BaseController()
{
}


//-----------------------------------------------------------------------------
void niftk::BaseController::SetupGUI(QWidget* parent)
{
  m_GUI = this->CreateGUI(parent);
}


//-----------------------------------------------------------------------------
niftk::BaseGUI* niftk::BaseController::GetGUI() const
{
  return m_GUI;
}


//-----------------------------------------------------------------------------
niftkIBaseView* niftk::BaseController::GetView() const
{
  return m_View;
}


//-----------------------------------------------------------------------------
mitk::DataStorage* niftk::BaseController::GetDataStorage() const
{
  return m_View->GetDataStorage();
}


//-----------------------------------------------------------------------------
void niftk::BaseController::RequestRenderWindowUpdate() const
{
  m_View->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
QList<mitk::DataNode::Pointer> niftk::BaseController::GetDataManagerSelection() const
{
  return m_View->GetDataManagerSelection();
}


//-----------------------------------------------------------------------------
mitk::SliceNavigationController* niftk::BaseController::GetSliceNavigationController() const
{
  return m_View->GetSliceNavigationController();
}


//-----------------------------------------------------------------------------
MIDASOrientation niftk::BaseController::GetOrientationAsEnum()
{
  MIDASOrientation orientation = MIDAS_ORIENTATION_UNKNOWN;
  const mitk::SliceNavigationController* sliceNavigationController = this->GetSliceNavigationController();
  if (sliceNavigationController != NULL)
  {
    mitk::SliceNavigationController::ViewDirection viewDirection = sliceNavigationController->GetViewDirection();

    if (viewDirection == mitk::SliceNavigationController::Axial)
    {
      orientation = MIDAS_ORIENTATION_AXIAL;
    }
    else if (viewDirection == mitk::SliceNavigationController::Sagittal)
    {
      orientation = MIDAS_ORIENTATION_SAGITTAL;
    }
    else if (viewDirection == mitk::SliceNavigationController::Frontal)
    {
      orientation = MIDAS_ORIENTATION_CORONAL;
    }
  }
  return orientation;
}
