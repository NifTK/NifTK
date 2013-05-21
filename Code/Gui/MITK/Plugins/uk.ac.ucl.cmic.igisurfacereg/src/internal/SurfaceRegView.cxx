/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Qmitk
#include "SurfaceRegView.h"
#include <mitkNodePredicateDataType.h>
#include <mitkPointSet.h>
#include <vtkMatrix4x4.h>
#include <mitkSurfaceBasedRegistration.h>
#include <mitkFileIOUtils.h>

const std::string SurfaceRegView::VIEW_ID = "uk.ac.ucl.cmic.igisurfacereg";

//-----------------------------------------------------------------------------
SurfaceRegView::SurfaceRegView()
: m_Controls(NULL)
{
}


//-----------------------------------------------------------------------------
SurfaceRegView::~SurfaceRegView()
{
  if (m_Controls != NULL)
  {
    delete m_Controls;
  }
}


//-----------------------------------------------------------------------------
std::string SurfaceRegView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void SurfaceRegView::CreateQtPartControl( QWidget *parent )
{
  if (!m_Controls)
  {
    m_Controls = new Ui::SurfaceRegView();
    m_Controls->setupUi(parent);

    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    RetrievePreferenceValues();
  }
}


//-----------------------------------------------------------------------------
void SurfaceRegView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void SurfaceRegView::RetrievePreferenceValues()
{
  berry::IPreferences::Pointer prefs = GetPreferences();
  if (prefs.IsNotNull())
  {

  }
}


//-----------------------------------------------------------------------------
void SurfaceRegView::SetFocus()
{
  // Set focus to a sensible widget for when the view is launched.
}
