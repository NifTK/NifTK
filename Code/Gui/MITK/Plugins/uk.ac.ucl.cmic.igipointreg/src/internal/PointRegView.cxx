/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Qmitk
#include "PointRegView.h"
#include <mitkNodePredicateDataType.h>
#include <mitkPointSet.h>
#include <vtkMatrix4x4.h>
#include <mitkPointBasedRegistration.h>
#include <mitkFileIOUtils.h>

const std::string PointRegView::VIEW_ID = "uk.ac.ucl.cmic.igipointreg";

//-----------------------------------------------------------------------------
PointRegView::PointRegView()
: m_Controls(NULL)
{
}


//-----------------------------------------------------------------------------
PointRegView::~PointRegView()
{
  if (m_Controls != NULL)
  {
    delete m_Controls;
  }
}


//-----------------------------------------------------------------------------
std::string PointRegView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void PointRegView::CreateQtPartControl( QWidget *parent )
{
  if (!m_Controls)
  {
    m_Controls = new Ui::PointRegView();
    m_Controls->setupUi(parent);

    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    RetrievePreferenceValues();
  }
}


//-----------------------------------------------------------------------------
void PointRegView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void PointRegView::RetrievePreferenceValues()
{
  berry::IPreferences::Pointer prefs = GetPreferences();
  if (prefs.IsNotNull())
  {

  }
}


//-----------------------------------------------------------------------------
void PointRegView::SetFocus()
{
  // Set focus to a sensible widget for when the view is launched.
}
