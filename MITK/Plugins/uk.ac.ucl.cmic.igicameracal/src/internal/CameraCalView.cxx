/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Qmitk
#include "CameraCalView.h"
#include "CameraCalViewPreferencePage.h"
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateDataType.h>
#include <QMessageBox>
#include <QFileDialog>

namespace niftk
{

const std::string CameraCalView::VIEW_ID = "uk.ac.ucl.cmic.igicameracal";

//-----------------------------------------------------------------------------
CameraCalView::CameraCalView()
: m_Controls(NULL)
{
}


//-----------------------------------------------------------------------------
CameraCalView::~CameraCalView()
{
  if (m_Controls != NULL)
  {
    delete m_Controls;
  }
}


//-----------------------------------------------------------------------------
std::string CameraCalView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void CameraCalView::CreateQtPartControl( QWidget *parent )
{
  if (!m_Controls)
  {
    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    m_Controls = new Ui::CameraCalView();
    m_Controls->setupUi(parent);

    RetrievePreferenceValues();
  }
}


//-----------------------------------------------------------------------------
void CameraCalView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void CameraCalView::RetrievePreferenceValues()
{
  berry::IPreferences::Pointer prefs = GetPreferences();
  if (prefs.IsNotNull())
  {
  }
}


//-----------------------------------------------------------------------------
void CameraCalView::SetFocus()
{
  m_Controls->m_LeftCameraComboBox->setFocus();
}

} // end namespace
