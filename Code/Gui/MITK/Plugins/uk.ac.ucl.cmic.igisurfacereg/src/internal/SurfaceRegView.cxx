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
#include <mitkSurface.h>
#include <vtkMatrix4x4.h>
#include <mitkSurfaceBasedRegistration.h>
#include <mitkFileIOUtils.h>
#include <QMessageBox>

const std::string SurfaceRegView::VIEW_ID = "uk.ac.ucl.cmic.igisurfacereg";

//-----------------------------------------------------------------------------
SurfaceRegView::SurfaceRegView()
: m_Controls(NULL)
, m_Matrix(NULL)
{
  m_Matrix = vtkMatrix4x4::New();
  m_Matrix->Identity();
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

    mitk::TNodePredicateDataType<mitk::Surface>::Pointer isSurface = 
      mitk::TNodePredicateDataType<mitk::Surface>::New();

    m_Controls->m_FixedSurfaceComboBox->SetPredicate(isSurface);
    m_Controls->m_FixedSurfaceComboBox->SetAutoSelectNewItems(false);

    m_Controls->m_MovingSurfaceComboBox->SetPredicate(isSurface);
    m_Controls->m_MovingSurfaceComboBox->SetAutoSelectNewItems(false);

    m_Controls->m_FixedSurfaceComboBox->SetDataStorage(dataStorage);
    m_Controls->m_MovingSurfaceComboBox->SetDataStorage(dataStorage);
    m_Controls->m_ComposeWithDataNode->SetDataStorage(dataStorage);

    m_Controls->m_MatrixWidget->setEditable(false);

    connect(m_Controls->m_SurfaceBasedRegistrationButton, SIGNAL(pressed()), this, SLOT(OnCalculateButtonPressed()));
    connect(m_Controls->m_ComposeWithDataButton, SIGNAL(pressed()), this, SLOT(OnComposeWithDataButtonPressed()));

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

void SurfaceRegView::OnCalculateButtonPressed()
{
  mitk::Surface::Pointer fixedSurface = NULL;
  mitk::DataNode* node = m_Controls->m_FixedSurfaceComboBox->GetSelectedNode();

  if ( node != NULL )
  {
    fixedSurface = dynamic_cast<mitk::Surface*>(node->GetData());
  }

  if (fixedSurface.IsNull())
  {
    QMessageBox msgBox;
    msgBox.setText("The fixed surface set is non-existent, or not-selected.");
    msgBox.setInformativeText("Please select a fixed surface.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }
  
  mitk::Surface::Pointer movingSurface = NULL;
  node = m_Controls->m_MovingSurfaceComboBox->GetSelectedNode();

  if ( node != NULL )
  {
    movingSurface = dynamic_cast<mitk::Surface*>(node->GetData());
  }

  if (movingSurface.IsNull())
  {
    QMessageBox msgBox;
    msgBox.setText("The moving surface set is non-existent, or not-selected.");
    msgBox.setInformativeText("Please select a moving surface.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }

  mitk::SurfaceBasedRegistration::Pointer registration = mitk::SurfaceBasedRegistration::New();;
  registration->Update(fixedSurface, movingSurface, m_Matrix);

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      m_Controls->m_MatrixWidget->setValue(i, j, m_Matrix->GetElement(i, j));
    }
  }
}

//-----------------------------------------------------------------------------
void SurfaceRegView::SetFocus()
{
  // Set focus to a sensible widget for when the view is launched.
}
