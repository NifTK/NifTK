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
#include <mitkNodePredicateOr.h>
#include <mitkSurface.h>
#include <vtkMatrix4x4.h>
#include <mitkSurfaceBasedRegistration.h>
#include <mitkPointBasedRegistration.h>
#include <mitkFileIOUtils.h>
#include <QMessageBox>
#include <mitkCoordinateAxesData.h>
#include <mitkAffineTransformDataNodeProperty.h>

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

    mitk::TNodePredicateDataType<mitk::PointSet>::Pointer isPointSet = 
      mitk::TNodePredicateDataType<mitk::PointSet>::New();

    mitk::NodePredicateOr::Pointer isSurfaceOrPoints = 
      mitk::NodePredicateOr::New(isSurface,isPointSet);

    m_Controls->m_FixedSurfaceComboBox->SetPredicate(isSurfaceOrPoints);
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
  mitk::PointSet::Pointer fixedPoints = NULL;
  mitk::DataNode* node = m_Controls->m_FixedSurfaceComboBox->GetSelectedNode();

  if ( node != NULL )
  {
    fixedSurface = dynamic_cast<mitk::Surface*>(node->GetData());
    fixedPoints = dynamic_cast<mitk::PointSet*>(node->GetData());
  }

  if ( fixedSurface.IsNull() == fixedPoints.IsNull() )
  {
    QMessageBox msgBox;
    msgBox.setText("The fixed surface or point set is non-existent, or not-selected.");
    msgBox.setInformativeText("Please select a fixed surface or point set.");
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
    msgBox.setText("The moving surface is non-existent, or not-selected.");
    msgBox.setInformativeText("Please select a moving surface.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }

  mitk::SurfaceBasedRegistration::Pointer registration = mitk::SurfaceBasedRegistration::New();;
  if ( fixedSurface.IsNull() ) 
  {
    registration->Update(fixedPoints, movingSurface, m_Matrix);
  }
  else
  {
    registration->Update(fixedSurface, movingSurface, m_Matrix);
  }

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      m_Controls->m_MatrixWidget->setValue(i, j, m_Matrix->GetElement(i, j));
    }
  }

  mitk::CoordinateAxesData* transform = dynamic_cast<mitk::CoordinateAxesData*>(node->GetData());

  if (transform != NULL)
  {
    mitk::AffineTransformDataNodeProperty::Pointer property = dynamic_cast<mitk::AffineTransformDataNodeProperty*>(node->GetProperty("niftk.transform"));
    if (property.IsNull())
    {
      MITK_ERROR << "LiverSurgeryManager::SetTransformation the node " << node->GetName() << " does not contain the niftk.transform property" << std::endl;
      return;
    }

    transform->SetVtkMatrix(*m_Matrix);
    transform->Modified();

    property->SetTransform(*m_Matrix);
    property->Modified();

  }
  else
  {
    mitk::Geometry3D::Pointer geometry = node->GetData()->GetGeometry();
    if (geometry.IsNotNull())
    {
     // geometry->SetIndexToWorldTransformByVtkMatrix(const_cast<vtkMatrix4x4*>(&m_Matrix));
      geometry->SetIndexToWorldTransformByVtkMatrix(m_Matrix);
      geometry->Modified();
    }
  }

}

//--------------------------------------------------------------------------------
void SurfaceRegView::OnComposeWithDataButtonPressed()
{
  mitk::BaseData::Pointer data = NULL;
  mitk::DataNode* node = m_Controls->m_ComposeWithDataNode->GetSelectedNode();

  if (node != NULL)
  {
    data = dynamic_cast<mitk::BaseData*>(node->GetData());
  }

  if (data.IsNull())
  {
    QMessageBox msgBox;
    msgBox.setText("The data set is non-existent, does not contain data or is not-selected.");
    msgBox.setInformativeText("Please select a valid data set.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }

  mitk::PointBasedRegistration::Pointer controller = mitk::PointBasedRegistration::New();
  bool successful = controller->ApplyToNode(node, *m_Matrix, true);

  if (!successful)
  {
    QMessageBox msgBox;
    msgBox.setText("Failed to apply transform.");
    msgBox.setInformativeText("Please check the console.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }

}

//-----------------------------------------------------------------------------
void SurfaceRegView::OnSaveToFileButtonPressed()
{
  QString fileName = m_Controls->m_SaveToFilePathEdit->currentPath();
  if (fileName.length() == 0)
  {
    QMessageBox msgBox;
    msgBox.setText("The file name is empty.");
    msgBox.setInformativeText("Please select a file name.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }

  mitk::PointBasedRegistration::Pointer controller = mitk::PointBasedRegistration::New();
  bool successful = controller->SaveToFile(fileName.toStdString(), *m_Matrix);

  if (!successful)
  {
    QMessageBox msgBox;
    msgBox.setText("The file failed to save.");
    msgBox.setInformativeText("Please check the location.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }
}

//-----------------------------------------------------------------------------
void SurfaceRegView::SetFocus()
{
  // Set focus to a sensible widget for when the view is launched.
}
