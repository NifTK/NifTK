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
#include "PointRegViewPreferencePage.h"
#include <mitkNodePredicateDataType.h>
#include <mitkPointSet.h>
#include <vtkMatrix4x4.h>
#include <mitkPointBasedRegistration.h>
#include <mitkFileIOUtils.h>
#include <mitkNodePredicateDataType.h>
#include <mitkDataStorageUtils.h>
#include <mitkFileIOUtils.h>
#include <QMessageBox>

const std::string PointRegView::VIEW_ID = "uk.ac.ucl.cmic.igipointreg";

//-----------------------------------------------------------------------------
PointRegView::PointRegView()
: m_Controls(NULL)
, m_Matrix(NULL)
, m_UseICPInitialisation(false)
{
  m_Matrix = vtkMatrix4x4::New();
  m_Matrix->Identity();
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
    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    m_Controls = new Ui::PointRegView();
    m_Controls->setupUi(parent);

    mitk::TNodePredicateDataType<mitk::PointSet>::Pointer isPointSet = mitk::TNodePredicateDataType<mitk::PointSet>::New();
    m_Controls->m_FixedPointsCombo->SetPredicate(isPointSet);
    m_Controls->m_FixedPointsCombo->SetAutoSelectNewItems(false);

    m_Controls->m_MovingPointsCombo->SetPredicate(isPointSet);
    m_Controls->m_MovingPointsCombo->SetAutoSelectNewItems(false);

    m_Controls->m_MovingPointsCombo->SetPredicate(isPointSet);
    m_Controls->m_MovingPointsCombo->SetAutoSelectNewItems(false);

    m_Controls->m_FixedPointsCombo->SetDataStorage(dataStorage);
    m_Controls->m_MovingPointsCombo->SetDataStorage(dataStorage);
    m_Controls->m_ComposeWithDataNode->SetDataStorage(dataStorage);

    m_Controls->m_MatrixWidget->setEditable(false);

    connect(m_Controls->m_PointBasedRegistrationButton, SIGNAL(pressed()), this, SLOT(OnCalculateButtonPressed()));
    connect(m_Controls->m_ComposeWithDataButton, SIGNAL(pressed()), this, SLOT(OnComposeWithDataButtonPressed()));
    connect(m_Controls->m_SaveToFileButton, SIGNAL(pressed()), this, SLOT(OnSaveToFileButtonPressed()));

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
    m_UseICPInitialisation = prefs->GetBool(PointRegViewPreferencePage::USE_ICP_INITIALISATION, mitk::PointBasedRegistration::DEFAULT_USE_ICP_INITIALISATION);
  }
}


//-----------------------------------------------------------------------------
void PointRegView::OnCalculateButtonPressed()
{
  mitk::PointSet::Pointer fixedPoints = NULL;
  mitk::DataNode* node = m_Controls->m_FixedPointsCombo->GetSelectedNode();

  if (node != NULL)
  {
    fixedPoints = dynamic_cast<mitk::PointSet*>(node->GetData());
  }

  if (fixedPoints.IsNull())
  {
    QMessageBox msgBox;
    msgBox.setText("The fixed point set is non-existent, or not-selected.");
    msgBox.setInformativeText("Please select a fixed point set.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }

  mitk::PointSet::Pointer movingPoints = NULL;
  node = m_Controls->m_MovingPointsCombo->GetSelectedNode();

  if (node != NULL)
  {
    movingPoints = dynamic_cast<mitk::PointSet*>(node->GetData());
  }

  if (movingPoints.IsNull())
  {
    QMessageBox msgBox;
    msgBox.setText("The moving point set is non-existent, or not-selected.");
    msgBox.setInformativeText("Please select a moving point set.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }

  if (m_UseICPInitialisation)
  {
    if (fixedPoints->GetSize() < 6 || movingPoints->GetSize() < 6)
    {
      QMessageBox msgBox;
      msgBox.setText("The point sets must have at least 6 points.");
      msgBox.setInformativeText("Please select more points.");
      msgBox.setStandardButtons(QMessageBox::Ok);
      msgBox.setDefaultButton(QMessageBox::Ok);
      msgBox.exec();
      return;
    }
  }
  else
  {
    if (fixedPoints->GetSize() < 3 || movingPoints->GetSize() < 3)
    {
      QMessageBox msgBox;
      msgBox.setText("The point sets must have at least 3 points.");
      msgBox.setInformativeText("Please select more points.");
      msgBox.setStandardButtons(QMessageBox::Ok);
      msgBox.setDefaultButton(QMessageBox::Ok);
      msgBox.exec();
      return;
    }
    if (fixedPoints->GetSize() != movingPoints->GetSize())
    {
      QMessageBox msgBox;
      msgBox.setText("The point sets must have the same number of points.");
      msgBox.setInformativeText("Please select ordered and corresponding points.");
      msgBox.setStandardButtons(QMessageBox::Ok);
      msgBox.setDefaultButton(QMessageBox::Ok);
      msgBox.exec();
      return;
    }
  }

  mitk::PointBasedRegistration::Pointer registration = mitk::PointBasedRegistration::New();
  double error = registration->Update(fixedPoints,
                                      movingPoints,
                                      m_UseICPInitialisation,
                                      *m_Matrix
                                      );

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      m_Controls->m_MatrixWidget->setValue(i, j, m_Matrix->GetElement(i, j));
    }
  }
  QString formattedDouble = QString::number(error);
  m_Controls->m_RMSError->setText(QString("FRE = ") + formattedDouble);
}


//-----------------------------------------------------------------------------
void PointRegView::OnComposeWithDataButtonPressed()
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
  bool successful = mitk::ApplyToNode(node, *m_Matrix, true);

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
void PointRegView::OnSaveToFileButtonPressed()
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
  bool successful = mitk::SaveVtkMatrix4x4ToFileIfFileName(fileName.toStdString(), *m_Matrix);

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
void PointRegView::SetFocus()
{
  m_Controls->m_FixedPointsCombo->setFocus();
}
