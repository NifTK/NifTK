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
#include <mitkDataStorageUtils.h>
#include <QMessageBox>
#include <QtConcurrentRun>
#include <vtkFunctions.h>
#include <vtkDoubleArray.h>


const std::string SurfaceRegView::VIEW_ID = "uk.ac.ucl.cmic.igisurfacereg";

//-----------------------------------------------------------------------------
SurfaceRegView::SurfaceRegView()
: m_Controls(NULL)
, m_Matrix(NULL)
{
  m_Matrix = vtkMatrix4x4::New();
  m_Matrix->Identity();

  bool ok = false;
  ok = connect(&m_BackgroundProcessWatcher, SIGNAL(finished()), this, SLOT(OnBackgroundProcessFinished()));
  assert(ok);
}


//-----------------------------------------------------------------------------
SurfaceRegView::~SurfaceRegView()
{
  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
  if (dataStorage.IsNotNull())
  {
    dataStorage->ChangedNodeEvent.RemoveListener(mitk::MessageDelegate1<SurfaceRegView, const mitk::DataNode*>(this, &SurfaceRegView::DataStorageEventListener));
  }

  m_BackgroundProcessWatcher.waitForFinished();

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
    m_Controls->m_MatrixWidget->setRange(-1e4, 1e4);

    m_Controls->m_LiveDistanceGroupBox->setCollapsed(true);

    connect(m_Controls->m_SurfaceBasedRegistrationButton, SIGNAL(pressed()), this, SLOT(OnCalculateButtonPressed()));
    connect(m_Controls->m_ComposeWithDataButton, SIGNAL(pressed()), this, SLOT(OnComposeWithDataButtonPressed()));

    connect(m_Controls->m_LiveDistanceUpdateButton, SIGNAL(clicked()), this, SLOT(OnComputeDistance()));

    dataStorage->ChangedNodeEvent.AddListener(mitk::MessageDelegate1<SurfaceRegView, const mitk::DataNode*>(this, &SurfaceRegView::DataStorageEventListener));

    RetrievePreferenceValues();
  }
}


//-----------------------------------------------------------------------------
void SurfaceRegView::OnComputeDistance()
{
  // wouldnt be visible otherwise
  assert(m_Controls->m_LiveDistanceGroupBox->isChecked());

  // disable it until we are done with the current computation.
  m_Controls->m_LiveDistanceUpdateButton->setEnabled(false);

  // should not be able to call/click here if it's still running.
  assert(!m_BackgroundProcess.isRunning());

  // essentially the same stuff that SurfaceBasedRegistration::Update() does.
  // we should do that before we kick off the worker thread! 
  // otherwise someone else might move around the node's matrices.
  vtkSmartPointer<vtkPolyData> fixedPoly = vtkPolyData::New();
  mitk::SurfaceBasedRegistration::NodeToPolyData(m_Controls->m_FixedSurfaceComboBox->GetSelectedNode(), fixedPoly);
  vtkSmartPointer<vtkPolyData> movingPoly = vtkPolyData::New();
  mitk::SurfaceBasedRegistration::NodeToPolyData(m_Controls->m_MovingSurfaceComboBox->GetSelectedNode(), movingPoly);

  // this seems a bit messy here:
  // the "surface" passed in first needs to have vtk cells, otherwise it crashes.
  // so if it doesnt then we swap, if both dont have any then dont do anything.
  if (fixedPoly->GetNumberOfCells() == 0)
  {
    if (movingPoly->GetNumberOfCells() == 0)
    {
      m_Controls->m_DistanceLineEdit->setText("ERROR: need cells on at least one of the objects");
    }
    std::swap(fixedPoly, movingPoly);
  }

  m_BackgroundProcess = QtConcurrent::run(this, &SurfaceRegView::ComputeDistance, fixedPoly, movingPoly);
  m_BackgroundProcessWatcher.setFuture(m_BackgroundProcess);
}


//-----------------------------------------------------------------------------
float SurfaceRegView::ComputeDistance(vtkSmartPointer<vtkPolyData> fixed, vtkSmartPointer<vtkPolyData> moving)
{
  // note: this is run in a worker thread! do not do any updates to data storage or nodes!

  vtkSmartPointer<vtkDoubleArray>   result;
  // FIXME: this crashes because targetLocator has a null tree member... sometimes???
  DistanceToSurface(moving, fixed, result);

  double  sum = 0;
  for (int i = 0; i < result->GetNumberOfTuples(); ++i)
  {
    double p = result->GetValue(i);
    sum += p;
  }

  return sum;
}


//-----------------------------------------------------------------------------
void SurfaceRegView::OnBackgroundProcessFinished()
{
  float   distance = m_BackgroundProcessWatcher.result();

  m_Controls->m_DistanceLineEdit->setText(tr("%1").arg(distance));
  m_Controls->m_LiveDistanceUpdateButton->setEnabled(true);
}


//-----------------------------------------------------------------------------
void SurfaceRegView::DataStorageEventListener(const mitk::DataNode* node)
{
  if (m_Controls->m_LiveDistanceGroupBox->isChecked())
  {
    if ((node == m_Controls->m_FixedSurfaceComboBox->GetSelectedNode()) ||
        (node == m_Controls->m_MovingSurfaceComboBox->GetSelectedNode()))
    {
      // re-use the existing gui mechanism
      m_Controls->m_LiveDistanceUpdateButton->click();
    }
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
void SurfaceRegView::OnCalculateButtonPressed()
{
  mitk::DataNode* fixednode = m_Controls->m_FixedSurfaceComboBox->GetSelectedNode();
  mitk::DataNode* movingnode = m_Controls->m_MovingSurfaceComboBox->GetSelectedNode();

  if ( fixednode == NULL )
  {
    QMessageBox msgBox;
    msgBox.setText("The fixed surface or point set is non-existent, or not-selected.");
    msgBox.setInformativeText("Please select a fixed surface or point set.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }
  

  if ( movingnode == NULL )
  {
    QMessageBox msgBox;
    msgBox.setText("The moving surface is non-existent, or not-selected.");
    msgBox.setInformativeText("Please select a moving surface.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }
  
  mitk::SurfaceBasedRegistration::Pointer registration = mitk::SurfaceBasedRegistration::New();
  registration->Update(fixednode, movingnode, m_Matrix);
  
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      m_Controls->m_MatrixWidget->setValue(i, j, m_Matrix->GetElement(i, j));
    }
  }

  registration->ApplyTransform(movingnode);
  // we seem to need an explicit node-modified to trigger the usual listeners.
  // and even with this, the window will not re-render on its own, have to click in it.
  movingnode->Modified();
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

  bool successful = mitk::SaveVtkMatrix4x4ToFile(fileName.toStdString(), *m_Matrix);

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
