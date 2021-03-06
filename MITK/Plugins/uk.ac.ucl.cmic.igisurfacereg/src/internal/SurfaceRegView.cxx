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

#include <limits>

#include <QFileDialog>
#include <QMessageBox>
#include <QtConcurrentRun>

#include <vtkDoubleArray.h>
#include <vtkMatrix4x4.h>

#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateOr.h>
#include <mitkSurface.h>
#include <QmitkIGIUtils.h>

#include <niftkPolyDataUtils.h>
#include <niftkDataStorageUtils.h>
#include <niftkVTKFunctions.h>

#include "SurfaceRegViewPreferencePage.h"


const QString SurfaceRegView::VIEW_ID = "uk.ac.ucl.cmic.igisurfacereg";

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

    // i've decided against a node filter for now. any node can have the suitable
    // matrix for representing the camera position.
    // if we decide otherwise then mitk::CoordinateAxisData would be suitable, i think.
    m_Controls->m_CameraNodeComboBox->SetDataStorage(dataStorage);
    m_Controls->m_CameraNodeComboBox->SetAutoSelectNewItems(false);

    m_Controls->m_MatrixWidget->setEditable(false);
    m_Controls->m_MatrixWidget->setRange(-1e4, 1e4);

    m_Controls->m_LiveDistanceGroupBox->setCollapsed(true);
    // disable it for now, we've never used it, and it seems to have bugs:
    //  https://cmiclab.cs.ucl.ac.uk/CMIC/NifTK/issues/2873
    //  https://cmiclab.cs.ucl.ac.uk/CMIC/NifTK/issues/2579
    m_Controls->m_LiveDistanceGroupBox->setEnabled(true);//false);

    // disabled by default, for now.
    m_Controls->m_HiddenSurfaceRemovalGroupBox->setCollapsed(true);

    bool  ok = false;
    ok = QObject::connect(m_Controls->m_SurfaceBasedRegistrationButton, SIGNAL(pressed()), this, SLOT(OnCalculateButtonPressed()));
    assert(ok);
    ok = QObject::connect(m_Controls->m_ComposeWithDataButton, SIGNAL(pressed()), this, SLOT(OnComposeWithDataButtonPressed()));
    assert(ok);
    ok = QObject::connect(m_Controls->m_SaveToFileButton, SIGNAL(pressed()), this, SLOT(OnSaveToFileButtonPressed()));
    assert(ok);
    ok = QObject::connect(m_Controls->m_LiveDistanceUpdateButton, SIGNAL(clicked()), this, SLOT(OnComputeDistance()));
    assert(ok);

    dataStorage->ChangedNodeEvent.AddListener(mitk::MessageDelegate1<SurfaceRegView, const mitk::DataNode*>(this, &SurfaceRegView::DataStorageEventListener));

    RetrievePreferenceValues();
  }
}


//-----------------------------------------------------------------------------
void SurfaceRegView::OnComputeDistance()
{
  // wouldnt be visible otherwise
  assert(m_Controls->m_LiveDistanceGroupBox->isChecked());

  if (m_Controls->m_FixedSurfaceComboBox->GetSelectedNode().IsNull())
    return;
  if (m_Controls->m_MovingSurfaceComboBox->GetSelectedNode().IsNull())
    return;

  // disable it until we are done with the current computation.
  m_Controls->m_LiveDistanceUpdateButton->setEnabled(false);

  // should not be able to call/click here if it's still running.
  assert(!m_BackgroundProcess.isRunning());

  try
  {
    // essentially the same stuff that ICPBasedRegistration::Update() does.
    // we should do that before we kick off the worker thread! 
    // otherwise someone else might move around the node's matrices.
    vtkPolyData *fixedPoly = vtkPolyData::New();
    niftk::NodeToPolyData(m_Controls->m_FixedSurfaceComboBox->GetSelectedNode(), *fixedPoly);

    vtkPolyData *movingPoly = vtkPolyData::New();
    niftk::NodeToPolyData(m_Controls->m_MovingSurfaceComboBox->GetSelectedNode(), *movingPoly);

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
  catch (...)
  {
    // just swallow it.
    // we certainly do not want to keep popping up error message boxes, if for example a node is moving
    // around constantly (being attached to a tracker).
    MITK_WARN << "Caught exception while preparing ICP distance calculation";
  }
}


//-----------------------------------------------------------------------------
float SurfaceRegView::ComputeDistance(vtkSmartPointer<vtkPolyData> fixed, vtkSmartPointer<vtkPolyData> moving)
{
  // note: this is run in a worker thread! do not do any updates to data storage or nodes!

  try
  {
    vtkSmartPointer<vtkDoubleArray>   result;

    // FIXME: this crashes because targetLocator has a null tree member... sometimes???
    niftk::DistanceToSurface(moving, fixed, result);

    double  sqsum = 0;
    for (int i = 0; i < result->GetNumberOfTuples(); ++i)
    {
      // p is the distance or error.
      double p = result->GetValue(i);
      sqsum += p * p;
    }

    sqsum /= result->GetNumberOfTuples();
    double  rms = std::sqrt(sqsum);
    return rms;
  }
  catch (...)
  {
    return std::numeric_limits<float>::quiet_NaN();
  }
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
    m_MaxIterations = prefs->GetInt(SurfaceRegViewPreferencePage::MAXIMUM_NUMBER_OF_ITERATIONS,
        niftk::ICPBasedRegistrationConstants::DEFAULT_MAX_ITERATIONS );
    m_MaxPoints = prefs->GetInt(SurfaceRegViewPreferencePage::MAXIMUM_NUMBER_OF_POINTS, 
        niftk::ICPBasedRegistrationConstants::DEFAULT_MAX_POINTS);
    m_TLSITerations = prefs->GetInt(SurfaceRegViewPreferencePage::TLS_ITERATIONS,
        niftk::ICPBasedRegistrationConstants::DEFAULT_TLS_ITERATIONS);
    m_TLSPercentage = prefs->GetInt(SurfaceRegViewPreferencePage::TLS_PERCENTAGE,
        niftk::ICPBasedRegistrationConstants::DEFAULT_TLS_PERCENTAGE);
  }
}


//-----------------------------------------------------------------------------
void SurfaceRegView::OnCalculateButtonPressed()
{
  mitk::DataNode::Pointer fixednode = m_Controls->m_FixedSurfaceComboBox->GetSelectedNode();
  mitk::DataNode::Pointer movingnode = m_Controls->m_MovingSurfaceComboBox->GetSelectedNode();

  if ( fixednode.IsNull() )
  {
    QMessageBox msgBox;
    msgBox.setText("The fixed surface or point set is non-existent, or not-selected.");
    msgBox.setInformativeText("Please select a fixed surface or point set.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }
  

  if ( movingnode.IsNull() )
  {
    QMessageBox msgBox;
    msgBox.setText("The moving surface is non-existent, or not-selected.");
    msgBox.setInformativeText("Please select a moving surface.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }
  
  niftk::ICPBasedRegistration::Pointer registration = niftk::ICPBasedRegistration::New();
  registration->SetMaximumNumberOfLandmarkPointsToUse(m_MaxPoints);
  registration->SetMaximumIterations(m_MaxIterations);
  registration->SetTLSIterations(m_TLSITerations);
  registration->SetTLSPercentage(m_TLSPercentage);
  if (m_Controls->m_HiddenSurfaceRemovalGroupBox->isChecked())
  {
    registration->Update(fixednode,
                         movingnode,
                         *m_Matrix,
                         m_Controls->m_CameraNodeComboBox->GetSelectedNode(),
                         m_Controls->m_FlipNormalsCheckBox->isChecked()
                        );
  }
  else
  {
    registration->Update(fixednode,
                         movingnode,
                         *m_Matrix
                        );
  }
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      m_Controls->m_MatrixWidget->setValue(i, j, m_Matrix->GetElement(i, j));
    }
  }

  // we seem to need an explicit node-modified to trigger the usual listeners.
  // and even with this, the window will not re-render on its own, have to click in it.
  movingnode->Modified();
}


//--------------------------------------------------------------------------------
void SurfaceRegView::OnComposeWithDataButtonPressed()
{
  ComposeTransformWithSelectedNodes(*m_Matrix, *m_Controls->m_ComposeWithDataNode);
}


//-----------------------------------------------------------------------------
void SurfaceRegView::OnSaveToFileButtonPressed()
{
  QString fileName = QFileDialog::getSaveFileName( NULL,
                                                   tr("Save Transform As ..."),
                                                   QDir::currentPath(),
                                                   "4x4 file (*.4x4);;Matrix file (*.mat);;Text file (*.txt);;All files (*.*)" );
  if (fileName.size() > 0)
  {
    SaveMatrixToFile(*m_Matrix, fileName);
  }
}


//-----------------------------------------------------------------------------
void SurfaceRegView::SetFocus()
{
  m_Controls->m_FixedSurfaceComboBox->setFocus();
}
