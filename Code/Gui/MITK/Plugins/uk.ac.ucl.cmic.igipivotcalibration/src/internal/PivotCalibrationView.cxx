/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Qmitk
#include "PivotCalibrationView.h"
#include "PivotCalibrationViewActivator.h"

#include <QmitkIGIUtils.h>
#include <mitkCoordinateAxesData.h>
#include <mitkIOUtil.h>
#include <mitkFileIOUtils.h>

#include "niftkFileHelper.h"
#include <niftkVTKFunctions.h>
#include <mitkPivotCalibration.h>

#include <ctkDictionary.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>

#include <mitkNodePredicateBase.h>
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateOr.h>
#include <mitkDataStorageUtils.h>
#include <mitkDataNodeFactory.h>
#include <mitkBaseGeometry.h>

#include <QMessageBox>
#include <QtConcurrentRun>
#include <QFileDialog>

#include <vtkMatrix4x4.h>
#include <vtkTransform.h>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>


//-----------------------------------------------------------------------------
const char* PivotCalibrationView::VIEW_ID = "uk.ac.ucl.cmic.igipivotcalibration";

//-----------------------------------------------------------------------------
PivotCalibrationView::PivotCalibrationView()
: m_Controls(NULL)
, m_Matrix(NULL)
{
  m_Matrix = vtkSmartPointer<vtkMatrix4x4>::New();
  m_Matrix->Identity();
}


//-----------------------------------------------------------------------------
PivotCalibrationView::~PivotCalibrationView()
{
  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
  if (dataStorage.IsNotNull())
  {
    dataStorage->ChangedNodeEvent.RemoveListener(mitk::MessageDelegate1<PivotCalibrationView, const mitk::DataNode*>(this, &PivotCalibrationView::DataStorageEventListener));
  }

  if (m_Controls != NULL)
  {
    delete m_Controls;
  }
}


//-----------------------------------------------------------------------------
void PivotCalibrationView::CreateQtPartControl( QWidget *parent )
{
  if (!m_Controls)
  {
    m_Controls = new Ui::PivotCalibrationView();
    m_Controls->setupUi(parent);

    bool  ok = false;

    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    m_Controls->m_MatrixDirectoryChooser->setFilters(ctkPathLineEdit::Dirs);
    m_Controls->m_MatrixDirectoryChooser->setOptions(ctkPathLineEdit::ShowDirsOnly);
    m_Controls->m_MatrixDirectoryChooser->setCurrentPath("");

    m_Controls->m_MatrixWidget->setEditable(false);
    m_Controls->m_MatrixWidget->setRange(-1e4, 1e4);

    ok = QObject::connect(m_Controls->m_DoPivotCalibrationPushButton, SIGNAL(clicked()), this, SLOT(OnPivotCalibrationButtonClicked()));

    ok = QObject::connect(m_Controls->m_SaveToFileButton, SIGNAL(clicked()), this, SLOT(OnSaveToFileButtonClicked()));
    assert(ok);

    dataStorage->ChangedNodeEvent.AddListener(mitk::MessageDelegate1<PivotCalibrationView, const mitk::DataNode*>(this, &PivotCalibrationView::DataStorageEventListener));
  }
}

//-----------------------------------------------------------------------------
void PivotCalibrationView::DataStorageEventListener(const mitk::DataNode* node)
{
}

//-----------------------------------------------------------------------------
void PivotCalibrationView::OnPivotCalibrationButtonClicked()
{
  QString matrixDirectoryName = m_Controls->m_MatrixDirectoryChooser->currentPath();
  if (matrixDirectoryName.length() == 0)
  {
    QMessageBox msgBox; 
    msgBox.setText("The tracking matrix folder is not-selected.");
    msgBox.setInformativeText("Please select a tracking matrix folder.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }
  
  double residualError = std::numeric_limits<double>::max();
  int percentage = 100;
  int numberOfReruns = 1000;
  
  // Do calibration
    mitk::PivotCalibration::Pointer calibration = mitk::PivotCalibration::New();
    calibration->CalibrateUsingFilesInDirectories(
      matrixDirectoryName.toStdString(),
      residualError,
      *m_Matrix,
      percentage,
      numberOfReruns
      );

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      m_Controls->m_MatrixWidget->setValue(i, j, m_Matrix->GetElement(i, j));
    }
  }

  m_Controls->m_ResidualErrorDisplayLabel->setText(QString::number(residualError));

  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}

//-----------------------------------------------------------------------------
void PivotCalibrationView::OnSaveToFileButtonClicked()
{
  QString fileName = QFileDialog::getSaveFileName( NULL,
                                                   tr("Save Transform As ..."),
                                                   QDir::currentPath(),
                                                   "Matrix file (*.mat);;4x4 file (*.4x4);;Text file (*.txt);;All files (*.*)" );
  if (fileName.size() > 0)
  {
    SaveMatrixToFile(*m_Matrix, fileName);
  }
}

//-----------------------------------------------------------------------------
void PivotCalibrationView::SetFocus()
{
  m_Controls->m_MatrixDirectoryChooser->setFocus();
}


//-----------------------------------------------------------------------------
std::string PivotCalibrationView::GetViewID() const
{
  return VIEW_ID;
}

