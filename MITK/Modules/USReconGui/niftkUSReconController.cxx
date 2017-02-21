/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkUSReconController.h"
#include <niftkUltrasoundProcessing.h>
#include <Internal/niftkUSReconGUI.h>
#include <mitkIOUtil.h>
#include <niftkFileHelper.h>
#include <QMutex>
#include <QMutexLocker>
#include <QFuture>
#include <QFutureWatcher>
#include <QtConcurrentRun>
#include <QMessageBox>
#include <QFileDialog>

namespace niftk
{

class USReconControllerPrivate
{
  Q_DECLARE_PUBLIC(USReconController);
  USReconController* const q_ptr;

public:

  USReconControllerPrivate(USReconController* q);
  ~USReconControllerPrivate();

  USReconGUI*                       m_GUI;
  QString                           m_PreviousDirName;
  QString                           m_RecordingDirName;
  bool                              m_IsRecording;
  int                               m_ReconstructedId;
  mitk::DataNode::Pointer           m_CurrentImage;
  mitk::DataNode::Pointer           m_CurrentTracking;
  QMutex                            m_Lock;
  QFuture<void>                     m_BackgroundProcess;
  QFutureWatcher<void>              m_BackgroundProcessWatcher;
  niftk::TrackedImageData           m_TrackedImages;
};


//-----------------------------------------------------------------------------
USReconControllerPrivate::USReconControllerPrivate(USReconController* usreconController)
: q_ptr(usreconController)
, m_Lock(QMutex::Recursive)
, m_RecordingDirName("")
, m_IsRecording(false)
, m_ReconstructedId(0)
{
  Q_Q(USReconController);
}


//-----------------------------------------------------------------------------
USReconControllerPrivate::~USReconControllerPrivate()
{
}


//-----------------------------------------------------------------------------
USReconController::USReconController(IBaseView* view)
: BaseController(view)
, d_ptr(new USReconControllerPrivate(this))
{
}


//-----------------------------------------------------------------------------
USReconController::~USReconController()
{
  Q_D(USReconController);
  QMutexLocker locker(&d->m_Lock);

  bool ok = disconnect(&d->m_BackgroundProcessWatcher, SIGNAL(finished()), this, SLOT(OnBackgroundProcessFinished()));
  assert(ok);

  d->m_BackgroundProcessWatcher.waitForFinished();
}


//-----------------------------------------------------------------------------
BaseGUI* USReconController::CreateGUI(QWidget* parent)
{
  return new USReconGUI(parent);
}


//-----------------------------------------------------------------------------
void USReconController::SetupGUI(QWidget* parent)
{
  Q_D(USReconController);
  BaseController::SetupGUI(parent);
  d->m_GUI = dynamic_cast<USReconGUI*>(this->GetGUI());
  d->m_GUI->SetDataStorage(this->GetDataStorage());

  connect(d->m_GUI, SIGNAL(OnImageSelectionChanged(const mitk::DataNode*)), this, SLOT(OnImageSelectionChanged(const mitk::DataNode*)));
  connect(d->m_GUI, SIGNAL(OnTrackingSelectionChanged(const mitk::DataNode*)), this, SLOT(OnTrackingSelectionChanged(const mitk::DataNode*)));
  connect(d->m_GUI, SIGNAL(OnGrabPressed()), this, SLOT(OnGrabPressed()));
  connect(d->m_GUI, SIGNAL(OnClearDataPressed()), this, SLOT(OnClearDataPressed()));
  connect(d->m_GUI, SIGNAL(OnSaveDataPressed()), this, SLOT(OnSaveDataPressed()));
  connect(d->m_GUI, SIGNAL(OnLoadCalibrationPressed()), this, SLOT(OnLoadCalibrationPressed()));
  connect(d->m_GUI, SIGNAL(OnCalibratePressed()), this, SLOT(OnCalibratePressed()));
  connect(d->m_GUI, SIGNAL(OnReconstructPressed()), this, SLOT(OnReconstructPressed()));
  connect(&d->m_BackgroundProcessWatcher, SIGNAL(finished()), this, SLOT(OnBackgroundProcessFinished()));
}


//-----------------------------------------------------------------------------
void USReconController::OnImageSelectionChanged(const mitk::DataNode* node)
{
  Q_D(USReconController);
  QMutexLocker locker(&d->m_Lock);

  if (d->m_CurrentImage.IsNotNull() && d->m_CurrentImage.GetPointer() != node)
  {
    this->OnClearDataPressed();
  }
}


//-----------------------------------------------------------------------------
void USReconController::OnTrackingSelectionChanged(const mitk::DataNode* node)
{
  Q_D(USReconController);
  QMutexLocker locker(&d->m_Lock);

  if (d->m_CurrentTracking.IsNotNull() && d->m_CurrentTracking.GetPointer() != node)
  {
    this->OnClearDataPressed();
  }
}


//-----------------------------------------------------------------------------
void USReconController::SetRecordingStarted(const QString& recordingDir)
{
  Q_D(USReconController);
  QMutexLocker locker(&d->m_Lock);

  d->m_RecordingDirName = recordingDir;
  d->m_IsRecording = true;
}


//-----------------------------------------------------------------------------
void USReconController::SetRecordingStopped()
{
  Q_D(USReconController);
  QMutexLocker locker(&d->m_Lock);

  d->m_IsRecording = false;
}


//-----------------------------------------------------------------------------
void USReconController::CaptureImages()
{
  Q_D(USReconController);
  QMutexLocker locker(&d->m_Lock);

  mitk::DataNode::Pointer imageNode = d->m_GUI->GetImageNode();
  mitk::DataNode::Pointer trackingNode = d->m_GUI->GetTrackingNode();

  if (imageNode.IsNotNull() && trackingNode.IsNotNull())
  {
    mitk::Image::Pointer clonedImage = dynamic_cast<mitk::Image*>(imageNode->GetData())->Clone();
    niftk::CoordinateAxesData::Pointer clonedTransform = dynamic_cast<niftk::CoordinateAxesData*>(trackingNode->GetData())->Clone();
    d->m_TrackedImages.push_back(TrackedImage(clonedImage, clonedTransform));
    d->m_GUI->SetNumberOfFramesLabel(d->m_TrackedImages.size());
  }
}


//-----------------------------------------------------------------------------
void USReconController::OnClearDataPressed()
{
  Q_D(USReconController);
  QMutexLocker locker(&d->m_Lock);

  MITK_INFO << "Clearing all previously collected image and tracking data";

  d->m_TrackedImages.clear();

  MITK_INFO << "Clearing all previously collected image and tracking data - DONE";
}


//-----------------------------------------------------------------------------
void USReconController::Update()
{
  Q_D(USReconController);

  if (d->m_IsRecording)
  {
    QMutexLocker locker(&d->m_Lock);
    this->CaptureImages();
  }
}


//-----------------------------------------------------------------------------
void USReconController::OnGrabPressed()
{
  Q_D(USReconController);
  QMutexLocker locker(&d->m_Lock);

  if (!d->m_IsRecording)
  {
    this->CaptureImages();
  }
}


//-----------------------------------------------------------------------------
void USReconController::OnSaveDataPressed()
{
  Q_D(USReconController);

  if (d->m_TrackedImages.empty())
  {
    QMessageBox msgBox;
    msgBox.setText("No data!");
    msgBox.setInformativeText("No data has been collected. Please grab some, or start recording.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }

  QString previous = d->m_PreviousDirName;
  if (previous.isEmpty())
  {
    previous = d->m_RecordingDirName;
  }

  QString dirName = QFileDialog::getExistingDirectory(d->m_GUI->GetParent(),
      tr("Output directory"), previous);

  if (!dirName.isEmpty())
  {
    bool savedSomething = false;

    for (int i = 0; i < d->m_TrackedImages.size(); i++)
    {
      mitk::DataNode::Pointer imageNode = d->m_GUI->GetImageNode();
      if (imageNode.IsNotNull())
      {
        std::ostringstream fileName;
        fileName << dirName.toStdString()
                 << niftk::GetFileSeparator()
                 << "image-"
                 << i
                 << ".png";

        mitk::IOUtil::Save(imageNode->GetData(), fileName.str());
        savedSomething = true;
      }
      mitk::DataNode::Pointer trackingNode = d->m_GUI->GetTrackingNode();
      if (trackingNode.IsNotNull())
      {
        std::ostringstream fileName;
        fileName << dirName.toStdString()
                 << niftk::GetFileSeparator()
                 << "tracker-"
                 << i
                 << ".4x4";

        mitk::IOUtil::Save(trackingNode->GetData(), fileName.str());
        savedSomething = true;
      }
    }
    if (savedSomething)
    {
      previous = dirName;
    }
    else
    {
      QMessageBox msgBox;
      msgBox.setText("Failed to save!");
      msgBox.setInformativeText("Failed to save tracked images, please check console.");
      msgBox.setStandardButtons(QMessageBox::Ok);
      msgBox.setDefaultButton(QMessageBox::Ok);
      msgBox.exec();
      return;
    }
  }
}


//-----------------------------------------------------------------------------
void USReconController::OnLoadCalibrationPressed()
{
  MITK_INFO << "OnLoadCalibrationPressed()";
}


//-----------------------------------------------------------------------------
void USReconController::OnCalibratePressed()
{
  MITK_INFO << "OnCalibratePressed()";
}


//-----------------------------------------------------------------------------
void USReconController::OnReconstructPressed()
{
  this->DoReconstruction();
}


//-----------------------------------------------------------------------------
void USReconController::DoReconstruction()
{
  Q_D(USReconController);

  if (!d->m_BackgroundProcess.isRunning())
  {
    MITK_INFO << "Launching Ultrasound Reconstruction.";

    QMutexLocker locker(&d->m_Lock);
    d->m_BackgroundProcess = QtConcurrent::run(this, &niftk::USReconController::DoReconstructionInBackground);
    d->m_BackgroundProcessWatcher.setFuture(d->m_BackgroundProcess);
    d->m_GUI->SetEnableButtons(false);
  }
}


//-----------------------------------------------------------------------------
void USReconController::DoReconstructionInBackground()
{
  Q_D(USReconController);
  QMutexLocker locker(&d->m_Lock);

  MITK_INFO << "Running Ultrasound Reconstruction in Background";

  mitk::Image::Pointer newImage = niftk::DoUltrasoundReconstruction(d->m_TrackedImages);
  if (newImage.IsNotNull())
  {
    // Temporary: Save image straight to disk.
    // We can use any valid file type recognised by MITK/ITK file IO.

    std::ostringstream imageName;
    imageName << "reconstructed-"
              << d->m_ReconstructedId;

    mitk::DataNode::Pointer newNode = mitk::DataNode::New();
    newNode->SetData(newImage);
    newNode->SetName(imageName.str());
    this->GetDataStorage()->Add(newNode);

    // Increase the volume number, so we can keep grabbing new volumes.
    d->m_ReconstructedId++;
  }
  else
  {
    MITK_WARN << "Ultrasound Reconstruction is returning NULL???";
  }
}


//-----------------------------------------------------------------------------
void USReconController::OnBackgroundProcessFinished()
{
  Q_D(USReconController);
  QMutexLocker locker(&d->m_Lock);

  MITK_INFO << "Re-enabling Ultrasound Reconstruction buttons";

  d->m_GUI->SetEnableButtons(true);
}

} // end namespace
