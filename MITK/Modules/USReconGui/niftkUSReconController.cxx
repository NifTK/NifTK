/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkUSReconController.h"
#include <niftkUSReconstructor.h>
#include <Internal/niftkUSReconGUI.h>
#include <mitkRenderingManager.h>
#include <mitkIOUtil.h>
#include <niftkFileHelper.h>
#include <QMutex>
#include <QMutexLocker>
#include <QFuture>
#include <QFutureWatcher>
#include <QtConcurrentRun>
#include <QMessageBox>

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
  QString                           m_OutputDirName;
  QString                           m_RecordingDirName;
  bool                              m_IsRecording;
  bool                              m_DumpEachFrameWhileRecording;
  bool                              m_DumpEachReconstructedVolume;
  int                               m_FrameId;
  int                               m_ReconstructedId;
  mitk::DataNode::Pointer           m_CurrentImage;
  mitk::DataNode::Pointer           m_CurrentTracking;
  QMutex                            m_Lock;
  QFuture<void>                     m_BackgroundProcess;
  QFutureWatcher<void>              m_BackgroundProcessWatcher;
  niftk::USReconstructor::Pointer   m_Reconstructor;
};


//-----------------------------------------------------------------------------
USReconControllerPrivate::USReconControllerPrivate(USReconController* usreconController)
: q_ptr(usreconController)
, m_Lock(QMutex::Recursive)
, m_OutputDirName("")
, m_RecordingDirName("")
, m_IsRecording(false)
, m_DumpEachFrameWhileRecording(false)
, m_DumpEachReconstructedVolume(false)
, m_FrameId(0)
, m_ReconstructedId(0)
{
  Q_Q(USReconController);
  m_Reconstructor = niftk::USReconstructor::New();
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
  connect(d->m_GUI, SIGNAL(OnReconstructPressed()), this, SLOT(OnReconstructPressed()));
  connect(d->m_GUI, SIGNAL(OnClearDataPressed()), this, SLOT(OnClearDataPressed()));

  connect(&d->m_BackgroundProcessWatcher, SIGNAL(finished()), this, SLOT(OnBackgroundProcessFinished()));
}


//-----------------------------------------------------------------------------
void USReconController::SetOutputDirName(const QString& outputDir)
{
  Q_D(USReconController);
  QMutexLocker locker(&d->m_Lock);

  d->m_OutputDirName = outputDir;
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
void USReconController::SaveImages(const QString& dirName)
{
  Q_D(USReconController);

  if(!dirName.isEmpty())
  {
    bool savedSomething = false;
    mitk::DataNode::Pointer imageNode = d->m_GUI->GetImageNode();
    if (imageNode.IsNotNull())
    {
      std::ostringstream fileName;
      fileName << dirName.toStdString()
               << niftk::GetFileSeparator()
               << "image-"
               << d->m_FrameId
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
               << d->m_FrameId
               << ".4x4";

      mitk::IOUtil::Save(trackingNode->GetData(), fileName.str());
      savedSomething = true;
    }
    if (savedSomething)
    {
      d->m_FrameId++;
    }
  }
}


//-----------------------------------------------------------------------------
void USReconController::CaptureImages()
{
  Q_D(USReconController);

  mitk::DataNode::Pointer imageNode = d->m_GUI->GetImageNode();
  mitk::DataNode::Pointer trackingNode = d->m_GUI->GetTrackingNode();

  if (imageNode.IsNotNull() && trackingNode.IsNotNull())
  {
    d->m_Reconstructor->AddPair(dynamic_cast<mitk::Image*>(imageNode->GetData()),
                                dynamic_cast<niftk::CoordinateAxesData*>(trackingNode->GetData())
                                );
  }
}


//-----------------------------------------------------------------------------
void USReconController::Update()
{
  Q_D(USReconController);

  if (d->m_IsRecording)
  {
    QMutexLocker locker(&d->m_Lock);
    this->CaptureImages();

    if (d->m_DumpEachFrameWhileRecording && !d->m_RecordingDirName.isEmpty())
    {
      this->SaveImages(d->m_RecordingDirName);
    }
  }
}


//-----------------------------------------------------------------------------
void USReconController::OnClearDataPressed()
{
  Q_D(USReconController);
  QMutexLocker locker(&d->m_Lock);

  MITK_INFO << "Clearing all previously collected image and tracking data";
  d->m_Reconstructor->ClearData();
  MITK_INFO << "Clearing all previously collected image and tracking data - DONE";
}


//-----------------------------------------------------------------------------
void USReconController::OnGrabPressed()
{
  Q_D(USReconController);
  QMutexLocker locker(&d->m_Lock);

  QString dirName = d->m_OutputDirName;

  if (dirName.isEmpty())
  {
    QMessageBox msgBox;
    msgBox.setText("An Error Occurred.");
    msgBox.setInformativeText("The output directory name is empty - check preferences.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }
  else
  {
    // Normally, the manual grab function is just for calibration, so we save the data.
    this->SaveImages(dirName);

    // But you could use the manual grab to do reconstruction!
    // So, here we optionally collect them up, if we are not already recording.
    if (!d->m_IsRecording)
    {
      this->CaptureImages();
    }
  }
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

  // Do some reconstruction, which happens in a background thread, and always generates a new image.
  mitk::Image::Pointer newImage = d->m_Reconstructor->DoReconstruction();

  if (newImage.IsNotNull())
  {
    // Temporary: Save image straight to disk.
    // We can use any valid file type recognised by MITK/ITK file IO.

    std::ostringstream imageName;
    imageName << "reconstructed-"
              << d->m_ReconstructedId
              << ".nii.gz";

    std::ostringstream fileName;
    if (d->m_IsRecording)
    {
      fileName << d->m_RecordingDirName.toStdString();
    }
    else
    {
      fileName << d->m_OutputDirName.toStdString();
    }
    fileName << niftk::GetFileSeparator()
             << imageName.str();

    if(d->m_DumpEachReconstructedVolume)
    {
      mitk::IOUtil::Save(newImage, fileName.str());
    }

    // Here we add it to data-storage, so the user can control
    // if/when they want to save it, and what file name etc.
    mitk::DataNode::Pointer newNode = mitk::DataNode::New();
    newNode->SetData(newImage);
    newNode->SetName(imageName.str());
    this->GetDataStorage()->Add(newNode);

    // Increase the volume number, so we can keep grabbing.
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


//-----------------------------------------------------------------------------
void USReconController::SetDumpEachFrameWhileRecording(bool doIt)
{
  Q_D(USReconController);
  QMutexLocker locker(&d->m_Lock);

  d->m_DumpEachFrameWhileRecording = doIt;
}


//-----------------------------------------------------------------------------
void USReconController::SetDumpEachReconstructedVolume(bool doIt)
{
  Q_D(USReconController);
  QMutexLocker locker(&d->m_Lock);

  d->m_DumpEachReconstructedVolume = doIt;
}

} // end namespace
