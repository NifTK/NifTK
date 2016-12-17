/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCaffeSegController.h"
#include <niftkCaffeFCNSegmentor.h>
#include <Internal/niftkCaffeSegGUI.h>
#include <mitkRenderingManager.h>
#include <QMutex>
#include <QMutexLocker>
#include <QFuture>
#include <QFutureWatcher>
#include <QtConcurrentRun>
#include <QMessageBox>

namespace niftk
{

class CaffeSegControllerPrivate
{
  Q_DECLARE_PUBLIC(CaffeSegController);
  CaffeSegController* const q_ptr;

public:

  CaffeSegControllerPrivate(CaffeSegController* q);
  ~CaffeSegControllerPrivate();

  CaffeSegGUI*                      m_GUI;
  std::string                       m_NetworkDescriptionFileName;
  std::string                       m_NetworkWeightsFileName;
  bool                              m_DoTranspose;
  std::string                       m_InputLayerName;
  std::string                       m_OutputBlobName;
  int                               m_GPUDevice;
  bool                              m_IsUpdatingManually;
  QMutex                            m_Lock;
  QFuture<void>                     m_BackgroundProcess;
  QFutureWatcher<void>              m_BackgroundProcessWatcher;

  niftk::CaffeFCNSegmentor::Pointer m_Segmentors[2];
  mitk::DataNode*                   m_DataNodes[2];
  mitk::DataNode::Pointer           m_SegmentedNodes[2];
};


//-----------------------------------------------------------------------------
CaffeSegControllerPrivate::CaffeSegControllerPrivate(CaffeSegController* caffeSegController)
: q_ptr(caffeSegController)
, m_IsUpdatingManually(false)
, m_Lock(QMutex::Recursive)
{
  Q_Q(CaffeSegController);
  for (int i = 0; i < 2; i++)
  {
    m_Segmentors[i] = nullptr;
    m_DataNodes[i] = nullptr;
    m_SegmentedNodes[i] = nullptr;
  }
}


//-----------------------------------------------------------------------------
CaffeSegControllerPrivate::~CaffeSegControllerPrivate()
{
}


//-----------------------------------------------------------------------------
CaffeSegController::CaffeSegController(IBaseView* view)
: BaseController(view)
, d_ptr(new CaffeSegControllerPrivate(this))
{
}


//-----------------------------------------------------------------------------
CaffeSegController::~CaffeSegController()
{
  Q_D(CaffeSegController);
  QMutexLocker locker(&d->m_Lock);

  bool ok = disconnect(&d->m_BackgroundProcessWatcher, SIGNAL(finished()), this, SLOT(OnBackgroundProcessFinished()));
  assert(ok);
  d->m_BackgroundProcessWatcher.waitForFinished();

  for (int i = 0; i < 2; i++)
  {
    if (d->m_SegmentedNodes[i].IsNotNull())
    {
      this->GetDataStorage()->Remove(d->m_SegmentedNodes[i]);
    }
  }
}


//-----------------------------------------------------------------------------
BaseGUI* CaffeSegController::CreateGUI(QWidget* parent)
{
  return new CaffeSegGUI(parent);
}


//-----------------------------------------------------------------------------
void CaffeSegController::SetupGUI(QWidget* parent)
{
  Q_D(CaffeSegController);
  BaseController::SetupGUI(parent);
  d->m_GUI = dynamic_cast<CaffeSegGUI*>(this->GetGUI());
  d->m_GUI->SetDataStorage(this->GetDataStorage());
  connect(d->m_GUI, SIGNAL(OnLeftSelectionChanged(const mitk::DataNode*)), this, SLOT(OnLeftSelectionChanged(const mitk::DataNode*)));
  connect(d->m_GUI, SIGNAL(OnRightSelectionChanged(const mitk::DataNode*)), this, SLOT(OnRightSelectionChanged(const mitk::DataNode*)));
  connect(d->m_GUI, SIGNAL(OnDoItNowPressed()), this, SLOT(OnDoItNowPressed()));
  connect(d->m_GUI, SIGNAL(OnManualUpdateClicked(bool)), this, SLOT(OnManualUpdateClicked(bool)));
  connect(d->m_GUI, SIGNAL(OnAutomaticUpdateClicked(bool)), this, SLOT(OnAutomaticUpdateClicked(bool)));
  connect(&d->m_BackgroundProcessWatcher, SIGNAL(finished()), this, SLOT(OnBackgroundProcessFinished()));
}


//-----------------------------------------------------------------------------
void CaffeSegController::SetNetworkDescriptionFileName(const QString& description)
{
  Q_D(CaffeSegController);
  QMutexLocker locker(&d->m_Lock);

  d->m_NetworkDescriptionFileName = description.toStdString();
}


//-----------------------------------------------------------------------------
void CaffeSegController::SetNetworkWeightsFileName(const QString& weights)
{
  Q_D(CaffeSegController);
  QMutexLocker locker(&d->m_Lock);

  d->m_NetworkWeightsFileName = weights.toStdString();
}


//-----------------------------------------------------------------------------
void CaffeSegController::SetDoTranspose(const bool& doTranspose)
{
  Q_D(CaffeSegController);
  QMutexLocker locker(&d->m_Lock);

  d->m_DoTranspose = doTranspose;
}


//-----------------------------------------------------------------------------
void CaffeSegController::SetInputLayerName(const QString& inputLayer)
{
  Q_D(CaffeSegController);
  QMutexLocker locker(&d->m_Lock);

  d->m_InputLayerName = inputLayer.toStdString();
}


//-----------------------------------------------------------------------------
void CaffeSegController::SetOutputBlobName(const QString& outputBlob)
{
  Q_D(CaffeSegController);
  QMutexLocker locker(&d->m_Lock);

  d->m_OutputBlobName = outputBlob.toStdString();
}


//-----------------------------------------------------------------------------
void CaffeSegController::SetGPUDevice(const int& deviceId)
{
  Q_D(CaffeSegController);
  QMutexLocker locker(&d->m_Lock);

  d->m_GPUDevice = deviceId;
}


//-----------------------------------------------------------------------------
void CaffeSegController::OnManualUpdateClicked(bool isChecked)
{
  Q_D(CaffeSegController);
  QMutexLocker locker(&d->m_Lock);

  d->m_IsUpdatingManually = isChecked;
}


//-----------------------------------------------------------------------------
void CaffeSegController::OnAutomaticUpdateClicked(bool isChecked)
{
  Q_D(CaffeSegController);
  QMutexLocker locker(&d->m_Lock);

  d->m_IsUpdatingManually = !isChecked;
}


//-----------------------------------------------------------------------------
void CaffeSegController::OnDoItNowPressed()
{
  Q_D(CaffeSegController);

  if (d->m_IsUpdatingManually)
  {
    this->InternalUpdate();
  }
}


//-----------------------------------------------------------------------------
void CaffeSegController::Update()
{
  Q_D(CaffeSegController);

  if (!d->m_IsUpdatingManually)
  {
    this->InternalUpdate();
  }
}


//-----------------------------------------------------------------------------
void CaffeSegController::ClearNode(const int& i)
{
  Q_D(CaffeSegController);
  QMutexLocker locker(&d->m_Lock);

  d->m_Segmentors[i] = nullptr;
  d->m_DataNodes[i] = nullptr;

  if (this->GetDataStorage()->Exists(d->m_SegmentedNodes[i]))
  {
    this->GetDataStorage()->Remove(d->m_SegmentedNodes[i]);
    d->m_SegmentedNodes[i] = nullptr;
  }
}


//-----------------------------------------------------------------------------
void CaffeSegController::OnNodeRemoved(const mitk::DataNode* node)
{
  Q_D(CaffeSegController);
  QMutexLocker locker(&d->m_Lock);

  for (int i = 0; i < 2; i++)
  {
    if (node == d->m_DataNodes[i] || node == d->m_SegmentedNodes[i])
    {
      this->ClearNode(i);
    }
  }
}


//-----------------------------------------------------------------------------
void CaffeSegController::SelectionChanged(const mitk::DataNode* node, const int& i)
{
  Q_D(CaffeSegController);
  QMutexLocker locker(&d->m_Lock);

  if (d->m_NetworkDescriptionFileName.empty() || d->m_NetworkWeightsFileName.empty())
  {
    QMessageBox::warning(NULL,"Preferences Not Set","Have you set the preferences to specify the network model and weights?");
    return;
  }

  if (node == nullptr)
  {
    this->ClearNode(i);
  }
  else if (node != d->m_DataNodes[i])
  {
    mitk::Image::Pointer im = dynamic_cast<mitk::Image*>(node->GetData());
    if (im.IsNotNull())
    {

      d->m_DataNodes[i] = const_cast<mitk::DataNode*>(node);

      d->m_Segmentors[i] = niftk::CaffeFCNSegmentor::New(d->m_NetworkDescriptionFileName,
                                                         d->m_NetworkWeightsFileName,
                                                         d->m_InputLayerName,
                                                         d->m_OutputBlobName,
                                                         d->m_GPUDevice
                                                        );
      d->m_Segmentors[i]->SetTransposingMode(d->m_DoTranspose);

      if (d->m_SegmentedNodes[i].IsNotNull())
      {
        this->GetDataStorage()->Remove(d->m_SegmentedNodes[i]);
      }

      mitk::PixelType pt = mitk::MakeScalarPixelType<unsigned char>();
      mitk::Image::Pointer op = mitk::Image::New();
      unsigned int dim[] = { im->GetDimension(0), im->GetDimension(1) };
      op->Initialize( pt, 2, dim);

      mitk::DataNode::Pointer segNode = mitk::DataNode::New();
      segNode->SetName(d->m_DataNodes[i]->GetName() + "_Mask");
      segNode->SetBoolProperty("binary", true);
      segNode->SetBoolProperty("outline binary", true);
      segNode->SetData(op);

      d->m_SegmentedNodes[i] = segNode;
      this->GetDataStorage()->Add(d->m_SegmentedNodes[i]);

      this->InternalUpdate();
    }
  }
}


//-----------------------------------------------------------------------------
void CaffeSegController::OnLeftSelectionChanged(const mitk::DataNode* node)
{
  this->SelectionChanged(node, 0);
}


//-----------------------------------------------------------------------------
void CaffeSegController::OnRightSelectionChanged(const mitk::DataNode* node)
{
  this->SelectionChanged(node, 1);
}


//-----------------------------------------------------------------------------
void CaffeSegController::InternalUpdate()
{
  Q_D(CaffeSegController);

  if (!d->m_BackgroundProcess.isRunning())
  {
    QMutexLocker locker(&d->m_Lock);
    d->m_BackgroundProcess = QtConcurrent::run(this, &niftk::CaffeSegController::InternalUpdateBackground);
  }
}


//-----------------------------------------------------------------------------
void CaffeSegController::InternalUpdateBackground()
{
  Q_D(CaffeSegController);
  QMutexLocker locker(&d->m_Lock);

  for (int i = 0; i < 2; i++)
  {
    if (   d->m_Segmentors[i].IsNotNull()
        && d->m_DataNodes[i] != nullptr)
    {
      mitk::Image::Pointer im1 = dynamic_cast<mitk::Image*>(d->m_DataNodes[i]->GetData());
      mitk::Image::Pointer im2 = dynamic_cast<mitk::Image*>(d->m_SegmentedNodes[i]->GetData());

      if (im1.IsNotNull() && im2.IsNotNull())
      {
        d->m_Segmentors[i]->Segment(im1, im2);
        d->m_SegmentedNodes[i]->Modified();
      }
    }
  }
}


//-----------------------------------------------------------------------------
void CaffeSegController::OnBackgroundProcessFinished()
{
  Q_D(CaffeSegController);

  // Nothing to do right now - placeholder.
}

} // end namespace
