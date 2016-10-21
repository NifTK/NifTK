/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkDistanceMeasurerController.h"
#include <Internal/niftkDistanceMeasurerGUI.h>
#include <niftkDistanceFromCamera.h>
#include <QMutex>
#include <QMutexLocker>
#include <QFuture>
#include <QFutureWatcher>
#include <QtConcurrentRun>

namespace niftk
{

class DistanceMeasurerControllerPrivate
{
  Q_DECLARE_PUBLIC(DistanceMeasurerController);
  DistanceMeasurerController* const q_ptr;

public:

  DistanceMeasurerControllerPrivate(DistanceMeasurerController* q);
  ~DistanceMeasurerControllerPrivate();

  DistanceMeasurerGUI*               m_GUI;
  mitk::DataNode*                    m_LeftImage;
  mitk::DataNode*                    m_LeftMask;
  mitk::DataNode*                    m_RightImage;
  mitk::DataNode*                    m_RightMask;
  QMutex                             m_Lock;
  niftk::DistanceFromCamera::Pointer m_DistanceFromCamera;
  QFuture<double>                    m_BackgroundProcess;
  QFutureWatcher<double>             m_BackgroundProcessWatcher;

};


//-----------------------------------------------------------------------------
DistanceMeasurerControllerPrivate::DistanceMeasurerControllerPrivate(DistanceMeasurerController* distanceMeasurerController)
: q_ptr(distanceMeasurerController)
, m_LeftImage(nullptr)
, m_LeftMask(nullptr)
, m_RightImage(nullptr)
, m_RightMask(nullptr)
, m_Lock(QMutex::Recursive)
{
  Q_Q(DistanceMeasurerController);
  m_DistanceFromCamera = niftk::DistanceFromCamera::New();
}


//-----------------------------------------------------------------------------
DistanceMeasurerControllerPrivate::~DistanceMeasurerControllerPrivate()
{
  m_BackgroundProcessWatcher.waitForFinished();
}


//-----------------------------------------------------------------------------
DistanceMeasurerController::DistanceMeasurerController(IBaseView* view)
: BaseController(view)
, d_ptr(new DistanceMeasurerControllerPrivate(this))
{
}


//-----------------------------------------------------------------------------
DistanceMeasurerController::~DistanceMeasurerController()
{
  Q_D(DistanceMeasurerController);
}


//-----------------------------------------------------------------------------
BaseGUI* DistanceMeasurerController::CreateGUI(QWidget* parent)
{
  return new DistanceMeasurerGUI(parent);
}


//-----------------------------------------------------------------------------
void DistanceMeasurerController::SetupGUI(QWidget* parent)
{
  Q_D(DistanceMeasurerController);
  BaseController::SetupGUI(parent);
  d->m_GUI = dynamic_cast<DistanceMeasurerGUI*>(this->GetGUI());
  d->m_GUI->SetDataStorage(this->GetDataStorage());

  bool ok = false;
  ok = connect(d->m_GUI, SIGNAL(LeftImageSelectionChanged(const mitk::DataNode*)), this, SLOT(OnLeftImageSelectionChanged(const mitk::DataNode*)));
  assert(ok);
  ok = connect(d->m_GUI, SIGNAL(LeftMaskSelectionChanged(const mitk::DataNode*)), this, SLOT(OnLeftMaskSelectionChanged(const mitk::DataNode*)));
  assert(ok);
  ok = connect(d->m_GUI, SIGNAL(RightImageSelectionChanged(const mitk::DataNode*)), this, SLOT(OnRightImageSelectionChanged(const mitk::DataNode*)));
  assert(ok);
  ok = connect(d->m_GUI, SIGNAL(RightMaskSelectionChanged(const mitk::DataNode*)), this, SLOT(OnRightMaskSelectionChanged(const mitk::DataNode*)));
  assert(ok);
  ok = connect(&d->m_BackgroundProcessWatcher, SIGNAL(finished()), this, SLOT(OnBackgroundProcessFinished()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void DistanceMeasurerController::OnNodeRemoved(const mitk::DataNode* node)
{
  Q_D(DistanceMeasurerController);
  QMutexLocker locker(&d->m_Lock);

  if (   node == d->m_LeftImage
      || node == d->m_LeftMask
      || node == d->m_RightImage
      || node == d->m_RightMask
      )
  {
    d->m_GUI->Reset();
  }
}


//-----------------------------------------------------------------------------
void DistanceMeasurerController::OnLeftImageSelectionChanged(const mitk::DataNode* node)
{
  Q_D(DistanceMeasurerController);
  QMutexLocker locker(&d->m_Lock);

  d->m_LeftImage = const_cast<mitk::DataNode*>(node);
}


//-----------------------------------------------------------------------------
void DistanceMeasurerController::OnLeftMaskSelectionChanged(const mitk::DataNode* node)
{
  Q_D(DistanceMeasurerController);
  QMutexLocker locker(&d->m_Lock);

  d->m_LeftMask = const_cast<mitk::DataNode*>(node);
}


//-----------------------------------------------------------------------------
void DistanceMeasurerController::OnRightImageSelectionChanged(const mitk::DataNode* node)
{
  Q_D(DistanceMeasurerController);
  QMutexLocker locker(&d->m_Lock);

  d->m_RightImage = const_cast<mitk::DataNode*>(node);
}


//-----------------------------------------------------------------------------
void DistanceMeasurerController::OnRightMaskSelectionChanged(const mitk::DataNode* node)
{
  Q_D(DistanceMeasurerController);
  QMutexLocker locker(&d->m_Lock);

  d->m_RightMask = const_cast<mitk::DataNode*>(node);
}


//-----------------------------------------------------------------------------
void DistanceMeasurerController::Update()
{
  Q_D(DistanceMeasurerController);
  QMutexLocker locker(&d->m_Lock);

  if (d->m_LeftImage != nullptr && d->m_RightImage != nullptr && !d->m_BackgroundProcess.isRunning())
  {
    d->m_BackgroundProcess = QtConcurrent::run(this, &DistanceMeasurerController::InternalUpdate);
    d->m_BackgroundProcessWatcher.setFuture(d->m_BackgroundProcess);
  }
}


//-----------------------------------------------------------------------------
double DistanceMeasurerController::InternalUpdate()
{
  Q_D(DistanceMeasurerController);
  QMutexLocker locker(&d->m_Lock);

  double distanceInMillimetres = d->m_DistanceFromCamera->GetDistance(d->m_LeftImage,
                                                                      d->m_RightImage,
                                                                      d->m_LeftMask,
                                                                      d->m_RightMask
                                                                     );

  return distanceInMillimetres;
}


//-----------------------------------------------------------------------------
void DistanceMeasurerController::OnBackgroundProcessFinished()
{
  Q_D(DistanceMeasurerController);
  QMutexLocker locker(&d->m_Lock);

  double distanceInMillimetres = d->m_BackgroundProcessWatcher.result();
  d->m_GUI->SetDistance(distanceInMillimetres / 10.0);
}

} // end namespace
