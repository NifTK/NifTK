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
#include <itkFastMutexLock.h>

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
  niftk::CaffeFCNSegmentor::Pointer m_LeftSegmentor;
  mitk::DataNode*                   m_LeftDataNode;
  mitk::DataNode::Pointer           m_LeftSegmentedNode;
  niftk::CaffeFCNSegmentor::Pointer m_RightSegmentor;
  mitk::DataNode*                   m_RightDataNode;
  mitk::DataNode::Pointer           m_RightSegmentedNode;
  bool                              m_IsUpdatingManually;
  itk::FastMutexLock::Pointer       m_Mutex;
};


//-----------------------------------------------------------------------------
CaffeSegControllerPrivate::CaffeSegControllerPrivate(CaffeSegController* caffeSegController)
: q_ptr(caffeSegController)
, m_LeftSegmentor(nullptr)
, m_LeftDataNode(nullptr)
, m_LeftSegmentedNode(nullptr)
, m_RightSegmentor(nullptr)
, m_RightDataNode(nullptr)
, m_RightSegmentedNode(nullptr)
, m_IsUpdatingManually(false)
, m_Mutex(itk::FastMutexLock::New())
{
  Q_Q(CaffeSegController);
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
  if (d->m_LeftSegmentedNode.IsNotNull())
  {
    this->GetDataStorage()->Remove(d->m_LeftSegmentedNode);
  }
  if (d->m_RightSegmentedNode.IsNotNull())
  {
    this->GetDataStorage()->Remove(d->m_RightSegmentedNode);
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
}


//-----------------------------------------------------------------------------
void CaffeSegController::SetNetworkDescriptionFileName(const std::string& description)
{
  Q_D(CaffeSegController);
  itk::MutexLockHolder<itk::FastMutexLock> lock(*(d->m_Mutex));

  d->m_NetworkDescriptionFileName = description;
}


//-----------------------------------------------------------------------------
void CaffeSegController::SetNetworkWeightsFileName(const std::string& weights)
{
  Q_D(CaffeSegController);
  itk::MutexLockHolder<itk::FastMutexLock> lock(*(d->m_Mutex));

  d->m_NetworkWeightsFileName = weights;
}


//-----------------------------------------------------------------------------
void CaffeSegController::OnManualUpdateClicked(bool isChecked)
{
  Q_D(CaffeSegController);
  itk::MutexLockHolder<itk::FastMutexLock> lock(*(d->m_Mutex));

  d->m_IsUpdatingManually = isChecked;
}


//-----------------------------------------------------------------------------
void CaffeSegController::OnAutomaticUpdateClicked(bool isChecked)
{
  Q_D(CaffeSegController);
  itk::MutexLockHolder<itk::FastMutexLock> lock(*(d->m_Mutex));

  d->m_IsUpdatingManually = !isChecked;
}


//-----------------------------------------------------------------------------
void CaffeSegController::OnDoItNowPressed()
{
  Q_D(CaffeSegController);
  itk::MutexLockHolder<itk::FastMutexLock> lock(*(d->m_Mutex));

  if (d->m_IsUpdatingManually)
  {
    this->InternalUpdate();
  }
}


//-----------------------------------------------------------------------------
void CaffeSegController::Update()
{
  Q_D(CaffeSegController);
  itk::MutexLockHolder<itk::FastMutexLock> lock(*(d->m_Mutex));

  if (!d->m_IsUpdatingManually)
  {
    this->InternalUpdate();
  }
}


//-----------------------------------------------------------------------------
void CaffeSegController::OnLeftSelectionChanged(const mitk::DataNode* node)
{
  Q_D(CaffeSegController);
  itk::MutexLockHolder<itk::FastMutexLock> lock(*(d->m_Mutex));

  if (node != nullptr
      && node != d->m_LeftDataNode
      && !(d->m_NetworkDescriptionFileName.empty())
      && !(d->m_NetworkWeightsFileName.empty()))
  {
    mitk::Image::Pointer im = dynamic_cast<mitk::Image*>(node->GetData());
    if (im.IsNotNull())
    {
      d->m_LeftSegmentor = niftk::CaffeFCNSegmentor::New(d->m_NetworkDescriptionFileName,
                                                         d->m_NetworkWeightsFileName
                                                        );

      if (d->m_LeftSegmentedNode.IsNotNull())
      {
        this->GetDataStorage()->Remove(d->m_LeftSegmentedNode);
      }

      mitk::PixelType pt = mitk::MakeScalarPixelType<unsigned char>();
      mitk::Image::Pointer op = mitk::Image::New();
      unsigned int dim[] = { im->GetDimension(0), im->GetDimension(1) };
      op->Initialize( pt, 2, dim);

      mitk::DataNode::Pointer segNode = mitk::DataNode::New();
      segNode->SetName(d->m_LeftSegmentedNode->GetName() + "_Mask");
      segNode->SetData(op);

      d->m_LeftSegmentedNode = segNode;
      d->m_LeftDataNode = const_cast<mitk::DataNode*>(node);
    }
  }
}


//-----------------------------------------------------------------------------
void CaffeSegController::OnRightSelectionChanged(const mitk::DataNode* node)
{
  Q_D(CaffeSegController);
  itk::MutexLockHolder<itk::FastMutexLock> lock(*(d->m_Mutex));

  if (node != nullptr
      && node != d->m_RightDataNode
      && !(d->m_NetworkDescriptionFileName.empty())
      && !(d->m_NetworkWeightsFileName.empty()))
  {
    mitk::Image::Pointer im = dynamic_cast<mitk::Image*>(node->GetData());
    if (im.IsNotNull())
    {
      d->m_RightSegmentor = niftk::CaffeFCNSegmentor::New(d->m_NetworkDescriptionFileName,
                                                          d->m_NetworkWeightsFileName
                                                         );

      if (d->m_RightSegmentedNode.IsNotNull())
      {
        this->GetDataStorage()->Remove(d->m_RightSegmentedNode);
      }

      mitk::PixelType pt = mitk::MakeScalarPixelType<unsigned char>();
      mitk::Image::Pointer op = mitk::Image::New();
      unsigned int dim[] = { im->GetDimension(0), im->GetDimension(1) };
      op->Initialize( pt, 2, dim);

      mitk::DataNode::Pointer segNode = mitk::DataNode::New();
      segNode->SetName(d->m_RightSegmentedNode->GetName() + "_Mask");
      segNode->SetData(op);

      d->m_RightSegmentedNode = segNode;
      d->m_RightDataNode = const_cast<mitk::DataNode*>(node);
    }
  }
}


//-----------------------------------------------------------------------------
void CaffeSegController::InternalUpdate()
{
  Q_D(CaffeSegController);
  itk::MutexLockHolder<itk::FastMutexLock> lock(*(d->m_Mutex));

  if (   d->m_LeftSegmentor.IsNotNull()
      && d->m_LeftDataNode != nullptr)
  {
    mitk::Image::Pointer im1 = dynamic_cast<mitk::Image*>(d->m_LeftDataNode->GetData());
    mitk::Image::Pointer im2 = dynamic_cast<mitk::Image*>(d->m_LeftSegmentedNode->GetData());

    d->m_LeftSegmentor->Segment(im1, im2);
  }

  if (   d->m_RightSegmentor.IsNotNull()
      && d->m_RightDataNode != nullptr)
  {
    mitk::Image::Pointer im1 = dynamic_cast<mitk::Image*>(d->m_RightDataNode->GetData());
    mitk::Image::Pointer im2 = dynamic_cast<mitk::Image*>(d->m_RightSegmentedNode->GetData());

    d->m_RightSegmentor->Segment(im1, im2);
  }
}

} // end namespace
