/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMaskMergerController.h"
#include <Internal/niftkMaskMergerGUI.h>
#include <niftkBinaryMaskUtils.h>
#include <niftkImageUtils.h>
#include <mitkDataNode.h>
#include <QMessageBox>
#include <QMutex>
#include <QMutexLocker>

namespace niftk
{

class MaskMergerControllerPrivate
{
  Q_DECLARE_PUBLIC(MaskMergerController);
  MaskMergerController* const q_ptr;

public:

  MaskMergerControllerPrivate(MaskMergerController* q);
  ~MaskMergerControllerPrivate();

  void InitialiseMask(const std::string& name,
                      const mitk::DataNode* input1,
                      const mitk::DataNode* input2,
                      mitk::DataNode::Pointer& output1
                     );

  void UpdateMask(const mitk::DataNode* input1,
                  const mitk::DataNode* input2,
                  mitk::DataNode::Pointer& output1
                 );

  MaskMergerGUI*          m_GUI;
  mitk::DataNode*         m_LeftMask1;
  mitk::DataNode*         m_LeftMask2;
  mitk::DataNode*         m_RightMask1;
  mitk::DataNode*         m_RightMask2;
  mitk::DataNode::Pointer m_LeftResult;
  mitk::DataNode::Pointer m_RightResult;
  QMutex                  m_Lock;
};


//-----------------------------------------------------------------------------
MaskMergerControllerPrivate::MaskMergerControllerPrivate(MaskMergerController* maskMergerController)
: q_ptr(maskMergerController)
, m_GUI(nullptr)
, m_LeftMask1(nullptr)
, m_LeftMask2(nullptr)
, m_RightMask1(nullptr)
, m_RightMask2(nullptr)
, m_LeftResult(nullptr)
, m_RightResult(nullptr)
, m_Lock(QMutex::Recursive)
{
  Q_Q(MaskMergerController);
}


//-----------------------------------------------------------------------------
MaskMergerControllerPrivate::~MaskMergerControllerPrivate()
{
  /* Do we want this?
  if (output1.IsNotNull()
      && q_ptr->GetDataStorage()->Exists(output1)
      )
  {
    q_ptr->GetDataStorage()->Remove(output1);
    output1 = nullptr;
  }
  */
}


//-----------------------------------------------------------------------------
void MaskMergerControllerPrivate::InitialiseMask(const std::string& name,
                                                 const mitk::DataNode* input1,
                                                 const mitk::DataNode* input2,
                                                 mitk::DataNode::Pointer& output1
                                                )
{
  bool didInitialise = false;

  if (   input1 != nullptr
      && input2 != nullptr
      && output1.IsNull()
     )
  {
    mitk::Image* im1 = dynamic_cast<mitk::Image*>(input1->GetData());
    mitk::Image* im2 = dynamic_cast<mitk::Image*>(input2->GetData());

    if (   im1 != nullptr
        && niftk::IsBinaryMask(im1)
        && im2 != nullptr
        && niftk::IsBinaryMask(im2)
        && niftk::ImagesHaveSameSpatialExtent(im1, im2)
       )
    {
      mitk::PixelType pt = mitk::MakeScalarPixelType<unsigned char>();
      mitk::Image::Pointer op = mitk::Image::New();
      unsigned int dim[] = { im1->GetDimension(0), im1->GetDimension(1) };
      op->Initialize( pt, 2, dim);

      mitk::DataNode::Pointer segNode = mitk::DataNode::New();
      segNode->SetName(name);
      segNode->SetBoolProperty("binary", true);
      segNode->SetBoolProperty("outline binary", true);
      segNode->SetData(op);

      output1 = segNode;
      didInitialise = true;
    }
  }

  if(   !didInitialise
     || (didInitialise && output1.IsNotNull()))
  {
    if (output1.IsNotNull()
        && q_ptr->GetDataStorage()->Exists(output1)
        )
    {
      q_ptr->GetDataStorage()->Remove(output1);
      output1 = nullptr;
    }
  }

  if (didInitialise)
  {
    q_ptr->GetDataStorage()->Add(output1);
  }
}


//-----------------------------------------------------------------------------
void MaskMergerControllerPrivate::UpdateMask(const mitk::DataNode* input1,
                                             const mitk::DataNode* input2,
                                             mitk::DataNode::Pointer& output1
                                            )
{
  if (   input1 != nullptr
      && input2 != nullptr
      && output1.IsNotNull()
     )
  {
    mitk::Image* im1 = dynamic_cast<mitk::Image*>(input1->GetData());
    mitk::Image* im2 = dynamic_cast<mitk::Image*>(input2->GetData());
    mitk::Image::Pointer op = dynamic_cast<mitk::Image*>(output1->GetData());

    if (im1 != nullptr && im2 != nullptr && op.IsNotNull())
    {
      niftk::BinaryMaskAndOperator(im1, im2, op);
      op->Modified();
    }
  }
}


//-----------------------------------------------------------------------------
MaskMergerController::MaskMergerController(IBaseView* view)
: BaseController(view)
, d_ptr(new MaskMergerControllerPrivate(this))
{
}


//-----------------------------------------------------------------------------
MaskMergerController::~MaskMergerController()
{
  Q_D(MaskMergerController);
}


//-----------------------------------------------------------------------------
BaseGUI* MaskMergerController::CreateGUI(QWidget* parent)
{
  return new MaskMergerGUI(parent);
}


//-----------------------------------------------------------------------------
void MaskMergerController::SetupGUI(QWidget* parent)
{
  Q_D(MaskMergerController);
  BaseController::SetupGUI(parent);
  d->m_GUI = dynamic_cast<MaskMergerGUI*>(this->GetGUI());
  d->m_GUI->SetDataStorage(this->GetDataStorage());
  connect(d->m_GUI, SIGNAL(LeftMask1SelectionChanged(const mitk::DataNode*)), this, SLOT(OnLeftMask1SelectionChanged(const mitk::DataNode*)));
  connect(d->m_GUI, SIGNAL(LeftMask2SelectionChanged(const mitk::DataNode*)), this, SLOT(OnLeftMask2SelectionChanged(const mitk::DataNode*)));
  connect(d->m_GUI, SIGNAL(RightMask1SelectionChanged(const mitk::DataNode*)), this, SLOT(OnRightMask1SelectionChanged(const mitk::DataNode*)));
  connect(d->m_GUI, SIGNAL(RightMask2SelectionChanged(const mitk::DataNode*)), this, SLOT(OnRightMask2SelectionChanged(const mitk::DataNode*)));
}


//-----------------------------------------------------------------------------
void MaskMergerController::OnNodeRemoved(const mitk::DataNode* node)
{
  Q_D(MaskMergerController);
  QMutexLocker locker(&d->m_Lock);

  if (   node == d->m_LeftMask1
      || node == d->m_LeftMask2
      || node == d->m_LeftResult.GetPointer()
      )
  {
    d->m_GUI->ResetLeft();
  }

  if (   node == d->m_RightMask1
      || node == d->m_RightMask2
      || node == d->m_RightResult.GetPointer()
      )
  {
    d->m_GUI->ResetRight();
  }
}


//-----------------------------------------------------------------------------
void MaskMergerController::OnLeftMask1SelectionChanged(const mitk::DataNode* node)
{
  Q_D(MaskMergerController);
  QMutexLocker locker(&d->m_Lock);

  d->m_LeftMask1 = const_cast<mitk::DataNode*>(node);
  d->InitialiseMask("MergedMask-Left", d->m_LeftMask1, d->m_LeftMask2, d->m_LeftResult);
}


//-----------------------------------------------------------------------------
void MaskMergerController::OnLeftMask2SelectionChanged(const mitk::DataNode* node)
{
  Q_D(MaskMergerController);
  QMutexLocker locker(&d->m_Lock);

  d->m_LeftMask2 = const_cast<mitk::DataNode*>(node);
  d->InitialiseMask("MergedMask-Left", d->m_LeftMask1, d->m_LeftMask2, d->m_LeftResult);
}


//-----------------------------------------------------------------------------
void MaskMergerController::OnRightMask1SelectionChanged(const mitk::DataNode* node)
{
  Q_D(MaskMergerController);
  QMutexLocker locker(&d->m_Lock);

  d->m_RightMask1 = const_cast<mitk::DataNode*>(node);
  d->InitialiseMask("MergedMask-Right", d->m_RightMask1, d->m_RightMask2, d->m_RightResult);
}


//-----------------------------------------------------------------------------
void MaskMergerController::OnRightMask2SelectionChanged(const mitk::DataNode* node)
{
  Q_D(MaskMergerController);
  QMutexLocker locker(&d->m_Lock);

  d->m_RightMask2 = const_cast<mitk::DataNode*>(node);
  d->InitialiseMask("MergedMask-Right", d->m_RightMask1, d->m_RightMask2, d->m_RightResult);
}


//-----------------------------------------------------------------------------
void MaskMergerController::Update()
{
  Q_D(MaskMergerController);
  QMutexLocker locker(&d->m_Lock);

  d->UpdateMask(d->m_LeftMask1, d->m_LeftMask2, d->m_LeftResult);
  d->UpdateMask(d->m_RightMask1, d->m_RightMask2, d->m_RightResult);
}

} // end namespace
