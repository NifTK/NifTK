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
#include <QMessageBox>

namespace niftk
{

class MaskMergerControllerPrivate
{
  Q_DECLARE_PUBLIC(MaskMergerController);
  MaskMergerController* const q_ptr;

public:

  MaskMergerControllerPrivate(MaskMergerController* q);
  ~MaskMergerControllerPrivate();

  MaskMergerGUI* m_GUI;

};


//-----------------------------------------------------------------------------
MaskMergerControllerPrivate::MaskMergerControllerPrivate(MaskMergerController* maskMergerController)
: q_ptr(maskMergerController)
{
  Q_Q(MaskMergerController);
}


//-----------------------------------------------------------------------------
MaskMergerControllerPrivate::~MaskMergerControllerPrivate()
{
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
}


//-----------------------------------------------------------------------------
void MaskMergerController::OnNodeRemoved(const mitk::DataNode* node)
{
  Q_D(MaskMergerController);
}


//-----------------------------------------------------------------------------
void MaskMergerController::Update()
{

}

} // end namespace
