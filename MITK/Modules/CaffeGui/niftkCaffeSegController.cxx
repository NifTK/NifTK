/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCaffeSegController.h"
#include <Internal/niftkCaffeSegGUI.h>

namespace niftk
{

class CaffeSegControllerPrivate
{
  Q_DECLARE_PUBLIC(CaffeSegController);
  CaffeSegController* const q_ptr;

public:

  CaffeSegControllerPrivate(CaffeSegController* q);
  ~CaffeSegControllerPrivate();

  CaffeSegGUI* m_GUI;
};


//-----------------------------------------------------------------------------
CaffeSegControllerPrivate::CaffeSegControllerPrivate(CaffeSegController* caffeSegController)
: q_ptr(caffeSegController)
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
}

} // end namespace
