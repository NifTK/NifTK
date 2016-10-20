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

namespace niftk
{

class DistanceMeasurerControllerPrivate
{
  Q_DECLARE_PUBLIC(DistanceMeasurerController);
  DistanceMeasurerController* const q_ptr;

public:

  DistanceMeasurerControllerPrivate(DistanceMeasurerController* q);
  ~DistanceMeasurerControllerPrivate();

  DistanceMeasurerGUI* m_GUI;
};


//-----------------------------------------------------------------------------
DistanceMeasurerControllerPrivate::DistanceMeasurerControllerPrivate(DistanceMeasurerController* distanceMeasurerController)
: q_ptr(distanceMeasurerController)
{
  Q_Q(DistanceMeasurerController);
}


//-----------------------------------------------------------------------------
DistanceMeasurerControllerPrivate::~DistanceMeasurerControllerPrivate()
{
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
}


//-----------------------------------------------------------------------------
void DistanceMeasurerController::Update()
{
  Q_D(DistanceMeasurerController);
}


//-----------------------------------------------------------------------------
void DistanceMeasurerController::OnNodeRemoved(const mitk::DataNode* node)
{
  Q_D(DistanceMeasurerController);
}

} // end namespace
