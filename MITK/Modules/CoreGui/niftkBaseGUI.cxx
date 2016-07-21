/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseGUI.h"

#include <QWidget>

//-----------------------------------------------------------------------------
niftk::BaseGUI::BaseGUI(QWidget* parent)
  : QObject(parent)
{
}


//-----------------------------------------------------------------------------
niftk::BaseGUI::~BaseGUI()
{
}


//-----------------------------------------------------------------------------
QWidget* niftk::BaseGUI::GetParent() const
{
  return dynamic_cast<QWidget*>(this->parent());
}
