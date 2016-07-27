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


namespace niftk
{

//-----------------------------------------------------------------------------
BaseGUI::BaseGUI(QWidget* parent)
  : QObject(parent)
{
}


//-----------------------------------------------------------------------------
BaseGUI::~BaseGUI()
{
}


//-----------------------------------------------------------------------------
QWidget* BaseGUI::GetParent() const
{
  return dynamic_cast<QWidget*>(this->parent());
}

}
