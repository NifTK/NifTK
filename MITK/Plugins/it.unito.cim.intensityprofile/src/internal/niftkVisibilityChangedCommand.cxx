/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkVisibilityChangedCommand.h"

#include "niftkVisibilityChangeObserver.h"

namespace niftk
{

VisibilityChangedCommand::VisibilityChangedCommand(VisibilityChangeObserver* observer, const mitk::DataNode* node)
: m_Observer(observer),
  m_Node(node)
{
}

VisibilityChangedCommand::~VisibilityChangedCommand()
{
}

void VisibilityChangedCommand::Execute(itk::Object* /*caller*/, const itk::EventObject& /*event*/)
{
  m_Observer->OnVisibilityChanged(m_Node);
}

void VisibilityChangedCommand::Execute(const itk::Object* /*caller*/, const itk::EventObject& /*event*/)
{
  m_Observer->OnVisibilityChanged(m_Node);
}

}
