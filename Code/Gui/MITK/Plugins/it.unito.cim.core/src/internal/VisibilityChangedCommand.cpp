/*=============================================================================

 KMaps:     An image processing toolkit for DCE-MRI analysis developed
            at the Molecular Imaging Center at University of Torino.

 See:       http://www.cim.unito.it

 Author:    Miklos Espak <espakm@gmail.com>

 Copyright (c) Miklos Espak
 All Rights Reserved.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "VisibilityChangedCommand.h"

#include "VisibilityChangeObserver.h"

VisibilityChangedCommand::VisibilityChangedCommand(VisibilityChangeObserver* observer, const mitk::DataNode* node)
: m_Observer(observer),
  m_Node(node)
{
}

VisibilityChangedCommand::~VisibilityChangedCommand()
{
}

void
VisibilityChangedCommand::Execute(itk::Object* /*caller*/, const itk::EventObject& /*event*/)
{
  m_Observer->onVisibilityChanged(m_Node);
}

void
VisibilityChangedCommand::Execute(const itk::Object* /*caller*/, const itk::EventObject& /*event*/)
{
  m_Observer->onVisibilityChanged(m_Node);
}
