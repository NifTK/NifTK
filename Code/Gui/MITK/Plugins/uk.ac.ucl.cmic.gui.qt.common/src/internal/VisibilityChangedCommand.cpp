/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-24 15:53:45 +0000 (Thu, 24 Nov 2011) $
 Revision          : $Revision: 7857 $
 Last modified by  : $Author: mjc $

 Original author   : Miklos Espak <espakm@gmail.com>

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

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
