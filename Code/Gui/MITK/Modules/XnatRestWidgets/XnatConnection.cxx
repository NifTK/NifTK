/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "XnatConnection.h"

#include "XnatRootNode.h"

// XnatConnection class

XnatNode* XnatConnection::getRoot()
{
  // create XNAT root node
  XnatNode* node = new XnatRootNode();
  node->addChild("XNAT");
  return node;
}
