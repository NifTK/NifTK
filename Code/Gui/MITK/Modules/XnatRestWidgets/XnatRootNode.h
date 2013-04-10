/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef XnatRootNode_h
#define XnatRootNode_h

#include "XnatRestWidgetsExports.h"

#include "XnatNode.h"

class XnatRestWidgets_EXPORT XnatRootNode : public XnatNode
{
public:
  explicit XnatRootNode();
  virtual ~XnatRootNode();

  virtual XnatNode* makeChildNode(int row);
};

#endif
