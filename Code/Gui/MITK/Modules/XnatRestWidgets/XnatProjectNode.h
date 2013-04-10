/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef XnatProjectNode_h
#define XnatProjectNode_h

#include "XnatRestWidgetsExports.h"

#include "XnatNode.h"

class XnatRestWidgets_EXPORT XnatProjectNode : public XnatNode
{
public:
  explicit XnatProjectNode(int row = -1, XnatNode* parent = NULL);
  virtual ~XnatProjectNode();

  virtual XnatNode* makeChildNode(int row);

  virtual const char* getKind() const;

};

#endif
