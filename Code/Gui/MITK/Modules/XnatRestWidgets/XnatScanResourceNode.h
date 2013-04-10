/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef XnatScanResourceNode_h
#define XnatScanResourceNode_h

#include "XnatRestWidgetsExports.h"

#include "XnatNode.h"

class XnatRestWidgets_EXPORT XnatScanResourceNode : public XnatNode
{
public:
  explicit XnatScanResourceNode(int row = -1, XnatNode* parent = NULL);
  virtual ~XnatScanResourceNode();

  virtual XnatNode* makeChildNode(int row);
  virtual void download(int row, const char* zipFilename);

  virtual const char* getKind() const;
  virtual bool holdsFiles() const;
};

#endif
