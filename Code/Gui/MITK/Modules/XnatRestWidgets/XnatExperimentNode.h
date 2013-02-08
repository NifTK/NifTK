/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef XnatExperimentNode_h
#define XnatExperimentNode_h

#include "XnatRestWidgetsExports.h"

#include "XnatNode.h"

class XnatRestWidgets_EXPORT XnatExperimentNode : public XnatNode
{
public:
  explicit XnatExperimentNode(int row = -1, XnatNode* parent = NULL);
  virtual ~XnatExperimentNode();

  virtual XnatNode* makeChildNode(int row);
  virtual void add(int row, const char* reconstruction);

  virtual const char* getKind() const;
  virtual const char* getModifiableChildKind(int row) const;
  virtual const char* getModifiableParentName(int row) const;

  virtual bool isModifiable(int row) const;
};

#endif
