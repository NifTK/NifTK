/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef XnatReconstructionNode_h
#define XnatReconstructionNode_h

#include "XnatRestWidgetsExports.h"

#include "XnatNode.h"

class XnatRestWidgets_EXPORT XnatReconstructionNode : public XnatNode
{
public:
  explicit XnatReconstructionNode(int row = -1, XnatNode* parent = NULL);
  virtual ~XnatReconstructionNode();

  virtual XnatNode* makeChildNode(int row);
  virtual void download(int row, const char* zipFilename);
  virtual void add(int row, const char* resource);
  virtual void remove(int row);

  virtual const char* getKind() const;
  virtual const char* getModifiableChildKind(int row) const;
  virtual const char* getModifiableParentName(int row) const;

  virtual bool holdsFiles() const;
  virtual bool isModifiable(int row) const;
  virtual bool isDeletable() const;
};

#endif
