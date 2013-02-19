/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "XnatExperimentNode.h"

extern "C"
{
#include "XnatRest.h"
}

#include "XnatCategoryNode.h"
#include "XnatException.h"

XnatExperimentNode::XnatExperimentNode(int row, XnatNode* parent)
: XnatNode(row, parent)
{
}

XnatExperimentNode::~XnatExperimentNode()
{
}

XnatNode* XnatExperimentNode::makeChildNode(int row)
{
  XnatNode* node = new XnatCategoryNode(row, this);

  const char* experiment = this->getChildName(row);
  const char* subject = this->getParentName();
  XnatNode* subjectNode = this->getParentNode();
  const char* project = subjectNode->getParentName();

  int numNames;
  char** names;
  XnatRestStatus status;

  status = getXnatRestScans(project, subject, experiment, &numNames, &names);
  if ( status != XNATREST_OK )
  {
    delete node;
    throw XnatException(status);
  }

  if ( numNames > 0 )
  {
    node->addChild("Scan");
  }

  freeXnatRestArray(numNames, names);

  status = getXnatRestReconstructions(project, subject, experiment, &numNames, &names);
  if ( status != XNATREST_OK )
  {
    delete node;
    throw XnatException(status);
  }

  if ( numNames > 0 )
  {
    node->addChild("Reconstruction");
  }

  freeXnatRestArray(numNames, names);

  return node;
}

void XnatExperimentNode::add(int row, const char* reconstruction)
{
  const char* experiment = this->getChildName(row);
  const char* subject = this->getParentName();
  XnatNode* subjectNode = this->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = putXnatRestReconstruction(project, subject, experiment, reconstruction);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

const char* XnatExperimentNode::getKind() const
{
  return "experiment";
}

const char* XnatExperimentNode::getModifiableChildKind(int row) const
{
  return "reconstruction";
}

const char* XnatExperimentNode::getModifiableParentName(int row) const
{
  return this->getChildName(row);
}

bool XnatExperimentNode::isModifiable(int row) const
{
  XnatNode* childNode = this->getChildNode(row);
  if (childNode == NULL)
  {
    return false;
  }
  int numChildren = childNode->getNumChildren();
  for (int i = 0; i < numChildren; i++)
  {
    if ( strcmp(childNode->getChildName(i), "Reconstruction") == 0 )
    {
      return false;
    }
  }

  return true;
}
