/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "XnatScanNode.h"

extern "C"
{
#include <XnatRest.h>
}

#include "XnatException.h"
#include "XnatScanResourceNode.h"

XnatScanNode::XnatScanNode(int row, XnatNode* parent)
: XnatNode(row, parent)
{
}

XnatScanNode::~XnatScanNode()
{
}

XnatNode* XnatScanNode::makeChildNode(int row)
{
  XnatNode* node = new XnatScanResourceNode(row, this);

  const char* scan = this->getChildName(row);
  XnatNode* categoryNode = this->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  int numResources;
  char** resources;
  XnatRestStatus status = getXnatRestScanResources(project, subject, experiment, scan,
                                                   &numResources, &resources);

  if ( status != XNATREST_OK )
  {
    delete node;
    throw XnatException(status);
  }

  for ( int i = 0 ; i < numResources; i++ )
  {
    node->addChild(resources[i]);
  }

  freeXnatRestArray(numResources, resources);

  return node;
}

void XnatScanNode::download(int row, const char* zipFilename)
{
  const char* scan = this->getChildName(row);
  XnatNode* categoryNode = this->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = getXnatRestAsynAllFilesInScan(project, subject, experiment, scan,
                                                        zipFilename);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

const char* XnatScanNode::getKind() const
{
  return "scan";
}

bool XnatScanNode::holdsFiles() const
{
  return true;
}
