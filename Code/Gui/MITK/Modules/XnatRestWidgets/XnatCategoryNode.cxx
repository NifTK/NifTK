/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "XnatCategoryNode.h"

extern "C"
{
#include <XnatRest.h>
}

#include "XnatScanNode.h"
#include "XnatReconstructionNode.h"
#include "XnatException.h"

XnatCategoryNode::XnatCategoryNode(int row, XnatNode* parent)
: XnatNode(row, parent)
{
}

XnatCategoryNode::~XnatCategoryNode()
{
}

bool XnatCategoryNode::holdsFiles() const
{
  return true;
}

XnatNode* XnatCategoryNode::makeChildNode(int row)
{
  XnatNode* node = NULL;

  const char* category = this->getChildName(row);
  XnatNode* experimentNode = this->getParentNode();
  const char* experiment = this->getParentName();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  if ( strcmp(category, "Scan") == 0 )
  {
    node = new XnatScanNode(row, this);
    int numScans;
    char** scans;
    XnatRestStatus status = getXnatRestScans(project, subject, experiment, &numScans, &scans);
    if ( status != XNATREST_OK )
    {
      delete node;
      throw XnatException(status);
    }

    for ( int i = 0 ; i < numScans; i++ )
    {
      node->addChild(scans[i]);
    }

    freeXnatRestArray(numScans, scans);
  }
  else if ( strcmp(category, "Reconstruction") == 0 )
  {
    node = new XnatReconstructionNode(row, this);
    int numReconstructions;
    char** reconstructions;
    XnatRestStatus status = getXnatRestReconstructions(project, subject, experiment,
                                                       &numReconstructions, &reconstructions);
    if ( status != XNATREST_OK )
    {
      delete node;
      throw XnatException(status);
    }

    for ( int i = 0 ; i < numReconstructions; i++ )
    {
      node->addChild(reconstructions[i]);
    }

    freeXnatRestArray(numReconstructions, reconstructions);
  }
  else
  {
    // code error
  }

  return node;
}

void XnatCategoryNode::download(int row, const char* zipFilename)
{
  const char* category = this->getChildName(row);
  const char* experiment = this->getParentName();
  XnatNode* experimentNode = this->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = XNATREST_ERROR;
  if ( strcmp(category, "Scan") == 0 )
  {
    status = getXnatRestAsynAllScanFilesInExperiment(project, subject, experiment, zipFilename);
  }
  else if ( strcmp(category, "Reconstruction") == 0 )
  {
    status = getXnatRestAsynAllReconFilesInExperiment(project, subject, experiment, zipFilename);
  }
  else
  {
    // code error
  }
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

void XnatCategoryNode::add(int row, const char* categoryEntry)
{
  if ( strcmp(this->getChildName(row), "Reconstruction") == 0 )
  {
    const char* experiment = this->getParentName();
    XnatNode* experimentNode = this->getParentNode();
    const char* subject = experimentNode->getParentName();
    XnatNode* subjectNode = experimentNode->getParentNode();
    const char* project = subjectNode->getParentName();

    XnatRestStatus status = putXnatRestReconstruction(project, subject, experiment, categoryEntry);
    if ( status != XNATREST_OK )
    {
      throw XnatException(status);
    }
  }
  else
  {
    // code error
  }
}

const char* XnatCategoryNode::getModifiableChildKind(int row) const
{
  if ( strcmp(this->getChildName(row), "Reconstruction") == 0 )
  {
    return "reconstruction";
  }

  return NULL;
}

const char* XnatCategoryNode::getModifiableParentName(int row) const
{
  if ( strcmp(this->getChildName(row), "Reconstruction") == 0 )
  {
    return this->getParentName();
  }

  return NULL;
}

bool XnatCategoryNode::isModifiable(int row) const
{
  if ( strcmp(this->getChildName(row), "Reconstruction") == 0 )
  {
    return true;
  }

  return false;
}
