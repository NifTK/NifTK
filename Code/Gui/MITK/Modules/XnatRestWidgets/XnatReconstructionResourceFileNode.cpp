#include "XnatReconstructionResourceFileNode.h"

extern "C"
{
#include "XnatRest.h"
}

#include "XnatEmptyNode.h"
#include "XnatException.h"

XnatReconstructionResourceFileNode::XnatReconstructionResourceFileNode(int row, XnatNode* parent)
: XnatNode(row, parent)
{
}

XnatReconstructionResourceFileNode::~XnatReconstructionResourceFileNode()
{
}

XnatNode* XnatReconstructionResourceFileNode::makeChildNode(int row)
{
  return new XnatEmptyNode();
}

void XnatReconstructionResourceFileNode::download(int row, const char* zipFilename)
{
  const char* filename = this->getChildName(row);
  const char* resource = this->getParentName();
  XnatNode* resourceNode = this->getParentNode();
  const char* reconstruction = resourceNode->getParentName();
  XnatNode* categoryNode = resourceNode->getParentNode()->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = getXnatRestAsynReconRsrcFile(project, subject, experiment, reconstruction,
                                                       resource, filename, zipFilename);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

void XnatReconstructionResourceFileNode::remove(int row)
{
  const char* filename = this->getChildName(row);
  const char* resource = this->getParentName();
  XnatNode* resourceNode = this->getParentNode();
  const char* reconstruction = resourceNode->getParentName();
  XnatNode* categoryNode = resourceNode->getParentNode()->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = deleteXnatRestReconRsrcFile(project, subject, experiment, reconstruction,
                                                      resource, filename);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

bool XnatReconstructionResourceFileNode::isFile() const
{
  return true;
}

bool XnatReconstructionResourceFileNode::isDeletable() const
{
  return true;
}
