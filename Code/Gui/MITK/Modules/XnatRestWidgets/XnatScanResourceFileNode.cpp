#include "XnatScanResourceFileNode.h"

extern "C"
{
#include "XnatRest.h"
}

#include "XnatEmptyNode.h"
#include "XnatException.h"

XnatScanResourceFileNode::XnatScanResourceFileNode(int row, XnatNode* parent)
: XnatNode(row, parent)
{
}

XnatScanResourceFileNode::~XnatScanResourceFileNode()
{
}

XnatNode* XnatScanResourceFileNode::makeChildNode(int row)
{
  return new XnatEmptyNode();
}

void XnatScanResourceFileNode::download(int row, const char* zipFilename)
{
  const char* filename = this->getChildName(row);
  const char* resource = this->getParentName();
  XnatNode* resourceNode = this->getParentNode();
  const char* scan = resourceNode->getParentName();
  XnatNode* categoryNode = resourceNode->getParentNode()->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = getXnatRestAsynScanRsrcFile(project, subject, experiment, scan, resource,
                                                      filename, zipFilename );
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

bool XnatScanResourceFileNode::isFile() const
{
  return true;
}
