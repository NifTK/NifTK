#include "XnatScanResourceNode.h"

extern "C"
{
#include "XnatRest.h"
}

#include "XnatScanResourceFileNode.h"
#include "XnatException.h"

XnatScanResourceNode::XnatScanResourceNode(int row, XnatNode* parent)
: XnatNode(row, parent)
{
}

XnatScanResourceNode::~XnatScanResourceNode()
{
}

XnatNode* XnatScanResourceNode::makeChildNode(int row)
{
  XnatNode* node = new XnatScanResourceFileNode(row, this);

  const char* resource = this->getChildName(row);
  const char* scan = this->getParentName();
  XnatNode* categoryNode = this->getParentNode()->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  int numFilenames;
  char** filenames;
  XnatRestStatus status = getXnatRestScanRsrcFilenames(project, subject, experiment, scan, resource,
                                                       &numFilenames, &filenames);
  if ( status != XNATREST_OK )
  {
    delete node;
    throw XnatException(status);
  }

  for ( int i = 0 ; i < numFilenames; i++ )
  {
    node->addChild(filenames[i]);
  }

  freeXnatRestArray(numFilenames, filenames);

  return node;
}

void XnatScanResourceNode::download(int row, const char* zipFilename)
{
  const char* resource = this->getChildName(row);
  const char* scan = this->getParentName();
  XnatNode* categoryNode = this->getParentNode()->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = getXnatRestAsynAllFilesInScanRsrc(project, subject, experiment, scan,
                                                            resource, zipFilename);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

const char* XnatScanResourceNode::getKind() const
{
  return "resource";
}

bool XnatScanResourceNode::holdsFiles() const
{
  return true;
}
