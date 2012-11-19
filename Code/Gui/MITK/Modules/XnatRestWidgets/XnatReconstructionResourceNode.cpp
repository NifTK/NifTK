#include "XnatReconstructionResourceNode.h"

extern "C"
{
#include "XnatRest.h"
}

#include "XnatReconstructionResourceFileNode.h"
#include "XnatException.h"

XnatReconstructionResourceNode::XnatReconstructionResourceNode(int row, XnatNode* parent)
: XnatNode(row, parent)
{
}

XnatReconstructionResourceNode::~XnatReconstructionResourceNode()
{
}

XnatNode* XnatReconstructionResourceNode::makeChildNode(int row)
{
  XnatNode* node = new XnatReconstructionResourceFileNode(row, this);

  const char* resource = this->getChildName(row);
  const char* reconstruction = this->getParentName();
  XnatNode* categoryNode = this->getParentNode()->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  int numFilenames;
  char** filenames;
  XnatRestStatus status = getXnatRestReconRsrcFilenames(project, subject, experiment, reconstruction,
                                                        resource, &numFilenames, &filenames);
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

void XnatReconstructionResourceNode::download(int row, const char* zipFilename)
{
  const char* resource = this->getChildName(row);
  const char* reconstruction = this->getParentName();
  XnatNode* categoryNode = this->getParentNode()->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = getXnatRestAsynAllFilesInReconRsrc(project, subject, experiment, reconstruction,
                                                             resource, zipFilename);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

void XnatReconstructionResourceNode::upload(int row, const char* zipFilename)
{
  const char* resource = this->getChildName(row);
  const char* reconstruction = this->getParentName();
  XnatNode* categoryNode = this->getParentNode()->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = putXnatRestAsynReconRsrcFiles(project, subject, experiment, reconstruction,
                                                        resource, zipFilename);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

void XnatReconstructionResourceNode::remove(int row)
{
  const char* resource = this->getChildName(row);
  const char* reconstruction = this->getParentName();
  XnatNode* categoryNode = this->getParentNode()->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = deleteXnatRestReconResource(project, subject, experiment,
                                                      reconstruction, resource);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

const char* XnatReconstructionResourceNode::getKind() const
{
  return "resource";
}

bool XnatReconstructionResourceNode::holdsFiles() const
{
  return true;
}

bool XnatReconstructionResourceNode::receivesFiles() const
{
  return true;
}

bool XnatReconstructionResourceNode::isDeletable() const
{
  return true;
}
