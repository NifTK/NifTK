#include "XnatReconstructionNode.h"

extern "C"
{
#include "XnatRest.h"
}

#include "XnatReconstructionResourceNode.h"
#include "XnatException.h"

XnatReconstructionNode::XnatReconstructionNode(int row, XnatNode* parent)
: XnatNode(row, parent)
{
}

XnatReconstructionNode::~XnatReconstructionNode()
{
}

XnatNode* XnatReconstructionNode::makeChildNode(int row)
{
  XnatNode* node = new XnatReconstructionResourceNode(row, this);

  const char* reconstruction = this->getChildName(row);
  XnatNode* categoryNode = this->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  int numResources;
  char** resources;
  XnatRestStatus status = getXnatRestReconResources(project, subject, experiment, reconstruction,
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

void XnatReconstructionNode::download(int row, const char* zipFilename)
{
  const char* reconstruction = this->getChildName(row);
  XnatNode* categoryNode = this->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = getXnatRestAsynAllFilesInReconstruction(project, subject, experiment,
                                                                  reconstruction, zipFilename);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

void XnatReconstructionNode::add(int row, const char* resource)
{
  const char* reconstruction = this->getChildName(row);
  XnatNode* categoryNode = this->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = putXnatRestReconResource(project, subject, experiment,
                                                   reconstruction, resource);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

void XnatReconstructionNode::remove(int row)
{
  const char* reconstruction = this->getChildName(row);
  XnatNode* categoryNode = this->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = deleteXnatRestReconstruction(project, subject, experiment, reconstruction);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

const char* XnatReconstructionNode::getKind() const
{
  return "reconstruction";
}

const char* XnatReconstructionNode::getModifiableChildKind(int row) const
{
  return "resource";
}

const char* XnatReconstructionNode::getModifiableParentName(int row) const
{
  return this->getChildName(row);
}

bool XnatReconstructionNode::holdsFiles() const
{
  return true;
}

bool XnatReconstructionNode::isModifiable(int row) const
{
  return true;
}

bool XnatReconstructionNode::isDeletable() const
{
  return true;
}
