#include "XnatRootNode.h"

extern "C"
{
#include <XnatRest.h>
}

#include "XnatException.h"
#include "XnatProjectNode.h"

XnatRootNode::XnatRootNode()
{
}

XnatRootNode::~XnatRootNode()
{
}

XnatNode* XnatRootNode::makeChildNode(int row)
{
  XnatNode* node = new XnatProjectNode(row, this);

  int numProjects;
  char** projects;
  XnatRestStatus status = getXnatRestProjects(&numProjects, &projects);
  if ( status != XNATREST_OK )
  {
    delete node;
    throw XnatException(status);
  }

  for ( int i = 0 ; i < numProjects; i++ )
  {
    node->addChild(projects[i]);
  }

  freeXnatRestArray(numProjects, projects);

  return node;
}
