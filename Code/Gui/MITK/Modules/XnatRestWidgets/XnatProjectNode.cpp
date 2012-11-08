#include "XnatProjectNode.h"

#include "XnatSubjectNode.h"
#include "XnatException.h"

extern "C"
{
#include "XnatRest.h"
}

XnatProjectNode::XnatProjectNode(int row, XnatNode* parent)
: XnatNode(row, parent)
{
}

XnatProjectNode::~XnatProjectNode()
{
}

XnatNode* XnatProjectNode::makeChildNode(int row)
{
  XnatNode* node = new XnatSubjectNode(row, this);

  const char* project = this->getChildName(row);

  int numSubjects;
  char** subjects;
  XnatRestStatus status = getXnatRestSubjects(project, &numSubjects, &subjects);
  if ( status != XNATREST_OK )
  {
    delete node;
    throw XnatException(status);
  }

  for ( int i = 0 ; i < numSubjects; i++ )
  {
    node->addChild(subjects[i]);
  }

  freeXnatRestArray(numSubjects, subjects);

  return node;
}

const char* XnatProjectNode::getKind() const
{
  return "project";
}
