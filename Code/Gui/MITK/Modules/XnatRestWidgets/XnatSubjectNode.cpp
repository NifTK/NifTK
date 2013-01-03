#include "XnatSubjectNode.h"

extern "C"
{
#include "XnatRest.h"
}

#include "XnatExperimentNode.h"
#include "XnatException.h"

XnatSubjectNode::XnatSubjectNode(int row, XnatNode* parent)
: XnatNode(row, parent)
{
}

XnatSubjectNode::~XnatSubjectNode()
{
}

XnatNode* XnatSubjectNode::makeChildNode(int row)
{
  XnatNode* node = new XnatExperimentNode(row, this);

  const char* subject = this->getChildName(row);
  const char* project = this->getParentName();

  int numExperiments;
  char** experiments;
  XnatRestStatus status = getXnatRestExperiments(project, subject, &numExperiments, &experiments);
  if ( status != XNATREST_OK )
  {
    delete node;
    throw XnatException(status);
  }

  for ( int i = 0 ; i < numExperiments; i++ )
  {
    node->addChild(experiments[i]);
  }

  freeXnatRestArray(numExperiments, experiments);

  return node;
}

const char* XnatSubjectNode::getKind() const
{
  return "subject";
}
