#include "XnatConnection.h"

#include "XnatNodeActivity.h"

// XnatConnection class

XnatNode* XnatConnection::getRoot()
{
  // create XNAT root node
  XnatNode* node = new XnatNode(XnatRootActivity::instance());
  node->addChild("XNAT");
  return node;
}
