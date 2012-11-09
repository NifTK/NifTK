#include "XnatConnection.h"

#include "XnatRootNode.h"

// XnatConnection class

XnatNode* XnatConnection::getRoot()
{
  // create XNAT root node
  XnatNode* node = new XnatRootNode();
  node->addChild("XNAT");
  return node;
}
