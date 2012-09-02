#include "XnatNode.h"

#include "XnatNodeActivity.h"


XnatNode::XnatNode(XnatNodeActivity& a, int r, XnatNode* p)
: nodeActivity(a)
, rowInParent(r)
, parent(p)
{
}

XnatNode::~XnatNode()
{
//  foreach (XnatChild* child, children)
//  {
//    delete child;
//  }
  for (unsigned i = 0; i < children.size(); ++i)
  {
    delete children[i];
  }
}

const char* XnatNode::getParentName()
{
  return ( ( parent != NULL ) ? parent->getChildName(rowInParent) : NULL );
}

int XnatNode::getRowInParent()
{
  return rowInParent;
}

XnatNode* XnatNode::getParentNode()
{
  return parent;
}

int XnatNode::getNumChildren()
{
  return children.size();
}

const char* XnatNode::getChildName(int row)
{
  return children[row]->name.c_str();
}

XnatNode* XnatNode::getChildNode(int row)
{
  return children[row]->node;
}

void XnatNode::addChild(const char* name)
{
  children.push_back(new XnatChild(name));
}

void XnatNode::addChildNode(int row, XnatNode* node)
{
  children[row]->node = node;
}

XnatNode* XnatNode::makeChildNode(int row)
{
  return nodeActivity.makeChildNode(row, this);
}

void XnatNode::removeChildNode(int row)
{
  if ( children[row]->node != NULL )
  {
    delete children[row]->node;
    children[row]->node = NULL;
  }
}

void XnatNode::download(int row, const char* zipFilename)
{
  nodeActivity.download(row, this, zipFilename);
}

void XnatNode::downloadAllFiles(int row, const char* zipFilename)
{
  nodeActivity.downloadAllFiles(row, this, zipFilename);
}

void XnatNode::upload(int row, const char* zipFilename)
{
  nodeActivity.upload(row, this, zipFilename);
}

void XnatNode::add(int row, const char* name)
{
  nodeActivity.add(row, this, name);
}

void XnatNode::remove(int row)
{
  nodeActivity.remove(row, this);
}

const char* XnatNode::getKind()
{
  return nodeActivity.getKind();
}

const char* XnatNode::getModifiableChildKind(int row)
{
  return nodeActivity.getModifiableChildKind(row, this);
}

const char* XnatNode::getModifiableParentName(int row)
{
  return nodeActivity.getModifiableParentName(row, this);
}

bool XnatNode::isFile()
{
  return nodeActivity.isFile();
}

bool XnatNode::holdsFiles()
{
  return nodeActivity.holdsFiles();
}

bool XnatNode::receivesFiles()
{
  return nodeActivity.receivesFiles();
}

bool XnatNode::isModifiable(int row)
{
  return nodeActivity.isModifiable(row, this);
}

bool XnatNode::isDeletable()
{
  return nodeActivity.isDeletable();
}

XnatNode::XnatChild::XnatChild(const char* who, XnatNode* ptr)
: name(who)
, node(ptr)
{
}

XnatNode::XnatChild::~XnatChild()
{
  // delete child node
  if ( node != NULL )
  {
    delete node;
  }
}
