/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "XnatNode.h"

XnatNode::XnatNode(int r, XnatNode* p)
: rowInParent(r)
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

const char* XnatNode::getParentName() const
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

const char* XnatNode::getChildName(int row) const
{
  return children[row]->name.c_str();
}

XnatNode* XnatNode::getChildNode(int row) const
{
  return children[row]->node;
}

void XnatNode::addChild(const char* name)
{
  children.push_back(new XnatChild(name));
}

void XnatNode::setChildNode(int row, XnatNode* node)
{
  children[row]->node = node;
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
  // do nothing
}

void XnatNode::upload(int row, const char* zipFilename)
{
  // do nothing
}

void XnatNode::add(int row, const char* name)
{
  // do nothing
}

void XnatNode::remove(int row)
{
  // do nothing
}

const char* XnatNode::getKind() const
{
  return NULL;
}

const char* XnatNode::getModifiableChildKind(int row) const
{
  return NULL;
}

const char* XnatNode::getModifiableParentName(int row) const
{
  return NULL;
}

bool XnatNode::isFile() const
{
  return false;
}

bool XnatNode::holdsFiles() const
{
  return false;
}

bool XnatNode::receivesFiles() const
{
  return false;
}

bool XnatNode::isModifiable(int row) const
{
  return false;
}

bool XnatNode::isDeletable() const
{
  return false;
}

XnatNode::XnatChild::XnatChild(const char* who)
: name(who)
, node(0)
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
