#include "XnatNodeProperties.h"


XnatNodeProperties::XnatNodeProperties(const QBitArray& other)
: propertiesArray(other)
{
}

QBitArray XnatNodeProperties::getBitArray()
{
  return propertiesArray;
}

bool XnatNodeProperties::isFile() const
{
  bool result = false;
  if(propertiesArray.size() > 0)
  {
    result = propertiesArray.at(isFileProperty);
  }
  return result;
}

bool XnatNodeProperties::holdsFiles() const
{
  bool result = false;
  if(propertiesArray.size() > 0)
  {
    result = propertiesArray.at(holdsFilesProperty);
  }
  return result;
}

bool XnatNodeProperties::receivesFiles() const
{
  bool result = false;
  if(propertiesArray.size() > 0)
  {
    result =  propertiesArray.at(receivesFilesProperty);
  }
  return result;
}

bool XnatNodeProperties::isModifiable() const
{
  bool result = false;
  if(propertiesArray.size() > 0)
  {
    result =  propertiesArray.at(isModifiableProperty);
  }
  return result;
}

bool XnatNodeProperties::isDeletable() const
{
  bool result = false;
  if(propertiesArray.size() > 0)
  {
    result =  propertiesArray.at(isDeletableProperty);
  }
  return result;
}

XnatNodeProperties::XnatNodeProperties(int row, XnatNode* node)
: propertiesArray(lastProperty, false)
{
  propertiesArray.setBit(isFileProperty, node->isFile());
  propertiesArray.setBit(holdsFilesProperty, node->holdsFiles());
  propertiesArray.setBit(receivesFilesProperty, node->receivesFiles());
  propertiesArray.setBit(isModifiableProperty, node->isModifiable(row));
  propertiesArray.setBit(isDeletableProperty, node->isDeletable());
}
