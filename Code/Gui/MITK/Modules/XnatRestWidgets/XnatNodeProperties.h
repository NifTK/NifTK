#ifndef XnatNodeProperties_h
#define XnatNodeProperties_h

#include "XnatRestWidgetsExports.h"

#include <QBitArray>

#include "XnatNode.h"

class XnatRestWidgets_EXPORT XnatNodeProperties
{
public:
  XnatNodeProperties(int row, XnatNode* node);
  XnatNodeProperties(const QBitArray& other);

  QBitArray getBitArray();

  bool isFile() const;
  bool holdsFiles() const;
  bool receivesFiles() const;
  bool isModifiable() const;
  bool isDeletable() const;

private:
  QBitArray propertiesArray;

  enum NodeProperty
  {
    isFileProperty,
    holdsFilesProperty,
    receivesFilesProperty,
    isModifiableProperty,
    isDeletableProperty,
    lastProperty
  };
};

#endif
