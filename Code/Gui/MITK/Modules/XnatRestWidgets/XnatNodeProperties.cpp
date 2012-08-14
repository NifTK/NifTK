#include <QtGui>
#include "XnatNodeProperties.h"


XnatNodeProperties::XnatNodeProperties(int row, XnatNode* node) : propertiesArray(lastProperty, false)
{
    propertiesArray.setBit(isFileProperty, node->isFile());
    propertiesArray.setBit(holdsFilesProperty, node->holdsFiles());
    propertiesArray.setBit(receivesFilesProperty, node->receivesFiles());
    propertiesArray.setBit(isModifiableProperty, node->isModifiable(row));
    propertiesArray.setBit(isDeletableProperty, node->isDeletable());
}

