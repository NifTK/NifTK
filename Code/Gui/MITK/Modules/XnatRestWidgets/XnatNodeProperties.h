#ifndef XNATNODEPROPERTIES_H
#define XNATNODEPROPERTIES_H

#include <QBitArray>
#include "XnatNode.h"


class XnatNodeProperties
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


inline XnatNodeProperties::XnatNodeProperties(const QBitArray& other) : propertiesArray(other) {}

inline QBitArray XnatNodeProperties::getBitArray() { return propertiesArray; }

inline bool XnatNodeProperties::isFile() const { 
	bool result = false;
	if(propertiesArray.size() > 0){
		result = propertiesArray.at(isFileProperty);
	}
	return result;
}

inline bool XnatNodeProperties::holdsFiles() const { 
	bool result = false;
	if(propertiesArray.size() > 0){
		result = propertiesArray.at(holdsFilesProperty);
	}
	return result;
}

inline bool XnatNodeProperties::receivesFiles() const { 
	bool result = false;
	if(propertiesArray.size() > 0){
		result =  propertiesArray.at(receivesFilesProperty);
	}
	return result;
}

inline bool XnatNodeProperties::isModifiable() const { 
	bool result = false;
	if(propertiesArray.size() > 0){
		result =  propertiesArray.at(isModifiableProperty);
	}
	return result;
}

inline bool XnatNodeProperties::isDeletable() const { 
	bool result = false;
	if(propertiesArray.size() > 0){
		result =  propertiesArray.at(isDeletableProperty);
	}
	return result;
}

#endif
