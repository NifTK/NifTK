/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkNamedLookupTablePropertySerializer.h"
#include "mitkNamedLookupTableProperty.h"

namespace mitk
{

NamedLookupTablePropertySerializer::NamedLookupTablePropertySerializer()
{
}

NamedLookupTablePropertySerializer::~NamedLookupTablePropertySerializer()
{
}

TiXmlElement* NamedLookupTablePropertySerializer::Serialize()
{
  if (const NamedLookupTableProperty* prop = dynamic_cast<const NamedLookupTableProperty*>(m_Property.GetPointer()))
  {
    TiXmlElement* element = new TiXmlElement("NamedLookupTable");
    element->SetAttribute("Name", prop->GetName() );
    element->SetAttribute("IsScaled", prop->GetIsScaled() );

    const LookupTableProperty* baseProp = dynamic_cast< const LookupTableProperty*>(m_Property.GetPointer());
    this->SetProperty( baseProp);
    TiXmlElement* child = this->Superclass::Serialize();

    element->LinkEndChild(child);
    return element;
  }
  else
    return NULL;

}

BaseProperty::Pointer NamedLookupTablePropertySerializer::Deserialize(TiXmlElement* element)
{
  if (!element) 
    return NULL;

  NamedLookupTableProperty::Pointer  namedLUT = NamedLookupTableProperty::New();

  std::string name;
  if( element->QueryStringAttribute("Name", &name) == TIXML_SUCCESS )
  {
    namedLUT->SetName(name);
  }
  bool* scaled;
  if( element->QueryBoolAttribute("IsScaled", scaled) == TIXML_SUCCESS )
  {
    namedLUT->SetIsScaled(scaled);
  }
  TiXmlElement* child = element->FirstChildElement("LookupTable");
  BaseProperty::Pointer baseProp;
  if( child )
  {
    baseProp = this->Superclass::Deserialize(child);
  }

  LookupTableProperty* mitkLUTProp = dynamic_cast< LookupTableProperty*>(baseProp.GetPointer());
  if(mitkLUTProp != NULL )
    namedLUT->SetLookupTable(mitkLUTProp->GetLookupTable());

  return namedLUT.GetPointer();

}

}

MITK_REGISTER_SERIALIZER(NamedLookupTablePropertySerializer)