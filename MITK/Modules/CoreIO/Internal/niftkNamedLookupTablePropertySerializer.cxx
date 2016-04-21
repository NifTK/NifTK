/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNamedLookupTablePropertySerializer.h"
#include "niftkNamedLookupTableProperty.h"

#include "niftkSerializerMacros.h"


//-----------------------------------------------------------------------------
niftk::NamedLookupTablePropertySerializer::NamedLookupTablePropertySerializer()
{
}


//-----------------------------------------------------------------------------
niftk::NamedLookupTablePropertySerializer::~NamedLookupTablePropertySerializer()
{
}


//-----------------------------------------------------------------------------
TiXmlElement* niftk::NamedLookupTablePropertySerializer::Serialize()
{
  if (const niftk::NamedLookupTableProperty* prop =
        dynamic_cast<const niftk::NamedLookupTableProperty*>(m_Property.GetPointer()))
  {
    TiXmlElement* element = new TiXmlElement("NamedLookupTable");
    element->SetAttribute("Name", prop->GetName());
    element->SetAttribute("IsScaled", prop->GetIsScaled());

    const mitk::LookupTableProperty* baseProp =
      dynamic_cast< const mitk::LookupTableProperty*>(m_Property.GetPointer());

    this->SetProperty( baseProp);
    TiXmlElement* child = this->Superclass::Serialize();

    element->LinkEndChild(child);
    return element;
  }
  else
  {
    return NULL;
  }
}


//-----------------------------------------------------------------------------
mitk::BaseProperty::Pointer niftk::NamedLookupTablePropertySerializer::Deserialize(TiXmlElement* element)
{
  if (!element)
  {
    return NULL;
  }

  niftk::NamedLookupTableProperty::Pointer namedLUT = niftk::NamedLookupTableProperty::New();

  std::string name;
  if (element->QueryStringAttribute("Name", &name) == TIXML_SUCCESS)
  {
    namedLUT->SetName(name);
  }

  bool scaled = false;
  if (element->QueryBoolAttribute("IsScaled", &scaled) == TIXML_SUCCESS)
  {
    namedLUT->SetIsScaled(scaled);
  }

  TiXmlElement* child = element->FirstChildElement("LookupTable");
  mitk::BaseProperty::Pointer baseProp;
  if (child)
  {
    baseProp = this->Superclass::Deserialize(child);
  }

  mitk::LookupTableProperty* mitkLUTProp = dynamic_cast<mitk::LookupTableProperty*>(baseProp.GetPointer());
  if (mitkLUTProp != NULL)
  {
    namedLUT->SetLookupTable(mitkLUTProp->GetLookupTable());
  }

  return namedLUT.GetPointer();
}

NIFTK_REGISTER_SERIALIZER(NamedLookupTablePropertySerializer)
