/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkLabeledLookupTablePropertySerializer.h"
#include "niftkLabeledLookupTableProperty.h"

#include "niftkSerializerMacros.h"


//-----------------------------------------------------------------------------
niftk::LabeledLookupTablePropertySerializer::LabeledLookupTablePropertySerializer()
{
}


//-----------------------------------------------------------------------------
niftk::LabeledLookupTablePropertySerializer::~LabeledLookupTablePropertySerializer()
{
}


//-----------------------------------------------------------------------------
TiXmlElement* niftk::LabeledLookupTablePropertySerializer::Serialize()
{
  if (const niftk::LabeledLookupTableProperty* prop =
        dynamic_cast<const niftk::LabeledLookupTableProperty*>(m_Property.GetPointer()))
  {
    TiXmlElement* element = new TiXmlElement("LabeledLookupTable");

    TiXmlElement* child = new TiXmlElement("LabelList");
    element->LinkEndChild(child);

    for (int index = 0; index < prop->GetLabels().size(); ++index)
    {
      TiXmlElement* grandChildNinife = new TiXmlElement("Label");
      double value = prop->GetLabels().at(index).first;
      std::string name = prop->GetLabels().at(index).second.toStdString();

      grandChildNinife->SetDoubleAttribute("LabelValue",value);
      grandChildNinife->SetAttribute("LabelName",name);
      child->LinkEndChild(grandChildNinife);
    }

    const niftk::NamedLookupTableProperty* baseProp =
      dynamic_cast<const niftk::NamedLookupTableProperty*>(m_Property.GetPointer());

    this->SetProperty(baseProp);
    child = this->Superclass::Serialize();

    element->LinkEndChild(child);
    return element;
  }
  else
  {
    return NULL;
  }
}


//-----------------------------------------------------------------------------
mitk::BaseProperty::Pointer niftk::LabeledLookupTablePropertySerializer::Deserialize(TiXmlElement* element)
{
  if (!element)
  {
    return NULL;
  }

  niftk::LabeledLookupTableProperty::Pointer  labeledLUT = niftk::LabeledLookupTableProperty::New();

  TiXmlElement* child  = element->FirstChildElement("LabelList");
  if (child)
  {
    niftk::LabeledLookupTableProperty::LabelListType labels;
    for (TiXmlElement* grandChild = child->FirstChildElement("Label");
         grandChild;
         grandChild = grandChild->NextSiblingElement("Label"))
    {
      double value;
      std::string labelName;
      if (grandChild->QueryDoubleAttribute("LabelValue", &value) != TIXML_SUCCESS)
      {
        return NULL;
      }
      if (grandChild->QueryStringAttribute("LabelName", &labelName) != TIXML_SUCCESS)
      {
        return NULL;
      }

      niftk::LabeledLookupTableProperty::LabelType newLabel =
        std::make_pair(int(value), QString::fromStdString(labelName));

      labels.push_back(newLabel);
    }

    labeledLUT->SetLabels(labels);
  }

  child = element->FirstChildElement("NamedLookupTable");
  mitk::BaseProperty::Pointer baseProp;
  if (child)
  {
    baseProp = this->Superclass::Deserialize(child);
  }

  niftk::NamedLookupTableProperty* namedLUTProp = dynamic_cast<niftk::NamedLookupTableProperty*>(baseProp.GetPointer());
  if (namedLUTProp != NULL)
  {
    labeledLUT->SetLookupTable(namedLUTProp->GetLookupTable());
    labeledLUT->SetIsScaled(namedLUTProp->GetIsScaled());
    labeledLUT->SetName(namedLUTProp->GetName());
  }

  return labeledLUT.GetPointer();
}

NIFTK_REGISTER_SERIALIZER(LabeledLookupTablePropertySerializer)
