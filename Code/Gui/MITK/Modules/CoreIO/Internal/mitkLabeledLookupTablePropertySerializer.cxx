/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkLabeledLookupTablePropertySerializer.h"
#include "mitkLabeledLookupTableProperty.h"

namespace mitk
{

//-----------------------------------------------------------------------------
LabeledLookupTablePropertySerializer::LabeledLookupTablePropertySerializer()
{
}


//-----------------------------------------------------------------------------
LabeledLookupTablePropertySerializer::~LabeledLookupTablePropertySerializer()
{
}


//-----------------------------------------------------------------------------
TiXmlElement* LabeledLookupTablePropertySerializer::Serialize()
{
  if (const LabeledLookupTableProperty* prop = dynamic_cast<const LabeledLookupTableProperty*>(m_Property.GetPointer()))
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

    const NamedLookupTableProperty* baseProp = dynamic_cast< const NamedLookupTableProperty*>(m_Property.GetPointer());
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
BaseProperty::Pointer LabeledLookupTablePropertySerializer::Deserialize(TiXmlElement* element)
{
  if (!element)
  { 
    return NULL;
  }

  LabeledLookupTableProperty::Pointer  labeledLUT = LabeledLookupTableProperty::New();

  TiXmlElement* child  = element->FirstChildElement("LabelList");
  if (child)
  {
    LabeledLookupTableProperty::LabelListType labels;
    for (TiXmlElement* grandChild = child->FirstChildElement("Label"); grandChild; grandChild = grandChild->NextSiblingElement("Label"))
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

      LabeledLookupTableProperty::LabelType newLabel = std::make_pair(int(value), QString::fromStdString(labelName));
      labels.push_back(newLabel);
    }

    labeledLUT->SetLabels(labels);
  }

  child = element->FirstChildElement("NamedLookupTable");
  BaseProperty::Pointer baseProp;
  if (child)
  {
    baseProp = this->Superclass::Deserialize(child);
  }

  NamedLookupTableProperty* namedLUTProp = dynamic_cast< NamedLookupTableProperty*>(baseProp.GetPointer());
  if (namedLUTProp != NULL)
  {
    labeledLUT->SetLookupTable(namedLUTProp->GetLookupTable());
    labeledLUT->SetIsScaled(namedLUTProp->GetIsScaled());
    labeledLUT->SetName(namedLUTProp->GetName());
  }

  return labeledLUT.GetPointer();
}

}

MITK_REGISTER_SERIALIZER(LabeledLookupTablePropertySerializer)
