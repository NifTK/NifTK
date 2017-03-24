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
#include <vtkIntArray.h>
#include <vtkStringArray.h>

namespace niftk
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
  if (const LabeledLookupTableProperty* prop =
        dynamic_cast<const LabeledLookupTableProperty*>(m_Property.GetPointer()))
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

    const NamedLookupTableProperty* baseProp =
      dynamic_cast<const NamedLookupTableProperty*>(m_Property.GetPointer());

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
mitk::BaseProperty::Pointer LabeledLookupTablePropertySerializer::Deserialize(TiXmlElement* element)
{
  if (!element)
  {
    return NULL;
  }

  LabeledLookupTableProperty::Pointer  labeledLUT = LabeledLookupTableProperty::New();

  TiXmlElement* child  = element->FirstChildElement("LabelList");
  LabeledLookupTableProperty::LabelListType labels;

  int maxValue = INT_MIN;

  if (child)
  {
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

      if (maxValue < int(value))
      {
        maxValue = int(value);
      }

      LabeledLookupTableProperty::LabelType newLabel =
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

  NamedLookupTableProperty* namedLUTProp = dynamic_cast<NamedLookupTableProperty*>(baseProp.GetPointer());

  if (namedLUTProp != NULL)
  {
    mitk::LookupTable* mitkLUT = namedLUTProp->GetLookupTable();
    vtkSmartPointer<vtkLookupTable> vtkLUT = mitkLUT->GetVtkLookupTable();

    vtkSmartPointer<vtkLookupTable> newVtkLUT = vtkSmartPointer<vtkLookupTable>::New();
    newVtkLUT->SetHueRange(vtkLUT->GetHueRange());
    newVtkLUT->SetValueRange(vtkLUT->GetValueRange());
    newVtkLUT->SetSaturationRange(vtkLUT->GetSaturationRange());
    newVtkLUT->SetAlphaRange(vtkLUT->GetAlphaRange());

    int numberOfValues = maxValue + 2;
    newVtkLUT->SetNumberOfTableValues(numberOfValues);
    newVtkLUT->SetTableRange(0, maxValue);
    newVtkLUT->SetNanColor(0, 0, 0, 0);
    newVtkLUT->SetIndexedLookup(true);

    newVtkLUT->Build();
    
    vtkSmartPointer<vtkIntArray> annotationValueArray = vtkIntArray::New();
    vtkSmartPointer<vtkStringArray> annotationNameArray = vtkStringArray::New();

    for (unsigned int i =0; i < labels.size(); ++i)
    {
      int vtkInd = labels.at(i).first;
      std::string name = labels.at(i).second.toStdString();

      newVtkLUT->SetTableValue(vtkInd, vtkLUT->GetTableValue(vtkInd));
      annotationValueArray->InsertValue(vtkInd, vtkInd);
      annotationNameArray->InsertValue(vtkInd, name);

    }

    newVtkLUT->SetAnnotations(annotationValueArray, annotationNameArray);

    mitkLUT->SetVtkLookupTable(newVtkLUT);
    labeledLUT->SetLookupTable(mitkLUT);
    labeledLUT->SetIsScaled(namedLUTProp->GetIsScaled());
    labeledLUT->SetName(namedLUTProp->GetName());
  }

  return labeledLUT.GetPointer();
}

}

NIFTK_REGISTER_SERIALIZER(LabeledLookupTablePropertySerializer)
