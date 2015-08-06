/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkLabeledLookupTableProperty.h"

namespace mitk {

//-----------------------------------------------------------------------------
LabeledLookupTableProperty::LabeledLookupTableProperty()
: Superclass()
{
}


//-----------------------------------------------------------------------------
LabeledLookupTableProperty::LabeledLookupTableProperty(const LabeledLookupTableProperty& other)
: Superclass(other)
, m_Labels(other.m_Labels)
{

}


//-----------------------------------------------------------------------------
LabeledLookupTableProperty::LabeledLookupTableProperty(const std::string& name, const mitk::LookupTable::Pointer lut, LabelListType labels)
: Superclass(name, lut, false)
{
  m_Labels = labels;
}


//-----------------------------------------------------------------------------
LabeledLookupTableProperty::LabeledLookupTableProperty(const std::string& name, const mitk::LookupTable::Pointer lut,LabelListType labels, bool scale)
: Superclass(name, lut, scale)
{
  m_Labels = labels;
}


//-----------------------------------------------------------------------------
LabeledLookupTableProperty::~LabeledLookupTableProperty()
{
}


//-----------------------------------------------------------------------------
itk::LightObject::Pointer LabeledLookupTableProperty::InternalClone() const
{
  itk::LightObject::Pointer result(new Self(*this));
  return result;
}


//-----------------------------------------------------------------------------
bool LabeledLookupTableProperty::IsEqual(const BaseProperty& property) const
{
  LabelListType otherLabels = static_cast<const Self&>(property).m_Labels;
  bool sameLabels = (m_Labels.size() == otherLabels.size());

  if(sameLabels)
  {
    for(unsigned int i=0;i<m_Labels.size() && i<otherLabels.size();i++)
    {
      LabelType labelPair = m_Labels.at(i);
      LabelType otherPair = otherLabels.at(i);

      if(labelPair.first!=otherPair.first ||
        labelPair.second.compare(otherPair.second) != 0 )
      {
        sameLabels = false;
        break;
      }
    }
  }

  return *(this->m_LookupTable) == *(static_cast<const Self&>(property).m_LookupTable)
      && this->GetName() == static_cast<const Self&>(property).GetName()
      && this->GetIsScaled() == static_cast<const Self&>(property).GetIsScaled()
      && sameLabels;
}


//-----------------------------------------------------------------------------
bool LabeledLookupTableProperty::Assign(const BaseProperty& property)
{
  this->m_LookupTable = static_cast<const Self&>(property).m_LookupTable;
  this->SetName( static_cast<const Self&>(property).GetName() );
  this->SetIsScaled( static_cast<const Self&>(property).GetIsScaled() );
  this->m_Labels = static_cast<const Self&>(property).m_Labels;
  return true;
}

} // namespace mitk
