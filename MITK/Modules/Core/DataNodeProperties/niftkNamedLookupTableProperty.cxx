/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNamedLookupTableProperty.h"

//-----------------------------------------------------------------------------
niftk::NamedLookupTableProperty::NamedLookupTableProperty()
: Superclass()
, m_Name("n/a")
, m_IsScaled(1)
{
}


//-----------------------------------------------------------------------------
niftk::NamedLookupTableProperty::NamedLookupTableProperty(const niftk::NamedLookupTableProperty& other)
: Superclass(other)
, m_Name(other.m_Name)
, m_IsScaled(other.m_IsScaled)
{
}


//-----------------------------------------------------------------------------
niftk::NamedLookupTableProperty::NamedLookupTableProperty(const std::string& name, const mitk::LookupTable::Pointer lut)
: Superclass(lut)
, m_Name(name)
, m_IsScaled(1)
{
}


//-----------------------------------------------------------------------------
niftk::NamedLookupTableProperty::NamedLookupTableProperty(const std::string& name, const mitk::LookupTable::Pointer lut, bool scale)
: Superclass(lut)
, m_Name(name)
, m_IsScaled(scale)
{
}


//-----------------------------------------------------------------------------
niftk::NamedLookupTableProperty::~NamedLookupTableProperty()
{
}


//-----------------------------------------------------------------------------
std::string niftk::NamedLookupTableProperty::GetValueAsString() const
{
  return m_Name;
}


//-----------------------------------------------------------------------------
itk::LightObject::Pointer niftk::NamedLookupTableProperty::InternalClone() const
{
  itk::LightObject::Pointer result(new Self(*this));
  return result;
}


//-----------------------------------------------------------------------------
bool niftk::NamedLookupTableProperty::IsEqual(const mitk::BaseProperty& property) const
{
  return *(this->m_LookupTable) == *(static_cast<const Self&>(property).m_LookupTable)
      && this->m_Name == static_cast<const Self&>(property).m_Name
      && this->m_IsScaled == static_cast<const Self&>(property).m_IsScaled;
}


//-----------------------------------------------------------------------------
bool niftk::NamedLookupTableProperty::Assign(const mitk::BaseProperty& property)
{
  this->m_LookupTable = static_cast<const Self&>(property).m_LookupTable;
  this->m_Name = static_cast<const Self&>(property).m_Name;
  this->m_IsScaled = static_cast<const Self&>(property).m_IsScaled;

  return true;
}
