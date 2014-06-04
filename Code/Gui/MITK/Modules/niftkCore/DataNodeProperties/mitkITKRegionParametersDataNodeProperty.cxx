/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkITKRegionParametersDataNodeProperty.h"

namespace mitk
{

//-----------------------------------------------------------------------------
ITKRegionParametersDataNodeProperty::ITKRegionParametersDataNodeProperty()
: m_Parameters(6)
, m_IsValid(false)
{
  std::fill(m_Parameters.begin(), m_Parameters.end(), 0);
}


//-----------------------------------------------------------------------------
ITKRegionParametersDataNodeProperty::ITKRegionParametersDataNodeProperty(const ITKRegionParametersDataNodeProperty& other)
: Superclass(other)
, m_IsValid(other.m_IsValid)
{
  m_Parameters = other.GetITKRegionParameters();
}


//-----------------------------------------------------------------------------
ITKRegionParametersDataNodeProperty::~ITKRegionParametersDataNodeProperty()
{
}


//-----------------------------------------------------------------------------
void ITKRegionParametersDataNodeProperty::Identity()
{
  std::fill(m_Parameters.begin(), m_Parameters.end(), 0);
  m_IsValid = false;
}


//-----------------------------------------------------------------------------
void ITKRegionParametersDataNodeProperty::SetIndex(int x, int y, int z)
{
  if (x != m_Parameters[0] || y != m_Parameters[1] || z != m_Parameters[2])
  {
    m_Parameters[0] = x;
    m_Parameters[1] = y;
    m_Parameters[2] = z;
    this->Modified();
  }
}


//-----------------------------------------------------------------------------
void ITKRegionParametersDataNodeProperty::SetSize(int x, int y, int z)
{
  if (x != m_Parameters[3] || y != m_Parameters[4] || z != m_Parameters[5])
  {
    m_Parameters[3] = x;
    m_Parameters[4] = y;
    m_Parameters[5] = z;
    this->Modified();
  }
}


//-----------------------------------------------------------------------------
bool ITKRegionParametersDataNodeProperty::HasVolume() const
{
  return m_Parameters[3] > 0 && m_Parameters[4] > 0 && m_Parameters[5] > 0;
}


//-----------------------------------------------------------------------------
bool ITKRegionParametersDataNodeProperty::IsValid() const
{
  return m_IsValid;
}


//-----------------------------------------------------------------------------
void ITKRegionParametersDataNodeProperty::SetValid(bool valid)
{
  if (valid != m_IsValid)
  {
    m_IsValid = valid;
    this->Modified();
  }
}


//-----------------------------------------------------------------------------
const ITKRegionParametersDataNodeProperty::ParametersType& ITKRegionParametersDataNodeProperty::GetITKRegionParameters() const
{
  return m_Parameters;
}


//-----------------------------------------------------------------------------
void ITKRegionParametersDataNodeProperty::SetITKRegionParameters(const ParametersType& parameters)
{
  assert(parameters.size() == 6);

  if (m_Parameters != parameters)
  {
    m_Parameters = parameters;
    this->Modified();
  }
}


//-----------------------------------------------------------------------------
std::string ITKRegionParametersDataNodeProperty::GetValueAsString() const
{
  std::stringstream myStr;
  myStr << "Valid = " << m_IsValid << ", "
        << "Index = [" << m_Parameters[0] << ", " << m_Parameters[1] << ", " << m_Parameters[2] << "], "
        << "Size = [" << m_Parameters[3] << ", " << m_Parameters[4] << ", " << m_Parameters[5] << "]" ;
  return myStr.str();
}


//-----------------------------------------------------------------------------
bool ITKRegionParametersDataNodeProperty::IsEqual(const BaseProperty& property) const
{
  const Self *other = dynamic_cast<const Self*>(&property);

  if (other == NULL)
  {
    return false;
  }

  ParametersType otherParameters = other->GetITKRegionParameters();

  return m_Parameters == otherParameters && m_IsValid == other->IsValid();
}


//-----------------------------------------------------------------------------
bool ITKRegionParametersDataNodeProperty::Assign(const BaseProperty& property)
{
  const Self* other = dynamic_cast<const Self*>(&property);

  if (other == NULL)
  {
    return false;
  }

  m_Parameters = other->GetITKRegionParameters();
  m_IsValid = other->IsValid();

  return true;
}


//-----------------------------------------------------------------------------
itk::LightObject::Pointer ITKRegionParametersDataNodeProperty::InternalClone() const
{
  itk::LightObject::Pointer result(new Self(*this));
  return result;
}

} // end namespace
