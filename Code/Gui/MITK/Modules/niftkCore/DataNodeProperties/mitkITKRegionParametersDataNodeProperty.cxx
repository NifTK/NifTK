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

ITKRegionParametersDataNodeProperty::ITKRegionParametersDataNodeProperty()
{
  this->Identity();
}

ITKRegionParametersDataNodeProperty::~ITKRegionParametersDataNodeProperty()
{
}

void ITKRegionParametersDataNodeProperty::Identity()
{
  m_IsValid = false;
  m_Parameters.resize(6);
  m_Parameters[0] = 0;
  m_Parameters[1] = 0;
  m_Parameters[2] = 0;
  m_Parameters[3] = 0;
  m_Parameters[4] = 0;
  m_Parameters[5] = 0;
}

void ITKRegionParametersDataNodeProperty::SetSize(int x, int y, int z)
{
  m_Parameters[0] = x;
  m_Parameters[1] = y;
  m_Parameters[2] = z;
}

bool ITKRegionParametersDataNodeProperty::HasVolume() const
{
  if (m_Parameters[0] > 0 && m_Parameters[1] > 0 && m_Parameters[2] > 0)
  {
    return true;
  }
  else
  {
    return false;
  }
}

bool ITKRegionParametersDataNodeProperty::IsValid() const
{
  return m_IsValid;
}

void ITKRegionParametersDataNodeProperty::SetValid(bool valid)
{
  if (valid != m_IsValid)
  {
    m_IsValid = valid;
    this->Modified();
  }
}

const ITKRegionParametersDataNodeProperty::ParametersType& ITKRegionParametersDataNodeProperty::GetITKRegionParameters() const
{
  return m_Parameters;
}

void ITKRegionParametersDataNodeProperty::SetITKRegionParameters(const ParametersType& parameters)
{
  if (m_Parameters != parameters)
  {
    m_Parameters = parameters;
    this->Modified();
  }
}

std::string ITKRegionParametersDataNodeProperty::GetValueAsString() const
{
  std::stringstream myStr;
  myStr <<   "Valid=" << m_IsValid \
        << ", Size=[" << m_Parameters[0] \
        << ", " << m_Parameters[1] \
        << ", " << m_Parameters[2] \
        << "], Index=[" << m_Parameters[3] \
        << ", " << m_Parameters[4] \
        << ", " << m_Parameters[5] \
        << "]" ;
  return myStr.str();
}

bool ITKRegionParametersDataNodeProperty::IsEqual(const BaseProperty& property) const
{
  const Self *other = dynamic_cast<const Self*>(&property);

  if(other==NULL) return false;

  ParametersType otherParameters = other->GetITKRegionParameters();
  if (otherParameters.size() != m_Parameters.size()) return false;

  return (m_Parameters == otherParameters && m_IsValid == other->IsValid());
}

bool ITKRegionParametersDataNodeProperty::Assign(const BaseProperty& property)
{
  const Self *other = dynamic_cast<const Self*>(&property);

  if(other==NULL) return false;

  ParametersType otherParameters = other->GetITKRegionParameters();
  this->m_Parameters = otherParameters;
  this->m_IsValid = other->IsValid();

  return true;
}

} // end namespace
