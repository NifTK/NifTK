/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkAffineTransformParametersDataNodeProperty.h"

namespace mitk
{


//-----------------------------------------------------------------------------
AffineTransformParametersDataNodeProperty::AffineTransformParametersDataNodeProperty()
{
  this->Identity();
}


//-----------------------------------------------------------------------------
AffineTransformParametersDataNodeProperty::AffineTransformParametersDataNodeProperty(const AffineTransformParametersDataNodeProperty& other)
: Superclass(other)
{
  m_Parameters = other.m_Parameters;
}


//-----------------------------------------------------------------------------
AffineTransformParametersDataNodeProperty::AffineTransformParametersDataNodeProperty(const ParametersType& parameters)
{
  SetAffineTransformParameters(parameters);
}


//-----------------------------------------------------------------------------
AffineTransformParametersDataNodeProperty::~AffineTransformParametersDataNodeProperty()
{
}


//-----------------------------------------------------------------------------
void AffineTransformParametersDataNodeProperty::Identity()
{
  m_Parameters.resize(13);  // extra element at the end to tell if we are rotating about centre

  for (int i = 0; i < 13; i++)
  {
    m_Parameters[i] = 0;
  }
  m_Parameters[6] = 100; // scaling
  m_Parameters[7] = 100;
  m_Parameters[8] = 100;
}


//-----------------------------------------------------------------------------
const AffineTransformParametersDataNodeProperty::ParametersType& AffineTransformParametersDataNodeProperty::GetAffineTransformParameters() const
{
    return m_Parameters;
}


//-----------------------------------------------------------------------------
void AffineTransformParametersDataNodeProperty::SetAffineTransformParameters(const ParametersType& parameters)
{
  if (m_Parameters != parameters)
  {
    m_Parameters = parameters;
    this->Modified();
  }
}


//-----------------------------------------------------------------------------
std::string AffineTransformParametersDataNodeProperty::GetValueAsString() const
{
  std::stringstream myStr;
  myStr <<   "rx:" << m_Parameters[0] \
        << ", ry:" << m_Parameters[1] \
        << ", rz:" << m_Parameters[2] \
        << ", tx:" << m_Parameters[3] \
        << ", ty:" << m_Parameters[4] \
        << ", tz:" << m_Parameters[5] \
        << ", sx:" << m_Parameters[6] \
        << ", sy:" << m_Parameters[7] \
        << ", sz:" << m_Parameters[8] \
        << ", k1:" << m_Parameters[9] \
        << ", k2:" << m_Parameters[10] \
        << ", k3:" << m_Parameters[11] \
        << ", cc:" << m_Parameters[12];
  return myStr.str();
}


//-----------------------------------------------------------------------------
bool AffineTransformParametersDataNodeProperty::IsEqual(const BaseProperty& property) const
{
  const Self *other = dynamic_cast<const Self*>(&property);

  if(other==NULL) return false;

  ParametersType otherParameters = other->GetAffineTransformParameters();
  if (otherParameters.size() != m_Parameters.size()) return false;

  return m_Parameters == otherParameters;
}


//-----------------------------------------------------------------------------
bool AffineTransformParametersDataNodeProperty::Assign(const BaseProperty& property)
{
  const Self *other = dynamic_cast<const Self*>(&property);

  if(other==NULL) return false;

  ParametersType otherParameters = other->GetAffineTransformParameters();
  m_Parameters = otherParameters;

  return true;
}


//-----------------------------------------------------------------------------
AffineTransformParametersDataNodeProperty::Pointer AffineTransformParametersDataNodeProperty::Clone() const
{
  AffineTransformParametersDataNodeProperty::Pointer result = static_cast<Self*>(this->InternalClone().GetPointer());
  return result;
}


//-----------------------------------------------------------------------------
itk::LightObject::Pointer AffineTransformParametersDataNodeProperty::InternalClone() const
{
  itk::LightObject::Pointer result(new Self(*this));
  return result;
}

} // end namespace
