/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGITrackerDataType.h"

namespace niftk
{

//-----------------------------------------------------------------------------
IGITrackerDataType::~IGITrackerDataType()
{
}


//-----------------------------------------------------------------------------
IGITrackerDataType::IGITrackerDataType()
{
  m_Rotation.Fill(0);
  m_Rotation[0] = 1;
  m_Translation.Fill(0);
}


//-----------------------------------------------------------------------------
IGITrackerDataType::IGITrackerDataType(const IGITrackerDataType& other)
: IGIDataType(other)
{
  m_ToolName = other.m_ToolName;
  m_Rotation = other.m_Rotation;
  m_Translation = other.m_Translation;
}


//-----------------------------------------------------------------------------
IGITrackerDataType::IGITrackerDataType(IGITrackerDataType&& other)
: IGIDataType(other)
{
  m_ToolName = other.m_ToolName;
  m_Rotation = other.m_Rotation;
  m_Translation = other.m_Translation;
}


//-----------------------------------------------------------------------------
IGITrackerDataType& IGITrackerDataType::operator=(const IGITrackerDataType& other)
{
  IGIDataType::operator=(other);
  m_ToolName = other.m_ToolName;
  m_Rotation = other.m_Rotation;
  m_Translation = other.m_Translation;
  return *this;
}


//-----------------------------------------------------------------------------
IGITrackerDataType& IGITrackerDataType::operator=(IGITrackerDataType&& other)
{
  IGIDataType::operator=(other);
  m_ToolName = other.m_ToolName;
  m_Rotation = other.m_Rotation;
  m_Translation = other.m_Translation;
  return *this;
}


//-----------------------------------------------------------------------------
void IGITrackerDataType::SetTransform(const mitk::Point4D& rotation, const mitk::Vector3D& translation)
{
  m_Rotation = rotation;
  m_Translation = translation;
}


//-----------------------------------------------------------------------------
void IGITrackerDataType::GetTransform(mitk::Point4D& rotation, mitk::Vector3D& translation) const
{
  rotation = m_Rotation;
  translation = m_Translation;
}


//-----------------------------------------------------------------------------
void IGITrackerDataType::Clone(const IGIDataType& other)
{
  IGIDataType::Clone(other);
  const IGITrackerDataType* tmp = dynamic_cast<const IGITrackerDataType*>(&other);
  if (tmp != nullptr)
  {
    m_Rotation = (*tmp).m_Rotation;
    m_Translation = (*tmp).m_Translation;
  }
  else
  {
    mitkThrow() << "Incorrect data type provided";
  }
}

} // end namespace
