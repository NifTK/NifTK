/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCUDAImageProperty.h"

namespace niftk
{

//-----------------------------------------------------------------------------
bool CUDAImageProperty::IsEqual(const mitk::BaseProperty& property) const
{
  const CUDAImageProperty*  other = dynamic_cast<const CUDAImageProperty*>(&property);
  if (other == 0)
    return false;

  return this->Get().GetId() == other->Get().GetId();
}


//-----------------------------------------------------------------------------
bool CUDAImageProperty::Assign(const mitk::BaseProperty& property)
{
  const CUDAImageProperty*  other = dynamic_cast<const CUDAImageProperty*>(&property);
  if (other == 0)
    return false;

  Set(other->Get());
  return true;
}


//-----------------------------------------------------------------------------
LightweightCUDAImage CUDAImageProperty::Get() const
{
  return m_LWCI;
}


//-----------------------------------------------------------------------------
void CUDAImageProperty::Set(LightweightCUDAImage lwci)
{
  m_LWCI = lwci;
}

} // end namespace
