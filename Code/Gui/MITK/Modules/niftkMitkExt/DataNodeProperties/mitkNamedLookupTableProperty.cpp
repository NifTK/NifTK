/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkNamedLookupTableProperty.h"

namespace mitk {

NamedLookupTableProperty::NamedLookupTableProperty()
: Superclass()
, m_Name("n/a")
{
}

NamedLookupTableProperty::NamedLookupTableProperty(const std::string& name, const mitk::LookupTable::Pointer lut)
: Superclass(lut)
, m_Name(name)
{
}

NamedLookupTableProperty::~NamedLookupTableProperty()
{
}

std::string
NamedLookupTableProperty::GetValueAsString() const
{
  return m_Name;
}

} // namespace mitk
