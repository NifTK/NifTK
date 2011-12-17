/*=========================================================================

Program:   Medical Imaging & Interaction Toolkit
Language:  C++
Date:      $Date: 2011-10-17 06:48:41 +0100 (Mon, 17 Oct 2011) $
Version:   $Revision: 7528 $

Copyright (c) German Cancer Research Center, Division of Medical and
Biological Informatics. All rights reserved.
See MITKCopyright.txt or http://www.mitk.org/copyright.html for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/


#include "NamedLookupTableProperty.h"

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
