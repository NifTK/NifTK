/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef LOOKUPTABLECONTAINER_CPP
#define LOOKUPTABLECONTAINER_CPP

#include <QFileInfo>
#include "LookupTableContainer.h"

LookupTableContainer::LookupTableContainer(const vtkLookupTable* lut)
{
	m_LookupTable = lut;
	m_DisplayName = QString("");
	m_Order = 0;
}

LookupTableContainer::~LookupTableContainer()
{
	if (m_LookupTable != NULL)
	{
		vtkLookupTable *nonConst = const_cast<vtkLookupTable*>(m_LookupTable);
		nonConst->Delete();
		nonConst = NULL;
	}
}

#endif
