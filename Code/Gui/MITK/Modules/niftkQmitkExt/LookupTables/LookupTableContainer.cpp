/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3326 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef LOOKUPTABLECONTAINER_CPP
#define LOOKUPTABLECONTAINER_CPP

#include <QFileInfo>
#include "LookupTableContainer.h"

LookupTableContainer::LookupTableContainer(const vtkLookupTable* lut)
{
	this->m_LookupTable = lut;
	this->m_DisplayName = QString("");
	this->m_Order = 0;
}

LookupTableContainer::~LookupTableContainer()
{
	if (this->m_LookupTable != NULL)
	{
		vtkLookupTable *nonConst = const_cast<vtkLookupTable*>(this->m_LookupTable);
		nonConst->Delete();
		nonConst = NULL;
	}
}

#endif
