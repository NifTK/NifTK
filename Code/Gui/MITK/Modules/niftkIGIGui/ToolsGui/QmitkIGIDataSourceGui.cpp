/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-25 07:31:59 +0100 (Wed, 25 Jul 2012) $
 Revision          : $Revision: 9401 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkIGIDataSourceGui.h"
#include <iostream>

//-----------------------------------------------------------------------------
QmitkIGIDataSourceGui::QmitkIGIDataSourceGui()
{

}


//-----------------------------------------------------------------------------
QmitkIGIDataSourceGui::~QmitkIGIDataSourceGui()
{
  m_ReferenceCountLock.Lock();
  m_ReferenceCount = 0; // otherwise ITK will complain in LightObject's destructor
  m_ReferenceCountLock.Unlock();

  // We don't own m_Tool or m_StdMultiWidget, so don't delete them.
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceGui::Register() const
{
  // empty on purpose, just don't let ITK handle calls to Register()
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceGui::UnRegister() const
{
  // empty on purpose, just don't let ITK handle calls to UnRegister()
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceGui::SetReferenceCount(int)
{
  // empty on purpose, just don't let ITK handle calls to SetReferenceCount()
}

//-----------------------------------------------------------------------------
void QmitkIGIDataSourceGui::SetDataSource( mitk::IGIDataSource* source )
{
  m_Source = source;
  emit NewSourceAssociated(m_Source);
}

