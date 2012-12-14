/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-20 14:25:53 +0000 (Sun, 20 Nov 2011) $
 Revision          : $Revision: 7818 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "QmitkNiftyIGIWorkbenchWindowAdvisor.h"

//-----------------------------------------------------------------------------
QmitkNiftyIGIWorkbenchWindowAdvisor::QmitkNiftyIGIWorkbenchWindowAdvisor(
    berry::WorkbenchAdvisor* wbAdvisor,
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
: QmitkBaseWorkbenchWindowAdvisor(wbAdvisor, configurer)
{
}

