/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-15 15:03:56 +0000 (Thu, 15 Dec 2011) $
 Revision          : $Revision: 8030 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "CommonActivator.h"
#include <QtPlugin>

namespace mitk {

ctkPluginContext* CommonActivator::s_PluginContext(NULL);

void CommonActivator::start(ctkPluginContext* context)
{
  s_PluginContext = context;
}

void CommonActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

ctkPluginContext* CommonActivator::GetPluginContext()
{
  return s_PluginContext;
}

}

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_gui_qt_common, mitk::CommonActivator)
