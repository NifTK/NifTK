/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : $Author$

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "uk_ac_ucl_cmic_surgicalguidance_Activator.h"
#include <QtPlugin>
#include "SurgicalGuidanceView.h"
#include "SurgicalGuidanceViewPreferencePage.h"

namespace mitk {

void uk_ac_ucl_cmic_surgicalguidance_Activator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(SurgicalGuidanceView, context)
  BERRY_REGISTER_EXTENSION_CLASS(SurgicalGuidanceViewPreferencePage, context)
}

void uk_ac_ucl_cmic_surgicalguidance_Activator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

}

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_surgicalguidance, mitk::uk_ac_ucl_cmic_surgicalguidance_Activator)
