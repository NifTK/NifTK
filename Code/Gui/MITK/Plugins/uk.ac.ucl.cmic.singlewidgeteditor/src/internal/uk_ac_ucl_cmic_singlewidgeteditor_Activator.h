/*===================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center, 
Division of Medical and Biological Informatics.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without 
even the implied warranty of MERCHANTABILITY or FITNESS FOR 
A PARTICULAR PURPOSE.

See LICENSE.txt or http://www.mitk.org for details.

===================================================================*/


#ifndef uk_ac_ucl_cmic_singlewidgeteditor_Activator_H_
#define uk_ac_ucl_cmic_singlewidgeteditor_Activator_H_

#include <ctkPluginActivator.h>

/**
 * \ingroup uk_ac_ucl_cmic_singlewidgeteditor
 */
class uk_ac_ucl_cmic_singlewidgeteditor_Activator : public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)

public:

  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);

};

#endif /* uk_ac_ucl_cmic_singlewidgeteditor_Activator_H_ */

