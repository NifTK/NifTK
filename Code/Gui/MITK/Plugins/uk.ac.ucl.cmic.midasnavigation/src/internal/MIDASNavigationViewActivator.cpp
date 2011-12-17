#include "MIDASNavigationViewActivator.h"

#include <QtPlugin>

#include "MIDASNavigationView.h"

namespace mitk {

ctkPluginContext* MIDASNavigationViewActivator::s_PluginContext(NULL);

void MIDASNavigationViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(MIDASNavigationView, context);
  s_PluginContext = context;
}

void MIDASNavigationViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

ctkPluginContext* MIDASNavigationViewActivator::GetPluginContext()
{
  return s_PluginContext;
}

}

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_midasnavigation, mitk::MIDASNavigationViewActivator)
