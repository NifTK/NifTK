/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKNIFTYIGIAPPLICATIONPLUGIN_H_
#define QMITKNIFTYIGIAPPLICATIONPLUGIN_H_

#include "QmitkCommonAppsApplicationPlugin.h"

/**
 * \class QmitkNiftyIGIApplicationPlugin
 * \brief Implements QT and CTK specific functionality to launch the application as a plugin.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftyigi_internal
 */
class QmitkNiftyIGIApplicationPlugin : public QmitkCommonAppsApplicationPlugin
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)
  
public:

  QmitkNiftyIGIApplicationPlugin();
  ~QmitkNiftyIGIApplicationPlugin();

  void start(ctkPluginContext*);
  void stop(ctkPluginContext*);

protected:

  /// \brief Called by framework to get a URL for help system.
  virtual QString GetHelpHomePageURL() const;

private:

};

#endif /* QMITKNIFTYIGIAPPLICATIONPLUGIN_H_ */
