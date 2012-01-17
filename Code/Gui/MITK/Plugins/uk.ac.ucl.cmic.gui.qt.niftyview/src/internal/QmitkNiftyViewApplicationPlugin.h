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


#ifndef QMITKNIFTYVIEWAPPLICATIONPLUGIN_H_
#define QMITKNIFTYVIEWAPPLICATIONPLUGIN_H_

#include <berryAbstractUICTKPlugin.h>

#include <QString>

#include <berryQCHPluginListener.h>

/**
 * \class QmitkNiftyViewApplicationPlugin
 * \brief Implements QT and CTK specific functionality to launch the application as a plugin.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftyview_internal
 */
class QmitkNiftyViewApplicationPlugin : public QObject, public berry::AbstractUICTKPlugin
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)
  
public:

  QmitkNiftyViewApplicationPlugin();
  ~QmitkNiftyViewApplicationPlugin();

  static QmitkNiftyViewApplicationPlugin* GetDefault();

  ctkPluginContext* GetPluginContext() const;

  void start(ctkPluginContext*);

  QString GetQtHelpCollectionFile() const;

  // Currently, creating state machine using hard coded string, as I don't know where to load them from.
  static const std::string MIDAS_SEED_TOOL_STATE_MACHINE_XML;

  // Currently, creating state machine using hard coded string, as I don't know where to load them from.
  static const std::string MIDAS_SEED_DROPPER_STATE_MACHINE_XML;

  // Currently, creating state machine using hard coded string, as I don't know where to load them from.
  static const std::string MIDAS_POLY_TOOL_STATE_MACHINE_XML;

  // Currently, creating state machine using hard coded string, as I don't know where to load them from.
  static const std::string MIDAS_DRAW_TOOL_STATE_MACHINE_XML;

  // Currently, creating state machine using hard coded string, as I don't know where to load them from.
  static const std::string MIDAS_PAINTBRUSH_TOOL_STATE_MACHINE_XML;

  // Currently, creating state machine using hard coded string, as I don't know where to load them from.
  static const std::string MIDAS_KEYPRESS_STATE_MACHINE_XML;

private:

  static QmitkNiftyViewApplicationPlugin* inst;

  ctkPluginContext* context;
  berry::QCHPluginListener* pluginListener;

  mutable QString helpCollectionFile;
};

#endif /* QMITKEXTAPPLICATIONPLUGIN_H_ */
