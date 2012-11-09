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

#include "QmitkCommonAppsApplicationPlugin.h"
#include <itkImage.h>

namespace mitk {
  class DataNode;
}

/**
 * \class QmitkNiftyViewApplicationPlugin
 * \brief Implements QT and CTK specific functionality to launch the application as a plugin.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftyview_internal
 */
class QmitkNiftyViewApplicationPlugin : public QmitkCommonAppsApplicationPlugin
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)
  
public:

  QmitkNiftyViewApplicationPlugin();
  ~QmitkNiftyViewApplicationPlugin();

  void start(ctkPluginContext*);

protected:

  /// \brief Called each time a data node is added, so we make sure it is initialised with a Window/Level.
  virtual void NodeAdded(const mitk::DataNode *node);

  /// \brief Called by framework to get a URL for help system.
  virtual QString GetHelpHomePageURL() const;

private:

  template<typename TPixel, unsigned int VImageDimension>
  void
  ITKGetStatistics(
      itk::Image<TPixel, VImageDimension> *itkImage,
      float &min,
      float &max,
      float &mean,
      float &stdDev);
};

#endif /* QMITKEXTAPPLICATIONPLUGIN_H_ */
