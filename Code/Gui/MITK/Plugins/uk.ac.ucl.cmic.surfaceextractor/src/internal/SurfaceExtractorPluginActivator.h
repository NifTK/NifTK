/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-01 19:03:07 +0100 (Fri, 01 Jul 2011) $
 Revision          : $Revision: 6628 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef SurfaceExtractorPluginActivator_h
#define SurfaceExtractorPluginActivator_h

#include <ctkPluginActivator.h>

namespace mitk {

/**
 * \class SurfaceExtractorPluginActivator
 * \brief CTK Plugin Activator class for the Surface Extractor plugin.
 * \ingroup uk_ac_ucl_cmic_surfaceextractor_internal
 */
class SurfaceExtractorPluginActivator :
  public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)

public:

  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);

}; // SurfaceExtractorPluginActivator

}

#endif // SurfaceExtractorPluginActivator_h
