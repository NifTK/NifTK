/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGIUtils_h
#define QmitkIGIUtils_h

#include "niftkIGIGuiExports.h"
#include <mitkSurface.h>
#include <QString>

/**
 * \brief For testing purposes, loads an STL file.
 * \param surfaceFilename the full filename
 */
NIFTKIGIGUI_EXPORT mitk::Surface::Pointer LoadSurfaceFromSTLFile(QString& surfaceFilename);

/**
 * \brief Creates a test NDI Polaris Vicra message, used for initial testing.
 */
NIFTKIGIGUI_EXPORT QString CreateTestDeviceDescriptor();

#endif
