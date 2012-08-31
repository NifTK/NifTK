/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : $Author: mjc $

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKIGIUTILS_H
#define QMITKIGIUTILS_H

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
