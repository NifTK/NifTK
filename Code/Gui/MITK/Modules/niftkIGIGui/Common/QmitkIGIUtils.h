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
#include <vtkMatrix4x4.h>
#include <QmitkDataStorageCheckableComboBox.h>

/**
 * \brief For testing purposes, loads an STL file.
 * \param surfaceFilename the full filename
 */
NIFTKIGIGUI_EXPORT mitk::Surface::Pointer LoadSurfaceFromSTLFile(QString& surfaceFilename);

/**
 * \brief Creates a test NDI Polaris Vicra message, used for initial testing.
 */
NIFTKIGIGUI_EXPORT QString CreateTestDeviceDescriptor();

/**
 * \brief Saves the matrix to file, returning true if successful and false otherwise.
 * TODO: Move this function somewhere sensible.
 */
NIFTKIGIGUI_EXPORT bool SaveMatrixToFile(const vtkMatrix4x4& matrix, const QString& fileName);

/**
 * \brief Saves the matrix to file, returning true if successful and false otherwise.
 * TODO: Move this function somewhere sensible.
 */
NIFTKIGIGUI_EXPORT void ApplyMatrixToNodes(const vtkMatrix4x4& matrix, const QmitkDataStorageCheckableComboBox& comboBox);

#endif
