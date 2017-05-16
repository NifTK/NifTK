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
#include <niftkDataStorageCheckableComboBox.h>
#include "igtlTypes.h"

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
 * This function also raises QMessage boxes, so, possibly no need to return the true/false status.
 */
NIFTKIGIGUI_EXPORT bool SaveMatrixToFile(const vtkMatrix4x4& matrix, const QString& fileName);

/**
 * \brief Applies (sets, i.e. copies) the given transform to all checked nodes in the comboBox.
 */
NIFTKIGIGUI_EXPORT void ApplyTransformToSelectedNodes(const vtkMatrix4x4& transform, const niftk::DataStorageCheckableComboBox& comboBox);

/**
 * \brief Composes the given transform with all checked nodes in the comboBox.
 */
NIFTKIGIGUI_EXPORT void ComposeTransformWithSelectedNodes(const vtkMatrix4x4& transform, const niftk::DataStorageCheckableComboBox& comboBox);

/**
 * \brief For visualisation purposes, creates a representation of the laparoscope.
 * \param the rigid body filename to define the location of the tracking markers
 * \param the handeye calibration to define the tool origin
 */
NIFTKIGIGUI_EXPORT mitk::Surface::Pointer MakeLaparoscope(QString& rigidBodyFilename, const vtkMatrix4x4& handeye );

/**
 * \brief For visualisation purposes, creates a representation of the pointer.
 * \param the rigid body filename to define the location of the tracking markers
 * \param the handeye calibration to define the tool origin
 */
NIFTKIGIGUI_EXPORT mitk::Surface::Pointer MakePointer(QString& rigidBodyFilename, const vtkMatrix4x4& handeye );

/**
 * \brief For visualisation purposes, creates a representation of the reference.
 * \param the rigid body filename to define the location of the tracking markers
 * \param the handeye calibration to define the tool origin
 */
NIFTKIGIGUI_EXPORT mitk::Surface::Pointer MakeReference(QString& rigidBodyFilename, const vtkMatrix4x4& handeye );

/**
 * \brief For visualisation purposes, make a wall of a cube
 * \param the size of the cube in mm 
 * \param which wall to make 
 * \param the xoffset, room will be centred at x= size * xOffset 
 * */
NIFTKIGIGUI_EXPORT mitk::Surface::Pointer MakeAWall( const int& whichwall, const float& size = 3000,
    const float& xOffset = 0.0 , const float& yOffset = 0.0, const float& zOffset = -0.3, 
    const float& thickness = 10.0);

/** 
 * \brief get the IRED positions from a rigid body definition file
 * \param the file name
 */
NIFTKIGIGUI_EXPORT std::vector<float [3]> ReadRigidBodyDefinitionFile(QString& rigidBodyFilename);

/**
 * \brief Converts an integer number of nanoseconds to a 9 digit string, prefixed with zeros.
 */
NIFTKIGIGUI_EXPORT QString ConvertNanoSecondsToString(const igtlUint32& nanosec);

/**
* \brief Returns a default path, to somewhere writable, like the desktop.
*/
NIFTKIGIGUI_EXPORT QString GetWritablePath(const char* defaultRecordingDir = nullptr);

/**
 * \brief Format's the time in millis to "yyyy.MM.dd_hh-mm-ss-zzz".
 */
NIFTKIGIGUI_EXPORT QString FormatDateTime(const qint64& timeInMillis);

NIFTKIGIGUI_EXPORT std::string FormatDateTimeAsStdString(const qint64& timeInMillis);

/**
* \brief Extracts a file from Qt resource file to a temporary file on file system.
* \return Full extracted file name.
*/
NIFTKIGIGUI_EXPORT QString ExtractTempFile(const QString& resourcePrefix, const QString& name);

/**
* \brief Increments a number on the end of a given string.
*
* note: we do not trim white-space!
* this is intentional: it allows the user to add a second dimension of numbers to it.
* for example:
* name="hello world"    --> "hello world 1"
* name="hello world 1"  --> "hello world 2"
* name "hello world 1 " --> "hello world 1 1"
*/
NIFTKIGIGUI_EXPORT std::string IncrementNodeName(const std::string& name);

#endif
