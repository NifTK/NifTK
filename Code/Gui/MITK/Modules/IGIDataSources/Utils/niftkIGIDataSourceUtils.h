/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIDataSourceUtils_h
#define niftkIGIDataSourceUtils_h

#include <niftkIGIDataSourcesExports.h>
#include <niftkIGIDataType.h>
#include <set>
#include <QDir>
#include <QString>
#include <QMap>

/**
 * \file mitkIGIDataSourceUtils.h
 * \brief Some useful functions to help process IGI Data Sources
 */
namespace niftk
{

/**
* \brief Scans the directory for individual files that match a timestamp pattern.
* \param suffix for example ".jpg" or "-ultrasoundImage.nii".
*/
NIFTKIGIDATASOURCES_EXPORT
std::set<niftk::IGIDataType::IGITimeType> ProbeTimeStampFiles(QDir path, const QString& suffix);

/**
* \brief Returns the platform specific directory separator.
*/
NIFTKIGIDATASOURCES_EXPORT
QString GetPreferredSlash();

/**
* \brief Returns the list of timestamps, by source name.
*/
NIFTKIGIDATASOURCES_EXPORT
QMap<QString, std::set<niftk::IGIDataType::IGITimeType> > GetPlaybackIndex(
    const QString& directory, const QString& fileExtension);

/**
* \brief Returns the minimum and maximum timestamped of all files under the specified
* path, with the specified fileExtension, that look like they are timestamped.
*/
NIFTKIGIDATASOURCES_EXPORT
bool ProbeRecordedData(const QString& path,
                       const QString& fileExtension,
                       niftk::IGIDataType::IGITimeType* firstTimeStampInStore,
                       niftk::IGIDataType::IGITimeType* lastTimeStampInStore);

} // end namespace

#endif
