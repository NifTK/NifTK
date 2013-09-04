/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkFileHelper_h
#define niftkFileHelper_h

#include <NifTKConfigure.h>
#include "niftkCommonWin32ExportHeader.h"

#define BOOST_FILESYSTEM_VERSION 2
#include <boost/filesystem.hpp>

#if (defined(WIN32) || defined(_WIN32))
#define FILE_SEPARATOR "\\"
#else
#define FILE_SEPARATOR "/"
#endif

#include "Exceptions/niftkIOException.h"

namespace niftk
{

  /**
   * Converts pathName to a boost::filesystem::path.
   * We throw std::logic_error if you pass an empty pathName.
   * \param pathName a string representing the path
   * \return boost path type
   */
  NIFTKCOMMON_WINEXPORT boost::filesystem::path ConvertToFullPath(const std::string& pathName);

  /**
   * Converts pathName to a full string by calling the above method.
   * \param pathName a string representing the path
   * \return a string representing the path
   */
  NIFTKCOMMON_WINEXPORT std::string ConvertToFullNativePath(const std::string& pathName);

  /**
   * Returns the file separator.
   * \return a / or a \ depending on OS
   */
  NIFTKCOMMON_WINEXPORT std::string GetFileSeparator();

  /**
   * Concatenates name onto path, taking care of file separator, its just
   * string concatenation, we don't check if the resulting value actually exists.
   * \param path a string representing the path
   * \param name a string representing a file or directory to be appended.
   */
  NIFTKCOMMON_WINEXPORT std::string ConcatenatePath(const std::string& path, const std::string& name);

  /**
   * \return True if directory exists, and false otherwise.
   */
  NIFTKCOMMON_WINEXPORT bool DirectoryExists(const std::string& directoryPath);

  /**
   * Creates a directory including all parents that don't exist.
   * \param the directory name
   * \return true if creation was successfull
   */
  NIFTKCOMMON_WINEXPORT bool CreateDirectoryAndParents(const std::string& directoryPath);

  /**
   * Creates a unique file name for a file located in the O/S temporary directory.
   * \param prefix file basename prefix
   * \param suffix file basename suffix
   * \return a unique file name
   */
  NIFTKCOMMON_WINEXPORT boost::filesystem::path CreateUniqueTempFileName(const std::string &prefix, const std::string &suffix = "") throw (niftk::IOException);

  /**
   * \return True if file exists, and false otherwise.
   */
  NIFTKCOMMON_WINEXPORT bool FileExists(const std::string& fileName);

  /**
   * \return True if file has size of zero, and false otherwise.
   */
  NIFTKCOMMON_WINEXPORT bool FileIsEmpty(const std::string& fileName);

  /**
   * \return the file size in bytes.
   */
  NIFTKCOMMON_WINEXPORT int FileSize(const std::string& fileName);

  /**
   * This method is used to find the plugin config files.
   * So you look for stuff with a given prefix (usually blank)
   * and the correct filename extension. ie. *.plg
   * \param a filename
   * \param normally blank
   * \param extension for example ".plg"
   * \return true for a match and false otherwise
   */
  NIFTKCOMMON_WINEXPORT bool FilenameHasPrefixAndExtension(
      const std::string& filename,
      const std::string& prefix,
      const std::string& extension);

  /**
   * Then this is used to find a library, as it must
   * specifically have a prefix, then the library name,
   * and also the correct extension.
   * \param filename a filename
   * \param prefix for example "lib" on unix and nothing on windows
   * \param middle for example "niftkonlinehelp" as specified in plugin config file
   * \param extension for example .so on Unix and .dll on Windows.
   * \return true for a match and false otherwise
   */
  NIFTKCOMMON_WINEXPORT bool FilenameMatches(
      const std::string& filename,
      const std::string& prefix,
      const std::string& middle,
      const std::string& extension);

  /**
   * Helper method to get the image directory, as it works off an environment variable.
   * \return NIFTK_HOME/images
   */
  NIFTKCOMMON_WINEXPORT std::string GetImagesDirectory();

  /**
   * Returns all files in a given directory, or empty list if none found.
   * \param fullDirectoryName Directory name
   * \throw logic_error if directory name is invalid
   * \return a list of files within that folder.
   */
  NIFTKCOMMON_WINEXPORT std::vector<std::string> GetFilesInDirectory(const std::string& fullDirectoryName);

  /**
   * Returns all files in a given directory and recursively in all sub-directories, or empty list if none found.
   * \param fullDirectoryName Directory name
   * \param fileNames The list of files found
   * \throw logic_error if directory name is invalid
   */
  NIFTKCOMMON_WINEXPORT void GetRecursiveFilesInDirectory(const std::string& fullDirectoryName, std::vector<std::string> &fileNames);

} // end namespace


#endif
