/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkOpenCVFileIOUtils_h
#define mitkOpenCVFileIOUtils_h

#include "niftkOpenCVUtilsExports.h"
#include <mitkTimeStampsContainer.h>
#include <mitkOpenCVPointTypes.h>
#include <highgui.h>

/**
 * \file mitkOpenCVFileIOUtils.h
 * \brief Various simple file IO utils, that may use open CV data types in their signatures.
 */
namespace mitk {

/**
 * \brief Iterates through a directory to see if it contains any files that have a timestamp as a name, and end in .txt
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT bool CheckIfDirectoryContainsTrackingMatrices(const std::string& directory);

/**
 * \brief Recursively hunts for all directories that look like they contain tracking matrices, <timestamp>.txt.
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT std::vector<std::string> FindTrackingMatrixDirectories(const std::string& directory);

/**
 * \brief Returns an mitk::TimeStampsContainer containing all the timestamps of tracking matrices.
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT mitk::TimeStampsContainer FindTrackingTimeStamps(std::string directory);

/**
 * \brief Recursively hunts for all files that look like they are a video frame map file, (.+)(framemap.log).
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT std::vector<std::string> FindVideoFrameMapFiles(const std::string directory);

/**
 * \brief Extracted from mitkVideoTrackerMatching, reads a 4x4 matrix into a cv::Mat, and if the matrix can't
 * be read, will return a new matrix thats initialised according to the default OpenCV macros (i.e. unitinitialised).
 * @return true if successful and false otherwise
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT bool ReadTrackerMatrix(const std::string& filename, cv::Mat& outputMatrix);

/**
 * \see ReadTrackerMatrix(const std::string& filename, cv::Mat& outputMatrix);
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT bool ReadTrackerMatrix(const std::string& filename, cv::Matx44d& outputMatrix);

/**
 * \brief Saves a 4x4 matrix;
 * @return true if successful and false otherwise
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT bool SaveTrackerMatrix(const std::string& filename, cv::Mat& outputMatrix);

/**
 * \brief See SaveTrackerMatrix(const std::string& filename, cv::Mat& outputMatrix);
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT bool SaveTrackerMatrix(const std::string& filename, cv::Matx44d& outputMatrix);

/**
 * \brief Attempts to open a video capture and checks for errors. see trac 3718. This 
 * attempts to avoid problems caused by the subtle decoding errors.
 * \param the filename
 * \param ignore errors, false by default
 * @return the video capture object
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT cv::VideoCapture* InitialiseVideoCapture(std::string filename, bool ignoreErrors = false);

/**
 * \brief Loads points from a directory, where each point is in a separate file, and the filename is a timestamp.
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT std::vector< std::pair<unsigned long long, cv::Point3d> > LoadTimeStampedPoints(const std::string& directory);

/**
 * \brief Loads points from a flat text file with each line having the timestamp, the triangulated point, then the left and right screen points
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT void LoadTimeStampedPoints(std::vector< std::pair<unsigned long long, cv::Point3d> >& points,
    std::vector <mitk::ProjectedPointPair >& screenPoints, const std::string& fileName);

/**
 * \brief Loads points from a flat text file with each line having the time stamp, followed by the on screen points
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT void LoadTimeStampedPoints(std::vector< std::pair<unsigned long long, cv::Point2d> >& points,
    const std::string& fileName);

/**
 * \brief Saves points to a flat text file.
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT void SaveTimeStampedPoints(const std::vector< std::pair<unsigned long long, cv::Point3d> >& points, const std::string& fileName);

/**
 * \brief Saves a vector of picked objects
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT void SavePickedObjects ( const std::vector < mitk::PickedObject > & points, std::ostream& os );

/**
 * \brief Loads a vector of picked objects 
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT void LoadPickedObjects (  std::vector < mitk::PickedObject > & points, std::istream& is );

/**
 * \brief Read a set of matrices, stored as plain text, 4x4 matrices from a directory and
 * put them in a vector of 4x4 cvMats
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT std::vector<cv::Mat> LoadMatricesFromDirectory (const std::string& fullDirectoryName);


/**
 * \brief Read a set of matrices, stored in openCV xml matrix format from a directory and
 * put them in a vector of 4x4 cvMats
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT std::vector<cv::Mat> LoadOpenCVMatricesFromDirectory (const std::string& fullDirectoryName);


 /**
  * \brief Load a set of matrices from a file describing the
  * extrinsic parameters of a standard camera calibration
  */
extern "C++" NIFTKOPENCVUTILS_EXPORT std::vector<cv::Mat> LoadMatricesFromExtrinsicFile (const std::string& fullFileName);


/**
  * \brief Load stereo camera parameters from a directory
  */
extern "C++" NIFTKOPENCVUTILS_EXPORT void LoadStereoCameraParametersFromDirectory (const std::string& directory,
  cv::Mat* leftCameraIntrinsic, cv::Mat* leftCameraDistortion,
  cv::Mat* rightCameraIntrinsic, cv::Mat* rightCameraDisortion,
  cv::Mat* rightToLeftRotationMatrix, cv::Mat* rightToLeftTranslationVector,
  cv::Mat* leftCameraToTracker);


/**
 * \brief Load camera intrinsics from a plain text file and return results as
 * cv::Mat
 * \param cameraIntrinsic 3x3 matrix (double!)
 * \param cameraDistortion is optional, number of components needs to match the file! (double!)
 * \throws exception if parsing fails for any reason.
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT void LoadCameraIntrinsicsFromPlainText ( const std::string& filename,
  cv::Mat* cameraIntrinsic, cv::Mat* cameraDistortion);


/**
 * \brief Load stereo camera parameters from a plain text file
 * cv::Mat
 * \throws exception if parsing fails for any reason.
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT void LoadStereoTransformsFromPlainText ( const std::string& filename,
  cv::Mat* rightToLeftRotationMatrix, cv::Mat* rightToLeftTranslationVector);


/**
 * \brief Load the handeye matrix from a plain text file
 * cv::Mat
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT void LoadHandeyeFromPlainText ( const std::string& filename,
    cv::Mat* leftCameraToTracker);

/**
 * \brief Create an mitk::PickedPointList object from a directory of mitk point files
 * directory contains files line_00.mps .. line_nn.mps and points.mps. Point ID's are stored
 * in the file, while line ID's are stored in the file name
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT void LoadPickedPointListFromDirectory (
    const std::string& directoryName, mitk::PickedPointList&  pointList );

} // end namespace

#endif



