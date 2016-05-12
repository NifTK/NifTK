/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyCalVideoCalibrateManager_h
#define niftkNiftyCalVideoCalibrateManager_h

#include "niftkNiftyCalExports.h"
#include <itkObject.h>
#include <itkObjectFactoryBase.h>
#include <mitkDataStorage.h>
#include <mitkDataNode.h>
#include <niftkPointUtilities.h>
#include <niftkIPoint2DDetector.h>
#include <cv.h>
#include <list>

namespace niftk {

/**
 * \class NiftyCalVideoCalibrationManager
 * \brief Manager class to perform video calibration as provided by NiftyCal.
 *
 * This one is not an MITK Service as it is stateful. So, it would
 * be more problematic to have a system-wide service, called from multiple threads.
 */
class NIFTKNIFTYCAL_EXPORT NiftyCalVideoCalibrationManager : public itk::Object
{
public:

  enum CalibrationPatterns
  {
    // Order must match that in niftk::CameraCalViewPreferencePage
    CHESSBOARD,
    CIRCLE_GRID,
    APRIL_TAGS
  };

  enum HandEyeMethod
  {
    // Order must match that in niftk::CameraCalViewPreferencePage
    TSAI,
    DIRECT,
    MALTI
  };

  const static bool                DefaultDoIterative;
  const static unsigned int        DefaultMinimumNumberOfSnapshotsForCalibrating;
  const static double              DefaultScaleFactorX;
  const static double              DefaultScaleFactorY;
  const static int                 DefaultGridSizeX;
  const static int                 DefaultGridSizeY;
  const static CalibrationPatterns DefaultCalibrationPattern;
  const static HandEyeMethod       DefaultHandEyeMethod;
  const static std::string         DefaultTagFamily;

  mitkClassMacroItkParent(NiftyCalVideoCalibrationManager, itk::Object);
  itkNewMacro(NiftyCalVideoCalibrationManager);

  void SetDataStorage(const mitk::DataStorage::Pointer storage);

  void SetLeftImageNode(mitk::DataNode::Pointer node);
  mitk::DataNode::Pointer GetLeftImageNode() const;
  void SetRightImageNode(mitk::DataNode::Pointer node);
  mitk::DataNode::Pointer GetRightImageNode() const;
  itkSetMacro(TrackingTransformNode, mitk::DataNode::Pointer);
  itkGetMacro(TrackingTransformNode, mitk::DataNode::Pointer);

  itkSetMacro(MinimumNumberOfSnapshotsForCalibrating, unsigned int);
  itkGetMacro(MinimumNumberOfSnapshotsForCalibrating, unsigned int);
  itkSetMacro(DoIterative, bool);
  itkGetMacro(DoIterative, bool);
  itkSetMacro(ScaleFactorX, double);
  itkGetMacro(ScaleFactorX, double);
  itkSetMacro(ScaleFactorY, double);
  itkGetMacro(ScaleFactorY, double);
  itkSetMacro(GridSizeX, int);
  itkGetMacro(GridSizeX, int);
  itkSetMacro(GridSizeY, int);
  itkGetMacro(GridSizeY, int);
  itkSetMacro(CalibrationPattern, CalibrationPatterns);
  itkGetMacro(CalibrationPattern, CalibrationPatterns);
  itkSetMacro(HandeyeMethod, HandEyeMethod);
  itkGetMacro(HandeyeMethod, HandEyeMethod);
  itkSetMacro(TagFamily, std::string);
  itkGetMacro(TagFamily, std::string);

  void Set3DModelFileName(const std::string& fileName);
  itkGetMacro(3DModelFileName, std::string);

  itkSetMacro(OutputDirName, std::string);
  itkGetMacro(OutputDirName, std::string);

  void SetReferenceDataFileNames(const std::string& imageFileName, const std::string& pointsFileName);
  itkGetMacro(ReferenceImageFileName, std::string);

  void SetModelToTrackerFileName(const std::string& fileName);
  itkGetMacro(ModelToTrackerFileName, std::string);

  unsigned int GetNumberOfSnapshots() const;

  /**
   * \brief Clears down the internal points and image arrays,
   * so calibration will restart with zero data.
   */
  void Restart();

  /**
   * \brief Grabs images and tracking, and runs the point extraction.
   * \return Returns true if successful and false otherwise.
   */
  bool Grab();

  /**
   * \brief Removes the last grabbed snapshot.
   */
  void UnGrab();

  /**
   * \brief Performs the actual calibration.
   *
   * This can be mono, stereo, iterative and include hand-eye,
   * depending on the configuration parameters stored in this class.
   *
   * \return rms re-projection error (pixels)
   */
  double Calibrate();

  /**
   * \brief Saves a bunch of standard (from a NifTK perspective)
   * calibration files to the output dir, overwriting existing files.
   */
  void Save();

protected:

  NiftyCalVideoCalibrationManager(); // Purposefully hidden.
  virtual ~NiftyCalVideoCalibrationManager(); // Purposefully hidden.

  NiftyCalVideoCalibrationManager(const NiftyCalVideoCalibrationManager&); // Purposefully not implemented.
  NiftyCalVideoCalibrationManager& operator=(const NiftyCalVideoCalibrationManager&); // Purposefully not implemented.

private:

  void ConvertImage(mitk::DataNode::Pointer imageNode, cv::Mat& outputImage);
  bool ExtractPoints(int imageIndex, const cv::Mat& image);

  // Data from Plugin.
  mitk::DataStorage::Pointer m_DataStorage;
  mitk::DataNode::Pointer    m_ImageNode[2];
  mitk::DataNode::Pointer    m_TrackingTransformNode;

  // Data from preferences.
  bool                       m_DoIterative;
  unsigned int               m_MinimumNumberOfSnapshotsForCalibrating;
  std::string                m_3DModelFileName;
  double                     m_ScaleFactorX;
  double                     m_ScaleFactorY;
  int                        m_GridSizeX;
  int                        m_GridSizeY;
  CalibrationPatterns        m_CalibrationPattern;
  HandEyeMethod              m_HandeyeMethod;
  std::string                m_TagFamily;
  std::string                m_OutputDirName;
  std::string                m_ModelToTrackerFileName;
  std::string                m_ReferenceImageFileName;
  std::string                m_ReferencePointsFileName;

  // Data used for temporary storage
  cv::Mat                    m_TmpImage[2];

  // Data used for calibration.
  cv::Size2i                                                               m_ImageSize;
  std::pair< cv::Mat, niftk::PointSet>                                     m_ReferenceDataForIterativeCalib;
  cv::Matx44d                                                              m_3DModelToTracker;
  niftk::Model3D                                                           m_3DModelPoints;
  std::list<niftk::PointSet>                                               m_Points[2];
  std::list<std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat> > m_OriginalImages[2];
  std::list<std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat> > m_ImagesForWarping[2];
  std::list<cv::Matx44d >                                                  m_TrackingMatrices;

  // Calibration result
  cv::Mat                    m_Intrinsic[2];
  cv::Mat                    m_Distortion[2];
  std::vector<cv::Mat>       m_Rvecs[2];
  std::vector<cv::Mat>       m_Tvecs[2];
  cv::Mat                    m_EssentialMatrix;
  cv::Mat                    m_FundamentalMatrix;
  cv::Mat                    m_Left2RightRotation;
  cv::Mat                    m_Left2RightTranslation;

}; // end class

} // end namespace

#endif
