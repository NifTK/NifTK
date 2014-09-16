/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkEvaluateIntrinsicParametersOnNumberOfFrames_h
#define mitkEvaluateIntrinsicParametersOnNumberOfFrames_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <itkObjectFactoryBase.h>
#include <mitkCommon.h>
#include <mitkVector.h>
#include <cv.h>

namespace mitk {

/**
 * \class EvaluateIntrinsicParametersOnNumberOfFrames_h
 * \brief Computes the tracking matrix for each frame of the video. 
 */
class NIFTKOPENCV_EXPORT EvaluateIntrinsicParametersOnNumberOfFrames : public itk::Object
{

public:

  mitkClassMacro(EvaluateIntrinsicParametersOnNumberOfFrames, itk::Object);
  itkNewMacro(EvaluateIntrinsicParametersOnNumberOfFrames);
  
  void InitialiseOutputDirectory();
  void InitialiseVideo();
  void RunExperiment();
  void Report();

  itkSetMacro(InputDirectory, std::string);
	itkSetMacro(InputMatrixDirectory, std::string);
  itkSetMacro(OutputDirectory, std::string);
	
  itkSetMacro(FramesToUse, unsigned int);
  itkSetMacro(AbsTrackerTimingError,long long);

  itkGetMacro(VideoInitialised, bool);
  itkGetMacro(TrackingDataInitialised, bool);
	itkSetMacro(SwapVideoChannels, bool);
  
  itkSetMacro(NumberCornersWidth, unsigned int);
  itkSetMacro(NumberCornersHeight, unsigned int);
  itkSetMacro(SquareSizeInMillimetres, double);
  itkSetMacro(PixelScaleFactor, mitk::Point2D);
	
  itkSetMacro(OptimiseIntrinsics, bool);

  
protected:

  EvaluateIntrinsicParametersOnNumberOfFrames();
  virtual ~EvaluateIntrinsicParametersOnNumberOfFrames();

  EvaluateIntrinsicParametersOnNumberOfFrames(const EvaluateIntrinsicParametersOnNumberOfFrames&); // Purposefully not implemented.
  EvaluateIntrinsicParametersOnNumberOfFrames& operator=(const EvaluateIntrinsicParametersOnNumberOfFrames&); // Purposefully not implemented.

private:

  void LoadVideoData(std::string filename);
  void ReadTimeStampFromLogFile();
	void MatchingVideoFramesToTrackingMatrix();
  void ComputeCamaraIntrinsicParameters(cv::Size &imageSize);

  void PairewiseFiles(std::vector<std::string> &fnvec_limg, 
                      std::vector<std::string> &fnvec_rimg, 
                      std::vector<std::string> &fnvec_lobj, 
                      std::vector<std::string> &fnvec_robj, 
                      std::vector<std::string> &fnvec_mtx);
  int Read2DPointsFromFile(const char *fn, std::vector<float*> &ptvec);
  int Read3DPointsFromFile(const char *fn, std::vector<float*> &ptvec);
  void ReadIntrinsicAndDistortionFromFile(const char *fn, 
                                          std::vector<float*> &intrinsicvec, 
                                          std::vector<float*> &distortionvec);
  
	std::string                         m_InputDirectory;
  std::string                         m_InputMatrixDirectory; 
	std::string                         m_OutputDirectory;
  bool                                m_SwapVideoChannels;
  unsigned int                        m_FramesToUse; //target frames to use 

  bool                                m_VideoInitialised;
  bool                                m_TrackingDataInitialised;
  long long                           m_AbsTrackerTimingError;
  
  unsigned int                        m_NumberCornersWidth;
  unsigned int                        m_NumberCornersHeight;
  double                              m_SquareSizeInMillimetres;
  mitk::Point2D                       m_PixelScaleFactor;
	
	CvMat*                              m_IntrinsicMatrixLeft;
  CvMat*                              m_IntrinsicMatrixRight;
  CvMat*                              m_DistortionCoefficientsLeft;
  CvMat*                              m_DistortionCoefficientsRight;
	
  bool                                m_OptimiseIntrinsics;
	bool                                m_OptimiseRightToLeft;
  bool                                m_PostProcessExtrinsicsAndR2L;
  bool                                m_PostProcessR2LThenExtrinsics;

  int                                 m_NumberOfFrames;
  std::vector<unsigned long long>     m_TimeStamps;
	std::vector<unsigned long long>     m_MatchedVideoFrames;
	//std::vector<std::string>            m_MatchedTrackingMatrix;
	std::vector<unsigned long long>     m_MatchedTrackingMatrix;
	
	std::string 												m_LeftDirectory;
	std::string 												m_RightDirectory;

}; // end class

} // end namespace

#endif
