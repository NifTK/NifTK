/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKSTEREOIMAGETOMODELMETRIC_H
#define MITKSTEREOIMAGETOMODELMETRIC_H

#include "itkExceptionObject.h"
#include "itkSingleValuedCostFunction.h"
#include <vtkPolyDataReader.h>
#include <cv.h>
#include <highgui.h>

namespace mitk
{

/**
 * \class StereoImageToModelMetric
 * \brief Abstract base class to compute similarity between a model (currently VTK), and two stereo images.
 */

class StereoImageToModelMetric : public itk::SingleValuedCostFunction
{
public:

  /** Standard class typedefs. */
  typedef StereoImageToModelMetric       Self;
  typedef itk::SingleValuedCostFunction  Superclass;
  typedef itk::SmartPointer<Self>        Pointer;
  typedef itk::SmartPointer<const Self>  ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(StereoImageToModelMetric, itk::SingleValuedCostFunction);

  /**  Type of the measure. */
  typedef Superclass::MeasureType        MeasureType;

  /**  Type of the derivative. */
  typedef Superclass::DerivativeType     DerivativeType;

  /**  Type of the parameters. */
  typedef Superclass::ParametersType     ParametersType;

  /** Set/Get the parameters defining the Transform. */
  itkSetMacro(Parameters, ParametersType);
  itkGetMacro(Parameters, ParametersType);

  itkSetMacro(DrawOutput, bool);
  itkGetMacro(DrawOutput, bool);

  itkSetStringMacro(Input3DModelFileName);
  itkGetStringMacro(Input3DModelFileName);

  itkSetStringMacro(InputLeftImageFileName);
  itkGetStringMacro(InputLeftImageFileName);

  itkSetStringMacro(InputRightImageFileName);
  itkGetStringMacro(InputRightImageFileName);

  itkSetStringMacro(OutputLeftImageFileName);
  itkGetStringMacro(OutputLeftImageFileName);

  itkSetStringMacro(OutputRightImageFileName);
  itkGetStringMacro(OutputRightImageFileName);

  itkSetStringMacro(IntrinsicLeftFileName);
  itkGetStringMacro(IntrinsicLeftFileName);

  itkSetStringMacro(DistortionLeftFileName);
  itkGetStringMacro(DistortionLeftFileName);

  itkSetStringMacro(RotationLeftFileName);
  itkGetStringMacro(RotationLeftFileName);

  itkSetStringMacro(TranslationLeftFileName);
  itkGetStringMacro(TranslationLeftFileName);

  itkSetStringMacro(IntrinsicRightFileName);
  itkGetStringMacro(IntrinsicRightFileName);

  itkSetStringMacro(DistortionRightFileName);
  itkGetStringMacro(DistortionRightFileName);

  itkSetStringMacro(RightToLeftRotationFileName);
  itkGetStringMacro(RightToLeftRotationFileName);

  itkSetStringMacro(RightToLeftTranslationFileName);
  itkGetStringMacro(RightToLeftTranslationFileName);

  /** Return the number of parameters required by the Transform (normally 6). */
  unsigned int GetNumberOfParameters(void) const { return m_Parameters.size(); }

  /** Initialize the Metric by making sure that all the components are present and plugged together correctly */
  virtual void Initialize();

  /**
   * \brief \see itk::SingleValuedCostFunction::GetValue() and derived classes should implement CalculateCost(const ParametersType & parameters );
   */
  virtual MeasureType GetValue( const ParametersType &parameters ) const ;

  /**
   * \brief \see itk::SingleValuedCostFunction::GetDerivative()
   */
  virtual void GetDerivative( const ParametersType & parameters,
                              DerivativeType & derivative ) const;

protected:

  StereoImageToModelMetric();
  virtual ~StereoImageToModelMetric();
  void PrintSelf(std::ostream& os, itk::Indent indent) const;

  vtkPolyDataReader *m_PolyDataReader;
  IplImage *m_InputLeftImage;
  IplImage *m_InputRightImage;
  CvMat *m_IntrinsicLeft;
  CvMat *m_DistortionLeft;
  CvMat *m_RotationLeft;
  CvMat *m_TranslationLeft;
  CvMat *m_IntrinsicRight;
  CvMat *m_DistortionRight;
  CvMat *m_RightToLeftRotation;
  CvMat *m_RightToLeftTranslation;
  CvMat *m_ModelPoints;
  CvMat *m_ModelNormals;
  CvMat *m_CameraNormal;
  IplImage *m_OutputLeftImage;
  IplImage *m_OutputRightImage;
  mutable ParametersType m_Parameters;

  virtual MeasureType CalculateCost(
      const CvMat& transformed3DPoints,
      const CvMat& transformed3DNormals,
      const CvMat& weights,
      const CvMat& transformed2DPointsLeft,
      const CvMat& transformed2DPointsRight,
      const ParametersType &parameters
      ) const = 0;

  /**
   * \brief Retrieves the value from the images for both left and right images.
   */
  virtual bool GetImageValues(const float &lx, const float &ly, const float &rx, const float &ry, float *leftValue, float *rightValue) const;

  /**
   * \brief Retrieves the value from a single image, using bi-linear interpolation, over n channels.
   */
  bool GetImageValue(const IplImage* image, const float &x, const float &y, float *imageValue) const;

private:

  StereoImageToModelMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  bool m_DrawOutput;
  std::string m_Input3DModelFileName;
  std::string m_InputLeftImageFileName;
  std::string m_InputRightImageFileName;
  std::string m_OutputLeftImageFileName;
  std::string m_OutputRightImageFileName;
  std::string m_IntrinsicLeftFileName;
  std::string m_DistortionLeftFileName;
  std::string m_RotationLeftFileName;
  std::string m_TranslationLeftFileName;
  std::string m_IntrinsicRightFileName;
  std::string m_DistortionRightFileName;
  std::string m_RightToLeftRotationFileName;
  std::string m_RightToLeftTranslationFileName;
}; // end class

} // end namespace mitk

#endif

