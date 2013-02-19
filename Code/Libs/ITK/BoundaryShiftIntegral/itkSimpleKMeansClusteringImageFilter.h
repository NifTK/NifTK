/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKSIMPLEKMEANSCLUSTERINGIMAGEFILTER_H_
#define ITKSIMPLEKMEANSCLUSTERINGIMAGEFILTER_H_

#include "itkImageToImageFilter.h"
#include "itkMacro.h"
#include "itkWeightedCentroidKdTreeGenerator.h"
#include "itkKdTreeBasedKmeansEstimator.h"
#include "itkListSample.h"

namespace itk
{

/**
 * K-Means clustering with a mask. 
 */
template <class TInputImage, class TInputMask, class TOutputImage>
class ITK_EXPORT SimpleKMeansClusteringImageFilter: 
  public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /**
   * House keeping for the object factory. 
   */ 
  typedef SimpleKMeansClusteringImageFilter Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;
  /** 
   * Method for creation through the object factory. 
   */
  itkNewMacro(Self);  
  /** 
   * Runtime information support. 
   */
  itkTypeMacro(SimpleKMeansClusteringImageFilter, ImageToImageFilter);
  /**
   * Typedefs. 
   */ 
  typedef typename TInputImage::Pointer TInputImagePointer;
  typedef typename TInputMask::Pointer TInputMaskPointer;
  typedef itk::Vector< double, 1 > MeasurementVectorType ;
  typedef typename itk::Statistics::ListSample< MeasurementVectorType > SampleType ;
  typedef itk::Statistics::WeightedCentroidKdTreeGenerator<SampleType> TreeGeneratorType;
  typedef typename TreeGeneratorType::KdTreeType TreeType;
  typedef itk::Statistics::KdTreeBasedKmeansEstimator<TreeType> EstimatorType;
  typedef typename EstimatorType::ParametersType ParametersType;
  /**
   * Get/Set functions. 
   */
  itkSetMacro(InputMask, TInputMaskPointer);
  itkGetMacro(InputMask, TInputMaskPointer);
  itkSetMacro(NumberOfClasses, unsigned int);
  itkGetMacro(NumberOfClasses, unsigned int);
  itkSetMacro(InitialMeans, ParametersType);
  itkGetMacro(InitialMeans, ParametersType);
  itkGetMacro(FinalMeans, ParametersType);
  itkGetMacro(FinalStds, ParametersType);
  itkGetMacro(RSS, double);
  itkGetMacro(NumberOfSamples, double);
  itkGetMacro(FinalClassSizes, ParametersType);
  
protected:
  /**
   * Constructor. 
   */
  SimpleKMeansClusteringImageFilter() { this->m_NumberOfClasses = 3; this->m_RSS = 0.0; this->m_NumberOfSamples = 0.0; }
  /**
   * Destructor. 
   */
  virtual ~SimpleKMeansClusteringImageFilter() {}
  /**
   * 
   */
  void GenerateData();
  
private:
  /**
   * The mask image. 
   */
  TInputMaskPointer m_InputMask; 
  /**
   * Number of classes we want to classify. 
   */
  unsigned int m_NumberOfClasses; 
  /**
   * Starting estimates of the means. 
   */
  ParametersType m_InitialMeans;
  /**
   *
   */  
  ParametersType m_FinalMeans;
  ParametersType m_FinalStds;
  ParametersType m_FinalClassSizes;
  /**
   * Residual sum of squared difference. 
   */
  double m_RSS; 
  /**
   * Number of samples. 
   */
  double m_NumberOfSamples; 
  

};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSimpleKMeansClusteringImageFilter.txx"
#endif

#endif 


