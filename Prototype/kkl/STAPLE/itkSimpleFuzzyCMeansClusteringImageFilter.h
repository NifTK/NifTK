/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 18:04:05 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3325 $
 Last modified by  : $Author: mjc $
 
 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKSimpleFuzzyCMEANSCLUSTERINGIMAGEFILTE_H_
#define ITKSimpleFuzzyCMEANSCLUSTERINGIMAGEFILTER_H_

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
template <class TInputImage, class TInputMask>
class ITK_EXPORT SimpleFuzzyCMeansClusteringImageFilter: 
  public ImageToImageFilter<TInputImage, Image<float, TInputImage::ImageDimension> >
{
public:
  /**
   * House keeping for the object factory. 
   */ 
  typedef SimpleFuzzyCMeansClusteringImageFilter Self;
  typedef ImageToImageFilter<TInputImage,Image<float, TInputImage::ImageDimension> > Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;
  /** 
   * Method for creation through the object factory. 
   */
  itkNewMacro(Self);  
  /** 
   * Runtime information support. 
   */
  itkTypeMacro(SimpleFuzzyCMeansClusteringImageFilter, ImageToImageFilter);
  /**
   * Typedefs. 
   */ 
  typedef typename TInputImage::Pointer TInputImagePointer;
  typedef typename TInputMask::Pointer TInputMaskPointer;
  typedef typename Superclass::OutputImageType TOutputImage; 
  typedef itk::Vector< float, 1 > MeasurementVectorType ;
  typedef typename itk::Statistics::ListSample< MeasurementVectorType > SampleType;  
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
  itkSetMacro(OutputFileNameFormat, typename std::string); 
  
protected:
  /**
   * Constructor. 
   */
  SimpleFuzzyCMeansClusteringImageFilter() { this->m_NumberOfClasses = 3; this->m_Fuzziness = 2.0; m_OutputFileNameFormat = "fuzzy-%d.hdr"; }
  /**
   * Destructor. 
   */
  virtual ~SimpleFuzzyCMeansClusteringImageFilter() {}
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
   * Controls the fuzziness of the classification. 
   * default: 2. 
   */
  typename MeasurementVectorType::ValueType m_Fuzziness; 
  /**
   * Output filename format. 
   */
  typename std::string m_OutputFileNameFormat; 

};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSimpleFuzzyCMeansClusteringImageFilter.txx"
#endif

#endif 


