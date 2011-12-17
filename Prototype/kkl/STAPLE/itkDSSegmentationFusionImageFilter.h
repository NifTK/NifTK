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
#ifndef __itkDSSegmentationFusionImageFilter_h
#define __itkDSSegmentationFusionImageFilter_h

#include "itkImage.h"
#include "itkImageToImageFilter.h"
#include "itkIdentityTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkNormalizedMutualInformationHistogramImageToImageMetric.h"
#include "itkCorrelationCoefficientHistogramImageToImageMetric.h"
#include "itkMeanSquaresHistogramImageToImageMetric.h"

namespace itk
{

/**
 * DSSegmentationFusionImageFilter: just copied from LabelVotingImageFilter
 * with added option to pick a random label when votes are equal. 
 */
template <typename TInputImage, typename TOutputImage>
class ITK_EXPORT DSSegmentationFusionImageFilter :
    public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef DSSegmentationFusionImageFilter Self;
  typedef ImageToImageFilter< TInputImage, TOutputImage > Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods) */
  itkTypeMacro(DSSegmentationFusionImageFilter, ImageToImageFilter);
  
  /** Extract some information from the image types.  Dimensionality
   * of the two images is assumed to be the same. */
  typedef typename TOutputImage::PixelType OutputPixelType;
  typedef typename TInputImage::PixelType  InputPixelType;
  
  /** Extract some information from the image types.  Dimensionality
   * of the two images is assumed to be the same. */
  itkStaticConstMacro(InputImageDimension, unsigned int, TInputImage::ImageDimension );
  itkStaticConstMacro(ImageDimension, unsigned int, TOutputImage::ImageDimension);
  
  /** Image typedef support */
  typedef TInputImage InputImageType;
  typedef TOutputImage OutputImageType;
  typedef typename InputImageType::ConstPointer InputImagePointer;
  typedef typename OutputImageType::Pointer OutputImagePointer;
  typedef Image<float, ImageDimension> ConflictImageType; 
  typedef Image<float, ImageDimension> BeliefImageType; 
  
  /**
   * Similarity typedefs. 
   */
  typedef HistogramImageToImageMetric<TInputImage,TInputImage> HistogramImageToImageMetricType; 
  typedef CorrelationCoefficientHistogramImageToImageMetric<TInputImage,TInputImage> CorrelationCoefficientHistogramImageToImageMetricType;
  typedef NormalizedMutualInformationHistogramImageToImageMetric<TInputImage,TInputImage> NormalizedMutualInformationHistogramImageToImageMetricType; 
  typedef MeanSquaresHistogramImageToImageMetric<TInputImage,TInputImage> MeanSquaresHistogramImageToImageMetricType; 
  typedef IdentityTransform<double, ImageDimension> TransformType; 
  typedef NearestNeighborInterpolateImageFunction<TInputImage, double> InterpolatorType;
  
  /** Superclass typedefs. */
  typedef typename Superclass::OutputImageRegionType OutputImageRegionType;
  
  /** 
   * Set label value for undecided pixels.
   */
  virtual void SetLabelForUndecidedPixels( const OutputPixelType l )
  {
    this->m_LabelForUndecidedPixels = l;
    this->m_HasLabelForUndecidedPixels = true;
    this->Modified();
  }
  
  /** Get label value used for undecided pixels.
   * After updating the filter, this function returns the actual label value
   * used for undecided pixels in the current output. Note that this value
   * is overwritten when SetLabelForUndecidedPixels is called and the new
   * value only becomes effective upon the next filter update.
   */
  virtual OutputPixelType GetLabelForUndecidedPixels() const
  {
    return this->m_LabelForUndecidedPixels;
  }
  /** 
   * Unset label value for undecided pixels and turn on automatic selection.
   */
  virtual void UnsetLabelForUndecidedPixels()
  {
    if ( this->m_HasLabelForUndecidedPixels )
    {
      this->m_HasLabelForUndecidedPixels = false;
      this->Modified();
    }
  }
  /**
   * Set the global segmentation reliability.
   */
  virtual void SetSegmentationReliability(int key, double reliability)
  {
    m_SegmentationReliability[key] = reliability; 
  }
  /**
   * Return the conflict image. 
   */
  virtual typename ConflictImageType::ConstPointer GetConflictImage() 
  { 
    return this->m_ConflictImage.GetPointer(); 
  }
  /**
   * Set the target image. 
   */
  virtual void SetTargetImage(typename TInputImage::Pointer target)
  {
    this->m_TargetImage = target; 
  }
  /**
   * Set up all the registered images. 
   */
  virtual void SetRegisteredAtlases(int key, typename TInputImage::Pointer atlas)
  {
    this->m_RegisteredAtlases[key] = atlas; 
  }
  /**
   * Set the reliability mode. 
   */
  virtual void SetReliabilityMode(int mode)
  {
    m_ReliabilityMode = mode; 
  }
  
  itkSetMacro(CombinationMode,  int); 
  itkSetMacro(LocalRegionRadius, unsigned int); 
  itkSetMacro(Gain, double); 
  itkSetMacro(PlausibilityThreshold, double); 
  itkGetMacro(ForegroundBeliefImage, typename BeliefImageType::Pointer);
  itkGetMacro(ForegroundPlausibilityImage, typename BeliefImageType::Pointer);
  
protected:   
  /**
   * Constructor. 
   */
  DSSegmentationFusionImageFilter() 
  { 
    srand(time(NULL)); 
    this->m_HasLabelForUndecidedPixels = false; 
    this->m_ReliabilityMode = 0; 
    this->m_CombinationMode = 1; 
    this->m_LocalRegionRadius = 5; 
    this->m_Gain = 1.0; 
    this->m_PlausibilityThreshold = 0.0; 
  }
  /**
   * Destructor. 
   */
  virtual ~DSSegmentationFusionImageFilter() {}  
  /**
   * Compute the max value in the input images. 
   */
  virtual InputPixelType ComputeMaximumInputValue(); 
  /**
   * House-keeping before going into the threads. 
   */
  virtual void BeforeThreadedGenerateData(); 
  /**
   * Override to add randomness. 
   */
  virtual void ThreadedGenerateData(const OutputImageRegionType &outputRegionForThread, int itkNotUsed); 
  /**
   * Get the relaibility from similarity measure. 
   */
  virtual double GetReliabiltiyFromSimilarityMeasure(int mode, HistogramImageToImageMetricType* metric); 
  /**
   * Calculate the expected segmetnation given the reliability. 
   */
  virtual void CalculateExpectedSegmentation(const OutputImageRegionType& outputRegionForThread); 
  /**
   * Calculate the reliability. 
   */
  virtual void CalculateReliability(const OutputImageRegionType& outputRegionForThread); 
  /**
   * Normalise the intensity of given image. 
   */
  virtual void NormaliseImage(typename TInputImage::Pointer inputImage, typename TInputImage::ConstPointer mask); 


  
protected:
  /**
   *  Label for undecided voxels. Use 240 for random label. 
   */
  OutputPixelType m_LabelForUndecidedPixels;
  /**
   * Used label for decided voxels?  
   */
  bool m_HasLabelForUndecidedPixels;
  /**
   * Total number of labels. 
   */
  InputPixelType m_TotalLabelCount;
  /**
   * Mapping between the label and the power set index. 
   * key: label, value: power set index. 
   */
  typename std::map<int, int> m_LabelToPowerSetIndexMap; 
  /**
   * Reverse mapping between the label and the power set index. 
   * key: power set index, value: label. 
   */
  typename std::map<int, int> m_PowerSetIndexToLabelMap; 
  /**
   * The reliability of the input segmentation. 
   * key: input image index, value: BPA. 
   */
  typename std::map<int, double> m_SegmentationReliability; 
  /**
   * The foreground label. 
   */
  InputPixelType m_ForegroundLabel; 
  /**
   * Store the conflicts. 
   */
  typename ConflictImageType::Pointer m_ConflictImage; 
  /**
   * Target image to be segmented. 
   */
  typename TInputImage::Pointer m_TargetImage; 
  /**
   * Registerd atlases. 
   */
  typename std::map<int, typename TInputImage::Pointer> m_RegisteredAtlases; 
  /**
   * Identity transform for the similarity measure. 
   */
  typename TransformType::Pointer m_IdentityTransform; 
  /**
   * Interpolator for the similarity measure. 
   */
  typename std::vector<typename InterpolatorType::Pointer> m_Interpolators; 
  /**
   * Store which method to use to calculate the believe function. 
   * 0: global user-specified ones. 
   * 1: local NCC. 
   * 2: global NCC. 
   * 3: local NMI. 
   * 4: global NMI. 
   * 5: local MSD. 
   * 6: global MSD. 
   */
  int m_ReliabilityMode; 
  /**
   * Mask with voxel without total consensus. 
   */
  typename TInputImage::Pointer m_ConsensusImage; 
  /**
   * The local similarity metrics. 
   */
  typename std::vector<typename HistogramImageToImageMetricType::Pointer> m_Metrics; 
  /**
   * Model of combining belief functions. 
   * 0: Dampster Shafer. 
   * 1: Transferable beilief model. 
   */
  int m_CombinationMode; 
  /**
   * Radius of the region for local similarity measure. 
   */
  unsigned int m_LocalRegionRadius; 
  /**
   * Gain setting in the local similarity measure.  
   */
  double m_Gain; 
  /**
   * Threshold for taking into account the plausibility in the final decision. 
   */
  double m_PlausibilityThreshold;
  /**
   * Debug information. 
   */
  typename BeliefImageType::Pointer m_ForegroundBeliefImage; 
  typename BeliefImageType::Pointer m_ForegroundPlausibilityImage; 
  
  
private:
  DSSegmentationFusionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
}; 
    
  
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDSSegmentationFusionImageFilter.txx"
#endif


#endif



