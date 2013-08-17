/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkSimilarityMeasure_h
#define itkSimilarityMeasure_h

#include <itkUCLBaseTransform.h>
#include "itkImageToImageMetricWithConstraint.h"
#include <itkCovariantVector.h>
#include <itkPoint.h>
#include <itkImageFileWriter.h>
#include <itkSignedMaurerDistanceMapImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkEulerAffineTransform.h>
#include <itkImageMaskSpatialObject.h>

namespace itk
{
/** 
 * \class SimilarityMeasure
 * \brief Abstract base class, implementing TemplateMethod [2] for similarity measures.
 *
 * The simplest use of this class is to extend it and implement the virtual methods:
 *
 * ResetAggregate()
 * 
 * AggregatePair(fixedValue, movingValue)
 * 
 * AggregateTotal()
 *  
 * See itkSSDImageToImageMetric for a simple example. 
 * 
 * itkImageToImageMetricWithConstraint overrides the default GetValue method, and 
 * calls the virtual GetSimilarity method. This class implements the GetSimilarity method,
 * calling the virtual methods shown above.  So derived classes can either 
 * override GetValue, at which point, you are on your own, and can do whatever you like.
 * Or, you can override GetSimilarity, which means you will have access to the constraint
 * mechanism in the itkImageToImageMetricWithConstraint class.
 * 
 * Note that this class is NOT thread safe.
 * 
 * \ingroup RegistrationMetrics
 */
template < typename TFixedImage, typename TMovingImage > 
class ITK_EXPORT SimilarityMeasure : 
    public ImageToImageMetricWithConstraint< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef SimilarityMeasure                                               Self;
  typedef ImageToImageMetricWithConstraint<TFixedImage, TMovingImage >    Superclass;
  typedef SmartPointer<Self>                                              Pointer;
  typedef SmartPointer<const Self>                                        ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(SimilarityMeasure, ImageToImageMetricWithConstraint);

  /** Types transferred from the base class */
  typedef typename Superclass::TransformType                TransformType;
  typedef typename itk::UCLBaseTransform<double, 
                    TFixedImage::ImageDimension,
                    TMovingImage::ImageDimension>           UCLBaseTransformType;  
  typedef typename Superclass::TransformPointer             TransformPointer;
  typedef typename Superclass::TransformParametersType      TransformParametersType;
  typedef typename Superclass::TransformJacobianType        TransformJacobianType;
  typedef typename Superclass::InputPointType               InputPointType;
  typedef typename Superclass::OutputPointType              OutputPointType;
  typedef typename Superclass::MeasureType                  MeasureType;
  typedef typename Superclass::DerivativeType               DerivativeType;
  typedef typename Superclass::FixedImageType               FixedImageType;
  typedef typename FixedImageType::SizeType                 FixedImageSizeType;
  typedef typename Superclass::FixedImageType::PixelType    FixedImagePixelType;
  typedef typename Superclass::MovingImageType              MovingImageType;
  typedef typename MovingImageType::SizeType                MovingImageSizeType;
  typedef typename Superclass::MovingImageType::PixelType   MovingImagePixelType;
  typedef typename Superclass::FixedImageConstPointer       FixedImageConstPointer;
  typedef typename Superclass::MovingImageConstPointer      MovingImageConstPointer;
  typedef ImageFileWriter<TFixedImage>                      ImageFileWriterType;
  typedef typename Superclass::InterpolatorType             InterpolatorType;
  typedef Image<char, TFixedImage::ImageDimension>          MidwayImageType; 
  typedef Image<float, TFixedImage::ImageDimension>         FloatImageType; 
  typedef Image<unsigned char, TFixedImage::ImageDimension> UnsignedCharImageType; 
  typedef SignedMaurerDistanceMapImageFilter<UnsignedCharImageType, FloatImageType> FixedDistanceMapImageFilterType;
  typedef SignedMaurerDistanceMapImageFilter<UnsignedCharImageType, FloatImageType> MovingDistanceMapImageFilterType;
  typedef LinearInterpolateImageFunction<FloatImageType, double>                    DistanceMapLinearInterpolatorType;
  typedef itk::EulerAffineTransform<double, TFixedImage::ImageDimension, TMovingImage::ImageDimension> AffineTransformType; 
  typedef const ImageMaskSpatialObject<TFixedImage::ImageDimension> FixedMaskType; 
  typedef const ImageMaskSpatialObject<TMovingImage::ImageDimension> MovingMaskType; 
  
  static const int SYMMETRIC_METRIC_AVERAGE; 
  static const int SYMMETRIC_METRIC_MID_WAY; 
  static const int SYMMETRIC_METRIC_BOTH_FIXED_AND_MOVING_TRANSFORM; 
  
  /** Initializes the metric. This is declared virtual in base class. */
  void Initialize() throw (ExceptionObject);

  /** Called from within Initialize */
  void InitializeIntensityBounds() throw (ExceptionObject);
  
  /** Set fixed and moving, min and max intensity values.  */
  void SetIntensityBounds( const FixedImagePixelType fixedLower, 
                           const FixedImagePixelType fixedUpper,
                           const MovingImagePixelType movingLower,
                           const MovingImagePixelType movingUpper);

  /** Get FixedLowerBound, lowest intensity value to use in fixed image. */
  itkGetConstMacro( FixedLowerBound, FixedImagePixelType );
  
  /** Get FixedUpperBound, highest intensity value to use in fixed image. */
  itkGetConstMacro( FixedUpperBound, FixedImagePixelType );

  /** Get MovingLowerBound, lowest intensity value to use in moving image. */
  itkGetConstMacro( MovingLowerBound, MovingImagePixelType );

  /** Get MovingUpperBound, highest intensity value to use in moving image. */
  itkGetConstMacro( MovingUpperBound, MovingImagePixelType );

  /**
   * Get the number of samples used in the most recent evaluation of the measure.
   */
  itkGetMacro( NumberOfFixedSamples, long int);

  /**
   * Get the number of samples used in the most recent evaluation of the measure.
   */
  itkGetMacro( NumberOfMovingSamples, long int);

  /**
   * Set whether we are using a Two Sided Metric.
   * 
   * There are three ways to handle the moving image mask.
   * 
   * If TwoSidedMetric is true, and we have supplied a moving
   * mask, we actually evaluate the metric twice, once using 
   * the baseline mask only and mapping points into the moving image, 
   * and once using the moving mask only mapping the points into
   * the fixed image. We then take the mean of the two measures.
   * This means the transformation must be invertable.
   * The reason for this is that you don't want to inadvertently
   * align the boundaries of the two masks.
   * 
   * If TwoSidedMetric is false, we evaluate the measure
   * once, and take all the points in the baseline masked
   * region, mapping them to the moving image, and only use
   * them if they fall within the moving masked region.
   * This means we are using the intersection of the two regions
   * as the region of interest. This means you MAY cause your
   * algorithm to be a bit prone to aligning the edges of the masks.
   * 
   * The alternative is to simply not supply the moving mask.
   * In this case, the value of this flag will not matter, as there is
   * no moving mask to worry about.
   * 
   * The default value is false.
   * 
   * This flag is used in the GetSimilarity method in this class.
   * So if otherclasses override GetSimilarity (as they are free to do so),
   * then this flag will probably have very little meaning.
   * 
   */
  itkSetMacro( TwoSidedMetric, bool );
  itkGetMacro( TwoSidedMetric, bool );
  
  /** Writes a fixed image after each similarity measure evaluation. */
  itkSetMacro( WriteFixedImage, bool );
  itkGetMacro( WriteFixedImage, bool );

  /** Set/Get the file name to dump to. Default to "tmp.similarity.fixed" */
  itkSetMacro( FixedImageFileName, std::string );
  itkGetMacro( FixedImageFileName, std::string );

  /** Set/Get the file ext to dump to. Default to "nii". */
  itkSetMacro( FixedImageFileExt, std::string );
  itkGetMacro( FixedImageFileExt, std::string );

  /** Writes a transformed moving image after each similarity measure evaluation. */
  itkSetMacro( WriteTransformedMovingImage, bool );
  itkGetMacro( WriteTransformedMovingImage, bool );

  /** Set/Get the file name to dump to. Default to "tmp.similarity.moving" */
  itkSetMacro( TransformedMovingImageFileName, std::string );
  itkGetMacro( TransformedMovingImageFileName, std::string );

  /** Set/Get the file ext to dump to. Default to "nii". */
  itkSetMacro( TransformedMovingImageFileExt, std::string );
  itkGetMacro( TransformedMovingImageFileExt, std::string );
  
  /** 
   * Set/Get this flag to do direct voxel comparison.
   * If this is true, we just iterate through the image, comparing voxel
   * for voxel. There is minimal checking, so we assume the size is identical.
   * This is useful for block matching.
   */
  itkSetMacro(DirectVoxelComparison, bool);
  itkGetMacro(DirectVoxelComparison, bool);
  
  /**
   * Set/Get this flag to return the symmetrical similarity measure by evaluating the similarity both ways. 
   * 1. FixedImage and TransformedMovingImage: transforming the moving image into the 
   *    space of fixed image. 
   * 2. TransformedFixedImage and MovingImage: transforming the fixeed image into the 
   *    space of moving image. 
   */
  itkSetMacro(SymmetricMetric, int);
  itkGetMacro(SymmetricMetric, int);
  
  itkSetMacro(IsUpdateMatrix, bool); 
  itkGetMacro(IsUpdateMatrix, bool); 
  
  /** Set/Get the transformed moving image pad value. Default 0. */
  itkSetMacro(TransformedMovingImagePadValue, MovingImagePixelType); 
  itkGetMacro(TransformedMovingImagePadValue, MovingImagePixelType); 
  
  /** Set/Get the UseWeighting. Default false. */
  itkSetMacro(UseWeighting, bool); 
  itkGetMacro(UseWeighting, bool); 
  
  /** Set/Get the WeightingDistanceThreshold. Default 2.0mm. */
  itkSetMacro(WeightingDistanceThreshold, double); 
  itkGetMacro(WeightingDistanceThreshold, double); 
  
  /**
   * Set/Get interpolators. 
   */
  itkSetObjectMacro(FixedImageInterpolator, InterpolatorType); 
  itkSetObjectMacro(MovingImageInterpolator, InterpolatorType); 
  itkGetObjectMacro(FixedImageInterpolator, InterpolatorType); 
  itkGetObjectMacro(MovingImageInterpolator, InterpolatorType); 
  
  /**
   * Set/Get FixedImageTransform. 
   */
  itkSetObjectMacro(FixedImageTransform, TransformType); 
  itkGetObjectMacro(FixedImageTransform, TransformType); 
  
  /**
   * Set/Get m_InitialiseIntensityBoundsUsingMask. 
   */
  itkSetMacro(InitialiseIntensityBoundsUsingMask, bool); 
  itkGetMacro(InitialiseIntensityBoundsUsingMask, bool); 
  
  /**
   * Set/Get IsResampleWholeImage. 
   */
  itkSetMacro(IsResampleWholeImage, bool); 
  itkGetMacro(IsResampleWholeImage, bool); 
  
  /** 
   * Subclasses should implement this.
   * Simply return true if the cost function should be maximized (like Mutual Info.)
   * or false if it should be minimized (like Sum Squared Difference).
   */ 
  virtual bool ShouldBeMaximized() { return false; }
  
  /**
   * Return the transformed moving image. 
   */
  virtual const TFixedImage* GetTransformedMovingImage() const { return this->m_TransformedMovingImage.GetPointer(); }

  /**
   * Return the transformed moving image. 
   */
  virtual const TFixedImage* GetTransformedFixedImage() const { return this->m_TransformedFixedImage.GetPointer(); }
  
  /** 
   * Get a measure of the change of parameters between two successive iterations. 
   */
  double GetMeasureOfParameterChange(TransformParametersType lastP, TransformParametersType p); 
  
protected:
  
  SimilarityMeasure();
  virtual ~SimilarityMeasure() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /**
   * itkImageToImageMetric implements GetValue, which calls this GetSimilarity,
   * which calls ResetAggregate, AggregatePair, AggregateTotal, which subclasses should override.
   */
  virtual MeasureType GetSimilarity( const TransformParametersType & parameters ) const;

  /**
   * Called at the start of each evaluation of the cost function.
   * Subclasses should implement this, and reset any internal variables.
   */
  virtual void ResetCostFunction() = 0;
  
  /** 
   * Use this method to add corresponding pairs of image values,
   * called repeatedly during a single value of the cost function.
   */
  virtual void AggregateCostFunctionPair(FixedImagePixelType fixedValue, MovingImagePixelType movingValue) = 0;
  
  /** 
   * Use this method to add corresponding pairs of image values,
   * called repeatedly during a single value of the cost function. With weighting. 
   */
  virtual void AggregateCostFunctionPairWithWeighting(FixedImagePixelType fixedValue, MovingImagePixelType movingValue, double weight) {       itkExceptionMacro(<<"AggregateCostFunctionPairWithWeighting not implemented.");
 }
  
  /**
   * Assuming we have all the data aggregated by AggregatePair method, 
   * this method is used to sum up, or average, or do something to get the final total.
   */
  virtual MeasureType FinalizeCostFunction() = 0;

  /**
   * As we iterate through image, its easy to calculate a
   * transformed moving image as we go. This is essential for 
   * the deformable type metrics, and optional for simple
   * metrics like SSD, NCC. However, we store this here, so
   * everyone can have one. 
   */
  typename TFixedImage::Pointer m_TransformedMovingImage;
  
  /**
   * Also keep a transformed fixed image for symmetric deformable transform. 
   */
  typename TFixedImage::Pointer m_TransformedFixedImage;
  
  /**
   * For symmetric registration, we need to interpolate fixed image and moveing image. 
   */
  /**
   * Fixed image interpolator. 
   */
  typename InterpolatorType::Pointer m_FixedImageInterpolator; 
  /**
   * Moving image interpolator. 
   */
  typename InterpolatorType::Pointer m_MovingImageInterpolator; 

  /** Lowest intensity value to use in fixed image. */
  FixedImagePixelType m_FixedLowerBound;
  
  /** Highest intensity value to use in fixed image. */
  FixedImagePixelType m_FixedUpperBound;
  
  /** Lowest intensity value to use in moving image. */
  MovingImagePixelType m_MovingLowerBound;
  
  /** Highest intensity value to use in moving image. */
  MovingImagePixelType m_MovingUpperBound;
  
  /**
   * If set, use weightings from the distance maps. 
   */
  bool m_UseWeighting; 
  
  /**
   * The weighting distance threshold - weighting set to 1 if the distance is greater than this threshold. 
   * Otherwise, the weighting is linearly set. 
   */
  double m_WeightingDistanceThreshold; 
  
  /**
   * This image defines the space for resampling the fixed and moving images at the mid-point. 
   */
  typename MidwayImageType::Pointer m_MidwayImage; 
  
  /**
   * Distance map of the fixed mask. Provides weighting for de-weighting the dependency of voxels
   * near the edges of the overlapping region. See Jenkinson (2001). 
   * Improved Optimization for the robust and accurate linear registration and motion
   * correction of brain image. 
   */
  typename FixedDistanceMapImageFilterType::Pointer m_FixedDistanceMap; 
  
  /**
   * Distance map of the moving mask. 
   */
  typename MovingDistanceMapImageFilterType::Pointer m_MovingDistanceMap; 
  
  /**
   * Interpolator for the distance map in the fixed image. 
   */
  typename DistanceMapLinearInterpolatorType::Pointer m_FixedDistanceMapInterpolator; 
  
  /**
   * Interpolator for the distance map in the moving image. 
   */
  typename DistanceMapLinearInterpolatorType::Pointer m_MovingDistanceMapInterpolator; 
  
  /**
   * Transform for the fixed image. 
   */
  typename TransformType::Pointer m_FixedImageTransform; 
  
  /**
   * Use the masks for the intensity bounds initialisation. 
   */
  bool m_InitialiseIntensityBoundsUsingMask; 
  
  /**
   * Resample the whole image in GetSimilarityUsingFixedAndMovingImageTransforms. 
   */
  bool m_IsResampleWholeImage; 

  /** Method that actually writes one of the fixed or transformed moving images to file. */
  void WriteImage(const TFixedImage* image, std::string filename) const;
  
  /** 
   * Return the symmetrical similarity measure by evaluating the similarity both ways. 
   * 1. FixedImage and TransformedMovingImage: transforming the moving image into the 
   *    space of fixed image. 
   * 2. TransformedFixedImage and MovingImage: transforming the fixeed image into the 
   *    space of moving image. 
   */
  virtual MeasureType GetSymmetricSimilarity(const TransformParametersType & parameters); 
  
  /**
   * Return the symmetrical similarity measure by evaluating the similarity at the mid-point. 
   * 
   * We are directly looking for the midway transformation, i.e. we evaluate
   *   similiary(A(T^-1) + B(T)). 
   * 
   */
  virtual MeasureType GetSymmetricSimilarityAtHalfway(const TransformParametersType & parameters); 
  
  /**
   * Get the similarity given both the fixed and moving image transform. 
   */
  virtual MeasureType GetSimilarityUsingFixedAndMovingImageTransforms(const TransformParametersType& parameters); 
  
  /** 
   * Initializes the symmetric metric. 
  */
  virtual void InitializeSymmetricMetric();
  
  /** 
   * Initializes the distance weightings. 
  */
  virtual void InitializeDistanceWeightings();

private:
  
  SimilarityMeasure(const Self&); // purposefully not implemented
  void operator=(const Self&);    // purposefully not implemented

  /** 
   * Boolean flag to indicate whether the user supplied the bounds or
   * whether they should be computed from the min and max of image intensities
   */
  bool m_BoundsSetByUser;

  /** 
   * For counting the number of samples in the
   * most recent evaluation. Must be mutable, as
   * the GetValue method is declared const.
   */
  mutable long int m_NumberOfFixedSamples; 

  /** 
   * For counting the number of samples in the
   * most recent evaluation. Must be mutable, as
   * the GetValue method is declared const.
   */
  mutable long int m_NumberOfMovingSamples;

  /** So we get a new file to dump to each time. */
  mutable int m_IterationNumber;
  
  /** Dumps fixed image. */
  bool m_WriteFixedImage;
  
  /** Dumps transformed moving image. */
  bool m_WriteTransformedMovingImage;

  /** File name to dump fixed image to. */
  std::string m_FixedImageFileName;

  /** File name to dump fixed image to. */
  std::string m_FixedImageFileExt;

  /** File name to dump transformed moving image to. */
  std::string m_TransformedMovingImageFileName;

  /** File name to dump transformed moving image to. */
  std::string m_TransformedMovingImageFileExt;

  /** TwoSidedMetric flag. */ 
  bool m_TwoSidedMetric;

  /** 
   * If we have no transform, and the images are the same size, we switch 
   * to a voxel by voxel comparison. Useful for block matching.
   * Default false.
   */
  bool m_DirectVoxelComparison;
  
  /** Symmetric similarity - 
   * if set to 1, calls GetSymmetricSimilarity 
   * if set to 2, calls GetSymmetricSimilarityAtHalfway 
   */
  int m_SymmetricMetric; 
  
  bool m_IsUpdateMatrix; 
  
  /** So we can specify the transformed image pad value. Default 0. */
  MovingImagePixelType m_TransformedMovingImagePadValue;
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSimilarityMeasure.txx"
#endif

#endif



