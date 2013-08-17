/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkBlockMatchingMethod_h
#define itkBlockMatchingMethod_h


#include "itkMaskedImageRegistrationMethod.h"
#include <itkResampleImageFilter.h>
#include <itkArray.h>
#include <itkPointSet.h>
#include <itkRegionOfInterestImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkSimilarityMeasure.h>
#include <itkDefaultStaticMeshTraits.h>
#include <itkPointSetToPointSetSingleValuedMetric.h>
#include <itkCovarianceCalculator.h>
#include <itkScalarImageToListAdaptor.h>
#include <itkGradientMagnitudeImageFilter.h>
#include <itkMinimumMaximumImageCalculator.h>

namespace itk
{
/** 
 * \class BlockMatchingMethod
 * \brief Initial implementation of Seb Ourselin's block matching algorithm. 
 *
 * This class implements the main algorithm of Ourselin et. al. Image
 * and Vision Computing 19 (2000) 25-31. It's a subclass of the 
 * itkSingleResolutionImageRegistrationMethod so that we can easily plug
 * it into our usual multi-resolution stuff in itkMultiResolutionImageRegistrationWrapper,
 * and hence put it into itkImageRegistrationFilter.
 * 
 * This implementation uses the same components as the other methods in
 * the RegistrationToolbox, but it doesn't use the same plumbing at all.
 * This class overrides a lot of the base class functionality, so you are
 * advised to look at the implementation.
 * 
 * The algorithm proceeds by:
 * 
 * 1. Take initial transform parameters, and resample the moving image.
 * 2. Subdivide the fixed image into blocks, and find the corresponding
 * block in the resampled moving image that is most similar.
 * 3. This results in a list of pairs of point correspondences between the
 * centre of the fixed image block, and the centre of the moving image block.
 * 4. Take these point correspondences, and perform a trimmed least squares fit,
 * optimising your choice of transformation. 
 * 5. Repeat from 1.
 * 
 * For further details read Ourselin et. al. Image and Vision Computing 19 (2000) 25-31. 
 * 
 * \sa MultiResolutionImageRegistrationWrapper
 * \sa ImageRegistrationFilter
 */
template <typename TImageType, class TScalarType>
class ITK_EXPORT BlockMatchingMethod 
: public MaskedImageRegistrationMethod<TImageType> 
{
public:

  /** Standard class typedefs. */
  typedef BlockMatchingMethod                                           Self;
  typedef MaskedImageRegistrationMethod<TImageType>                     Superclass;
  typedef SmartPointer<Self>                                            Pointer;
  typedef SmartPointer<const Self>                                      ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(BlockMatchingMethod, MaskedImageRegistrationMethod);

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, TImageType::ImageDimension);

  /** Typdef's anonymous. */
  typedef TImageType                                                    ImageType;
  typedef typename ImageType::SizeType                                  ImageSizeType;
  typedef typename ImageType::PixelType                                 ImagePixelType;
  typedef typename ImageType::SpacingType                               ImageSpacingType;
  typedef typename ImageType::RegionType                                ImageRegionType;
  typedef typename ImageType::IndexType                                 ImageIndexType;
  typedef ResampleImageFilter<TImageType, TImageType >                  ResampleFilterType;
  typedef typename ResampleFilterType::Pointer                          ResampleFilterPointer;
  typedef RegionOfInterestImageFilter<TImageType, TImageType>           RegionOfInterestFilterType;
  typedef typename RegionOfInterestFilterType::Pointer                  RegionOfInterestFilterPointer;
  
  // We either calculate the variance, based on the image, which will be whatever scalar type is passed in as template parameter
  typedef itk::Statistics::ScalarImageToListAdaptor< ImageType >                    ImageTypeListAdaptorType;
  typedef typename ImageTypeListAdaptorType::Pointer                                ImageTypeListAdaptorPointer;
  typedef itk::Statistics::CovarianceCalculator< ImageTypeListAdaptorType >         ImageTypeCovarianceCalculatorType;
  typedef typename ImageTypeCovarianceCalculatorType::Pointer                       ImageTypeCovarianceCalculatorPointer;

  // Or we take the gradient magnitude image, so we should do it as a float.
  typedef Image<float, TImageType::ImageDimension>                                  GradientImageType;
  typedef GradientMagnitudeImageFilter<TImageType, GradientImageType>               GradientMagnitudeFilterType;
  typedef typename GradientMagnitudeFilterType::Pointer                             GradientMagnitudeFilterPointer;
  typedef itk::Statistics::ScalarImageToListAdaptor< GradientImageType >            GradientImageTypeListAdaptorType;
  typedef typename GradientImageTypeListAdaptorType::Pointer                        GradientImageTypeListAdaptorPointer;
  typedef itk::Statistics::CovarianceCalculator< GradientImageTypeListAdaptorType > GradientImageTypeCovarianceCalculatorType;
  typedef typename GradientImageTypeCovarianceCalculatorType::Pointer               GradientImageTypeCovarianceCalculatorPointer;
  
  typedef SimilarityMeasure<TImageType, TImageType >                    SimilarityMeasureType;
  typedef SimilarityMeasureType*                                        SimilarityMeasurePointer;
  typedef NearestNeighborInterpolateImageFunction< TImageType, 
                                                   TScalarType>         DummyInterpolatorType;
  typedef typename DummyInterpolatorType::Pointer                       DummyInterpolatorPointer;
  typedef typename Superclass::ParametersType                           ParametersType;
  typedef PointSet< TScalarType, 
                    TImageType::ImageDimension,
                    DefaultStaticMeshTraits
                      < 
                        TScalarType, 
                        TImageType::ImageDimension, 
                        TImageType::ImageDimension,
                        TScalarType,
                        TScalarType
                      > 
                    >                                                   PointSetType;
  typedef typename PointSetType::PointType                              PointType;
  typedef typename PointSetType::PointsContainer::ConstIterator         PointIterator;
  typedef typename PointSetType::Pointer                                PointSetPointer;
  typedef typename PointSetType::PointsContainer                        PointsContainerType;
  typedef typename PointsContainerType::Pointer                         PointsContainerPointer;
  typedef PointSetToPointSetSingleValuedMetric<PointSetType, 
                                               PointSetType>            PointSetMetricType;
  typedef typename PointSetMetricType::Pointer                          PointSetMetricPointer;
  typedef typename Superclass::TransformType                            TransformType;
  typedef TransformType*                                                TransformPointer;
  typedef MinimumMaximumImageCalculator<TImageType>                     MinimumMaximumImageCalculatorType;
  typedef typename MinimumMaximumImageCalculatorType::Pointer           MinimumMaximumImageCalculatorPointer;
  
  /** Set/Get the point based metric. */                                              
  itkSetObjectMacro(PointSetMetric, PointSetMetricType);
  itkGetObjectMacro(PointSetMetric, PointSetMetricType);
                                                 
  /** Max iterations round main 1. ) block match, 2.) optimisation loop. Default 10. */
  itkSetMacro(MaximumNumberOfIterationsRoundMainLoop, unsigned int);
  itkGetMacro(MaximumNumberOfIterationsRoundMainLoop, unsigned int);
  
  /** 
   * Set all the block parameters in one go. Otherwise, they 
   * default to -1, and this class works them out according to
   * section 2.3 in Ourselin et. al. Image and Vision Computing 
   * 19 (2000) 25-31.
   * 
   * blockSize is big N in the IVC paper.
   * blockHalfWidth is big Omega in the IVC paper.
   * blockSpacing is big Delta_1 in the IVC paper.
   * blockSubSampling is big Delta_2 in the IVC paper.
   * 
   * You should set these values in millimetres,
   * and then internally, we set up different values for
   * each image axis depending on voxel size.
   */
  void SetBlockParameters(
    double blockSize,
    double blockHalfWidth,
    double blockSpacing,
    double blockSubSampling);
    
  /** 
   * Get the block size, this is big N in the IVC paper.
   */
  itkGetMacro(BlockSize, double);

  /** 
   * Get the block half width, this is big Omega in the IVC paper.
   */
  itkGetMacro(BlockHalfWidth, double);

  /** 
   * Get the spacing between blocks, this is big Delta_1 in the IVC paper.
   */
  itkGetMacro(BlockSpacing, double);

  /** 
   * Get the sub-sampling for moving blocks, this is big Delta_2 in the IVC paper.
   */
  itkGetMacro(BlockSubSampling, double);

  /** For writing out the transformed moving image. */
  itkSetMacro(TransformedMovingImageFileName, std::string);
  itkGetMacro(TransformedMovingImageFileName, std::string);

  /** For writing out the transformed moving image. */
  itkSetMacro(TransformedMovingImageFileExt, std::string);
  itkGetMacro(TransformedMovingImageFileExt, std::string);
  
  /** For writing out the transformed moving image. */
  itkSetMacro(WriteTransformedMovingImage, bool);
  itkGetMacro(WriteTransformedMovingImage, bool);

  /** 
   * Set/Get Epsilon to control the termination criteria. 
   * See Section 2.3. in IVC 2000 paper. Defaults to 1. 
   */
  itkSetMacro(Epsilon, double);
  itkGetMacro(Epsilon, double);

  /** 
   * According to paper, once we have solved at one level, we half the parameters and continue.
   * This parameter gives us the option to reduce it more quickly or more slowly. Default 0.5 
   */
  itkSetMacro(ParameterReductionFactor, double);
  itkGetMacro(ParameterReductionFactor, double);
  
  /** Set the minimum block size (Big N in IVC 2000 paper). Default 4.0 mm. */
  itkSetMacro(MinimumBlockSize, double);
  itkGetMacro(MinimumBlockSize, double);
  
  /** 
   * Set the percentage of fixed image points to used, based on variance. 
   * In Ourselin et. al. MICCAI 2002, the multi-resolution approach describes
   * how you use all points at lower resolution, and halve the number of relevant
   * voxels at each resolution, with a lower bound of 20%. So, this percentage
   * should be set externally, as this class just applies it blindly.
   * Defaults to 100%.
   */
  itkSetMacro(PercentageOfPointsToKeep, int);
  itkGetMacro(PercentageOfPointsToKeep, int);

  /**
   * Set the percentage of points to keep in least trimmed squares.
   * As with PercentageOfPointsToKeep, this is applied blindly,
   * so this class knows nothing of how you change it for different
   * multi-resolution levels.
   * Defaults to 100%
   */
  itkSetMacro(PercentageOfPointsInLeastTrimmedSquares, int);
  itkGetMacro(PercentageOfPointsInLeastTrimmedSquares, int);
  
  /** 
   * And when we are calculating the variance, if this is true, we use the gradient magnitude image
   * and if this is false, we use the normal intensity image. 
   */
  itkSetMacro(UseGradientMagnitudeVariance, bool);
  itkGetMacro(UseGradientMagnitudeVariance, bool);

  /** Scale block parameters from voxels to millimetres. */
  itkSetMacro(ScaleByMillimetres, bool);
  itkGetMacro(ScaleByMillimetres, bool);
  
  /** Set the file name. The file extension .vtk is added automatically. */
  itkSetMacro(PointSetFileNameWithoutExtension, std::string);
  itkGetMacro(PointSetFileNameWithoutExtension, std::string);
  
  /** To determine if we write the point set. */
  itkSetMacro(WritePointSet, bool);
  itkGetMacro(WritePointSet, bool);
  
  /** To determine if we include pairs with zero displacement. */
  itkSetMacro(NoZero, bool);
  itkGetMacro(NoZero, bool);
  
  /** Set the transformed moving image pad value. Default 0. */  
  itkSetMacro(TransformedMovingImagePadValue, ImagePixelType);
  itkGetMacro(TransformedMovingImagePadValue, ImagePixelType);
  
protected:

  BlockMatchingMethod();
  virtual ~BlockMatchingMethod() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** We override this method to wire everything together. */
  virtual void Initialize() throw (ExceptionObject);

  /** We override this method to actually do the registration. */
  virtual void DoRegistration() throw (ExceptionObject);

  /** Writes the point set to file. */
  virtual void WritePointSet(const PointsContainerPointer& fixedPointContainer, 
      const PointsContainerPointer& movingPointContainer);
  
  /** Unfortunately, need a 2D version. */
  virtual void GetPointCorrespondencies2D(
    ImageSizeType& size,
    ImageSizeType& bigN,
    ImageSizeType& bigOmega,
    ImageSizeType& bigDeltaOne,
    ImageSizeType& bigDeltaTwo,
    PointsContainerPointer& fixedPointContainer,
    PointsContainerPointer& movingPointContainer
    );
  
  /** Unfortunately, need a 3D version aswell. */      
  virtual void GetPointCorrespondencies3D(
    ImageSizeType& size,
    ImageSizeType& bigN,
    ImageSizeType& bigOmega,
    ImageSizeType& bigDeltaOne,
    ImageSizeType& bigDeltaTwo,
    PointsContainerPointer& fixedPointContainer,
    PointsContainerPointer& movingPointContainer
    );

  /** Used to compare the previousParameters to currentParameters and decide if we should keep iterating, or change scale */
  virtual bool CheckEpsilon(ParametersType& previousParameters, ParametersType& currentParameters);
  
  /** Multiplies the given point by the two sets of transforms, and returns the difference between the results. */
  virtual double CheckSinglePoint(ParametersType& previousParameters, ParametersType& currentParameters, ImageIndexType& index);
  
  /** Method to trim points. */
  virtual void TrimPoints(const TransformType* transform, 
      const PointsContainerType* fixedPoints,
      const PointsContainerType* movingPoints,
      PointsContainerType* trimmedFixedPoints,
      PointsContainerType* trimmedMovingPoints);
  
  /**
   * \class VarianceHeapDataType
   * \brief So we can have an ordered list of indexes, based on variance of a block.
   */
  class VarianceHeapDataType 
    {
      public:
        VarianceHeapDataType(const TScalarType& aKey, const ImageIndexType& anIndex)
          {
            this->key = aKey;
            this->index = anIndex;
          }
          
      void operator=(const VarianceHeapDataType& another)
        {
          this->key = another.key;
          this->index = another.index;
        }
        
      friend bool operator<(
        const VarianceHeapDataType& x, const VarianceHeapDataType& y) 
          {
            if(x.GetKey() < y.GetKey())
              return true;
            else
              return false;
          }

      void SetKey(const TScalarType& aKey) { this->key = aKey; }
      TScalarType GetKey() const { return this->key; }
      
      void SetIndex(const ImageIndexType& anIndex) { this->index = anIndex; }
      ImageIndexType GetIndex() const { return this->index; }
      
      private:
        TScalarType key;
        ImageIndexType index;  
    };
   
  typedef std::priority_queue<VarianceHeapDataType> VarianceHeap;
  
  /**
   * \class  ResidualHeapDataType
   * \brief So we can have an ordered list of points, based on the residual between transformed fixed and moving.
   */
  class ResidualHeapDataType
    {

      public:
        ResidualHeapDataType(const TScalarType& aResidual, const PointType& aFixedPoint, const PointType& aMovingPoint)
          {
            this->residual = aResidual;
            this->fixed = aFixedPoint;
            this->moving = aMovingPoint;
          }
        
      void operator=(const ResidualHeapDataType& another)
        {
          this->residual = another.residual;
          this->fixed = another.fixed;
          this->moving = another.moving;
        }
      
      friend bool operator<(
        const ResidualHeapDataType& x, const ResidualHeapDataType& y) 
          {
            if(y.GetResidual() < x.GetResidual())
              return true;
            else
              return false;
          }

      void SetResidual(const TScalarType& aResidual) { this->residual = aResidual; }
      TScalarType GetResidual() const { return this->residual; }
    
      void SetFixed(const PointType& aFixedPoint) { this->fixed = aFixedPoint; }
      PointType GetFixed() const { return this->fixed; }

      void SetMoving(const PointType& aMovingPoint) { this->moving = aMovingPoint; }
      PointType GetMoving() const { return this->moving; }

      private:
        TScalarType residual;
        PointType fixed;
        PointType moving;
    };
  
  typedef std::priority_queue<ResidualHeapDataType> ResidualHeap;
  
private:
  
  BlockMatchingMethod(const Self&); // purposely not implemented
  void operator=(const Self&);      // purposely not implemented
  
  ResampleFilterPointer                        m_MovingImageResampler;
  
  RegionOfInterestFilterPointer                m_FixedImageRegionFilter;
  
  ImageTypeListAdaptorPointer                  m_FixedImageListAdaptor;
  ImageTypeCovarianceCalculatorPointer         m_FixedImageCovarianceCalculator;

  GradientMagnitudeFilterPointer               m_GradientMagnitudeImageFilter;
  GradientImageTypeListAdaptorPointer          m_GradientMagnitudeListAdaptor;
  GradientImageTypeCovarianceCalculatorPointer m_GradientMagnitudeCovarianceCalculator;
  
  DummyInterpolatorPointer                     m_DummyInterpolator;
 
  PointSetPointer                              m_FixedPointSet;
  PointsContainerPointer                       m_FixedPointSetContainer;
  
  PointSetPointer                              m_MovingPointSet;
  PointsContainerPointer                       m_MovingPointSetContainer;
  
  PointSetMetricPointer                        m_PointSetMetric;
  
  unsigned int                                 m_MaximumNumberOfIterationsRoundMainLoop;
  
  double                                       m_BlockSize;
  
  double                                       m_BlockHalfWidth;
  
  double                                       m_BlockSpacing;
  
  double                                       m_BlockSubSampling;
  
  double                                       m_Epsilon;
  
  double                                       m_ParameterReductionFactor;
  
  double                                       m_MinimumBlockSize;
  
  int                                          m_PercentageOfPointsToKeep;
  
  int                                          m_PercentageOfPointsInLeastTrimmedSquares;
  
  std::string                                  m_TransformedMovingImageFileName;
  
  std::string                                  m_TransformedMovingImageFileExt;
  
  bool                                         m_WriteTransformedMovingImage;

  std::string                                  m_PointSetFileNameWithoutExtension;
  
  bool                                         m_WritePointSet;
  
  bool                                         m_ScaleByMillimetres;
  
  bool                                         m_UseGradientMagnitudeVariance;
  
  bool                                         m_NoZero;
  
  ImagePixelType                               m_TransformedMovingImagePadValue;
  
  MinimumMaximumImageCalculatorPointer         m_MinMaxCalculator;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBlockMatchingMethod.txx"
#endif

#endif



