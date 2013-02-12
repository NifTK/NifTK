/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkRelaxStreamlinesFilter_h
#define __itkRelaxStreamlinesFilter_h

#include "itkImage.h"
#include "itkVector.h"
#include "itkPoint.h"
#include "itkBaseCTEStreamlinesFilter.h"
#include "itkVectorInterpolateImageFunction.h"
#include "itkInterpolateImageFunction.h"


namespace itk {
/** 
 * \class RelaxStreamlinesFilter
 * \brief Calculates length between two boundaries, solving PDE by iterative relaxation.
 * 
 * This filter implements algorithm 1) in Yezzi and Prince 2003 , IEEE TMI
 * Vol. 22, No. 10, p 1332-1339. The first input should be a scalar image,
 * such as the output of itkLaplacianSolverImageFilter. The second image 
 * should be the vector field of the gradient of the first input.
 * 
 * In this implementation, you specify the the voltage potentials that
 * your Laplacian was solved on. This enables the filter to set the 
 * boundaries correctly. Only voxels that are > LowVoltage and
 * < HighVoltage are solved.
 * 
 * \sa BaseStreamlinesFilter
 * \sa IntegrateStreamlinesFilter
 * \sa OrderedTraversalStreamlinesFilter
 */
template < class TImageType, typename TScalarType, unsigned int NDimensions> 
class ITK_EXPORT RelaxStreamlinesFilter :
  public BaseCTEStreamlinesFilter<TImageType, TScalarType, NDimensions>
{
public:

  /** Standard "Self" typedef. */
  typedef RelaxStreamlinesFilter                                          Self;
  typedef BaseCTEStreamlinesFilter<TImageType, TScalarType, NDimensions>  Superclass;
  typedef SmartPointer<Self>                                              Pointer;
  typedef SmartPointer<const Self>                                        ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RelaxStreamlinesFilter, BaseStreamlinesFilter);

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, NDimensions);

  /** Standard typedefs. */
  typedef Vector< TScalarType, NDimensions >                     InputVectorImagePixelType;
  typedef Image< InputVectorImagePixelType, NDimensions >        InputVectorImageType;
  typedef typename InputVectorImageType::Pointer                 InputVectorImagePointer;
  typedef typename InputVectorImageType::ConstPointer            InputVectorImageConstPointer;
  typedef typename InputVectorImageType::IndexType               InputVectorImageIndexType; 
  typedef TScalarType                                            InputScalarImagePixelType;
  typedef Image< InputScalarImagePixelType, NDimensions >        InputScalarImageType;
  typedef Point<TScalarType, NDimensions>                        InputScalarImagePointType;
  typedef typename InputScalarImageType::SpacingType             InputScalarImageSpacingType;
  typedef typename InputScalarImageType::Pointer                 InputScalarImagePointer;
  typedef typename InputScalarImageType::IndexType               InputScalarImageIndexType;
  typedef typename InputScalarImageType::ConstPointer            InputScalarImageConstPointer;
  typedef typename InputScalarImageType::RegionType              InputScalarImageRegionType;
  typedef InputScalarImageType                                   OutputImageType;
  typedef typename OutputImageType::PixelType                    OutputImagePixelType;
  typedef typename OutputImageType::Pointer                      OutputImagePointer;
  typedef typename OutputImageType::ConstPointer                 OutputImageConstPointer;
  typedef typename OutputImageType::IndexType                    OutputImageIndexType;
  typedef typename OutputImageType::SpacingType                  OutputImageSpacingType;
  typedef typename OutputImageType::RegionType                   OutputImageRegionType;
  typedef typename OutputImageType::SizeType                     OutputImageSizeType;
  typedef VectorInterpolateImageFunction<InputVectorImageType
                                         ,TScalarType
                                        >                        VectorInterpolatorType;
  typedef typename VectorInterpolatorType::Pointer               VectorInterpolatorPointer;
  typedef typename VectorInterpolatorType::PointType             VectorInterpolatorPointType;
  typedef InterpolateImageFunction< InputScalarImageType
                                    ,TScalarType >               ScalarInterpolatorType;
  typedef typename ScalarInterpolatorType::Pointer               ScalarInterpolatorPointer;
  typedef typename ScalarInterpolatorType::PointType             ScalarInterpolatorPointType;
                                        
  /** Sets the scalar (Laplacian) image, at input 0. */
  void SetScalarImage(const InputScalarImageType *image) {this->SetNthInput(0, const_cast<InputScalarImageType *>(image)); }

  /** Sets the vector image, at input 1. */
  void SetVectorImage(const InputVectorImageType* image) { this->SetNthInput(1, const_cast<InputVectorImageType *>(image)); }

  /** Sets the segmented/label image, at input 2. */
  void SetSegmentedImage(const InputScalarImageType *image) {this->SetNthInput(2, const_cast<InputScalarImageType *>(image)); }

  /** Set convergence threshold. Default 10E-5. */
  itkSetMacro(EpsilonConvergenceThreshold, double);
  itkGetMacro(EpsilonConvergenceThreshold, double);

  /** Set MaximumNumberOfIterations threshold. Default 200. */
  itkSetMacro(MaximumNumberOfIterations, unsigned long int );
  itkGetMacro(MaximumNumberOfIterations, unsigned long int );

  /** Set Maximum length. Default 10. */
  itkSetMacro(MaximumLength, double );
  itkGetMacro(MaximumLength, double );

  /** Initialize boundaries or not. */
  itkSetMacro(InitializeBoundaries, bool);
  itkGetMacro(InitializeBoundaries, bool);
  
  OutputImageType* GetL0Image() const { return m_L0Image.GetPointer(); }
  OutputImageType* GetL1Image() const { return m_L1Image.GetPointer(); }

protected:
  RelaxStreamlinesFilter();
  ~RelaxStreamlinesFilter();
  void PrintSelf(std::ostream& os, Indent indent) const;

  // The main filter method. Note, single threaded.
  virtual void GenerateData();
  
  // We set the whole of L0 and L1 image to zero,
  // and extract the set of GM voxels. We then 
  // call this method to initialize the boundaries.
  // The implementation in this class does nothing.
  // However, derived classes can initialize points in
  // the L0 and L1 image, and also change the size of
  // the listOfGreyMatterPixels vector.
  virtual void InitializeBoundaries(
    std::vector<InputScalarImageIndexType>& completeListOfGreyMatterPixels,
    InputScalarImageType* scalarImage,
    InputVectorImageType* vectorImage,
    OutputImageType* L0Image,
    OutputImageType* L1Image,
    std::vector<InputScalarImageIndexType>& L0greyList,
    std::vector<InputScalarImageIndexType>& L1greyList
    );

  /** This is called twice, once for L0 boundary, and once for L1 boundary. */
  virtual void SolvePDE(
      bool isInnerBoundary,
      std::vector<InputScalarImageIndexType>& listOfGreyPixels,
      InputScalarImageType* scalarImage,
      InputVectorImageType* vectorImage,
      OutputImageType* outputImage
      );
  
  /** For controlling convergence of the iteration. */
  double m_EpsilonConvergenceThreshold;
  
  /** Just in case we need to force a stop. */
  unsigned long int m_MaximumNumberOfIterations;
  
  /** So we can force a maximum length. */
  double m_MaximumLength;

  /** To control if we initialize boundaries. */
  bool m_InitializeBoundaries;
  
private:
  
  /**
   * Prohibited copy and assingment. 
   */
  RelaxStreamlinesFilter(const Self&); 
  void operator=(const Self&); 

  /** Added these, so that once the processing is done, we can expose these images, and do post processing on them. */
  OutputImagePointer m_L0Image;
  OutputImagePointer m_L1Image;

};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRelaxStreamlinesFilter.txx"
#endif

#endif
