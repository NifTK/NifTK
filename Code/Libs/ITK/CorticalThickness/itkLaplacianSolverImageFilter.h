/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkLaplacianSolverImageFilter_h
#define itkLaplacianSolverImageFilter_h

#include <itkImage.h>
#include "itkBaseCTEFilter.h"


namespace itk
{
/**
 * \class LaplacianSolverImageFilter
 * \brief Solves Laplace equation over the cortical volume.
 * 
 * This filter implements step (7) in Jones et al. Human Brain Mapping
 * 11:12-32 (2000). The input must be an image with exactly three values,
 * corresponding to white matter, grey matter (cortical volume), and 
 * extra-cerebral volume, (i.e. non-brain). Laplaces equation is solved
 * using the Jacobi method (Press et al.), where convergence stops once
 * the change in field energy goes below a threshold. The Jones 2000 paper 
 * uses a formulation simplified to isotropic voxels. However, 
 * in Diep et. al. ISBI 2007, it is generalised to anisotropic voxels.
 * So this implementation can do anisotropic voxels, using Than Dieps
 * generalization.
 * 
 * The output is an image of voltage potentials.
 * 
 * You should specify the values for each type of matter.
 * They all default to -1. If you are constructing a pipeline,
 * then the itkCheckForThreeLevelsFilter will check this for you,
 * and try to set reasonable defaults, so there is no checking code
 * contained within this class.
 * 
 * \ingroup ImageFeatureExtraction */
template <class TInputImage, typename TScalarType=double>
class ITK_EXPORT LaplacianSolverImageFilter : 
    public BaseCTEFilter< TInputImage >
{
public:
  /** Standard "Self" & Superclass typedef.   */
  typedef LaplacianSolverImageFilter   Self;
  typedef BaseCTEFilter< TInputImage > Superclass;
  typedef SmartPointer<Self>           Pointer;
  typedef SmartPointer<const Self>     ConstPointer;

  /** Run-time type information (and related methods)  */
  itkTypeMacro(LaplacianSolverImageFilter, BaseCTEFilter);
  
  /** Method for creation through the object factory.  */
  itkNewMacro(Self);

  /** Extract some information from the image types.  Dimensionality
   * of the two images is assumed to be the same. */
  typedef typename TInputImage::PixelType                     InputPixelType;
  typedef InputPixelType                                      OutputPixelType;
  
  /** Image typedef support. */
  typedef TInputImage                                         InputImageType;
  typedef typename InputImageType::Pointer                    InputImagePointer;
  typedef typename InputImageType::IndexType                  InputImageIndexType;
  typedef Image<OutputPixelType, TInputImage::ImageDimension> OutputImageType;
  typedef typename OutputImageType::Pointer                   OutputImagePointer;
  typedef typename OutputImageType::SpacingType               OutputImageSpacing;

  /** Sets the segmented image, at input 0. */
  void SetSegmentedImage(const InputImageType *image) {this->SetNthInput(0, const_cast<InputImageType *>(image)); }
  
  /** Check this once we have finished. */
  itkGetMacro(CurrentIteration, unsigned long int);
  itkSetMacro(CurrentIteration, unsigned long int);
  
  /** Set the low voltage threshold. Default 0. */
  itkSetMacro(LowVoltage, OutputPixelType);
  itkGetMacro(LowVoltage, OutputPixelType);

  /** Set the high voltage threshold. Default 10000. */
  itkSetMacro(HighVoltage, OutputPixelType);
  itkGetMacro(HighVoltage, OutputPixelType);

  /** Set convergence threshold. Default 10E-5. */
  itkSetMacro(EpsilonConvergenceThreshold, OutputPixelType);
  itkGetMacro(EpsilonConvergenceThreshold, OutputPixelType);

  /** Set MaximumNumberOfIterations threshold. Default 200. */
  itkSetMacro(MaximumNumberOfIterations, unsigned long int );
  itkGetMacro(MaximumNumberOfIterations, unsigned long int );

  /** Turns Gauss Siedel optimisation on or off. Default on.*/
  itkSetMacro(UseGaussSeidel, bool);
  itkGetMacro(UseGaussSeidel, bool);
  
protected:
  LaplacianSolverImageFilter();
  virtual ~LaplacianSolverImageFilter()  {}

  /** Standard Print Self. */
  virtual void PrintSelf(std::ostream&, Indent) const;

  // The main filter method. Note, single threaded.
  virtual void GenerateData();
  
private:
  LaplacianSolverImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
  /** The low voltage value (see paper). */
  OutputPixelType m_LowVoltage;
  
  /** The high voltage value (see paper). */
  OutputPixelType m_HighVoltage;
  
  /** Convergence stops when the field energy is below this threshold. Default 10E-5. */
  OutputPixelType m_EpsilonConvergenceThreshold;
 
  /** Just in case we need to force a stop. */
  unsigned long int m_MaximumNumberOfIterations;
  
  /** So we can keep track of current iteration. */
  unsigned long int m_CurrentIteration;
  
  /** Use Gauss Siedel or not. Default on. */
  bool m_UseGaussSeidel;
  
};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLaplacianSolverImageFilter.txx"
#endif

#endif
