/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkFluidVelocityToDeformationFilter_h
#define __itkFluidVelocityToDeformationFilter_h

#include <itkVector.h>
#include <itkImage.h>
#include <itkImageToImageFilter.h>


/**
 * Friend class for unit testing. 
 */  
class FluidVelocityToDeformationFilterUnitTest;

namespace itk {
/** 
 * \class FluidVelocityToDeformationFilter.
 * \brief This class takes two inputs, the first is the current deformation vector field
 * and the second is the fluid velocity vector field. The output is a vector field
 * of deformation according to Christensens paper. 
 */
template <
    class TScalarType = double,          // Data type for scalars
    unsigned int NDimensions = 3>        // Number of Dimensions i.e. 2D or 3D
class ITK_EXPORT FluidVelocityToDeformationFilter :
  public ImageToImageFilter< Image< Vector<TScalarType, NDimensions>,  NDimensions>, // Input image
                             Image< Vector<TScalarType, NDimensions>,  NDimensions>  // Output image
                           >
{
public:

  /** Standard "Self" typedef. */
  typedef FluidVelocityToDeformationFilter                                                 Self;
  typedef ImageToImageFilter< Image< Vector<TScalarType, NDimensions>,  NDimensions>,
                              Image< Vector<TScalarType, NDimensions>,  NDimensions>
                            >                                                           Superclass;
  typedef SmartPointer<Self>                                                            Pointer;
  typedef SmartPointer<const Self>                                                      ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FluidVelocityToDeformationFilter, ImageToImageFilter);

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, NDimensions);

  /** Standard typedefs. */
  typedef Vector< TScalarType, itkGetStaticConstMacro(Dimension) >                    OutputPixelType;
  typedef Image< OutputPixelType, itkGetStaticConstMacro(Dimension) >                 OutputImageType;
  typedef typename Superclass::InputImageType                                         InputImageType;
  typedef typename Superclass::InputImagePointer                                      InputImagePointer;
  typedef typename Superclass::InputImageRegionType                                   InputImageRegionType;
  typedef Image<float, Dimension>                                                     InputImageMaskType; 

  /** Set the current deformation field at position 0. */
  virtual void SetCurrentDeformationField(const InputImageType *image) { this->SetNthInput(0, image); }

  /** Set the velocity field at position 1. */
  virtual void SetVelocityField(const InputImageType *image) { this->SetNthInput(1, image); }

  /** We set the input images by number. */
  virtual void SetNthInput(unsigned int idx, const InputImageType *);
  
  /** 
   * Set fixed image mask. 
   */
  virtual void SetInputMask(const InputImageMaskType* inputImageMask) { this->m_InputMask = inputImageMask; } 
  
  /** Get the maximum deformation. */
  itkGetMacro(MaxDeformation, double);
  
  /**
   * Set/Get. 
   */
  itkSetMacro(IsNegativeVelocity, bool); 
  itkGetMacro(IsNegativeVelocity, bool); 
  itkSetMacro(IsTakeDerivative, bool); 

protected:
  FluidVelocityToDeformationFilter();
  virtual ~FluidVelocityToDeformationFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const;
  
  // Check before we start.
  virtual void BeforeThreadedGenerateData();
  
  // The main method to implement in derived classes, note, its threaded.
  virtual void ThreadedGenerateData( const InputImageRegionType &outputRegionForThread, int);
  
  /**
   * Compute the maximum deformation. 
   */ 
  virtual void AfterThreadedGenerateData(); 
  
  /** The time step size. */
  double m_MaxDeformation;
  
  /**
   * Negative the velocity field? 
   */
  bool m_IsNegativeVelocity; 
  
  /**
   * Input image mask. 
   */
  const InputImageMaskType* m_InputMask; 
  
  /**
   * Taking derivative?
   */
  bool m_IsTakeDerivative; 
  
private:
  
  /**
   * Prohibited copy and assingment. 
   */
  FluidVelocityToDeformationFilter(const Self&); 
  void operator=(const Self&); 

  /**
   * Friend class for unit testing. 
   */  
  friend class ::FluidVelocityToDeformationFilterUnitTest;
  
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFluidVelocityToDeformationFilter.txx"
#endif

#endif
