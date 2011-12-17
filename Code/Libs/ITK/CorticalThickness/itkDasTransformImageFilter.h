/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-14 11:37:54 +0100 (Wed, 14 Sep 2011) $
 Revision          : $Revision: 7310 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkDasTransformImageFilter_h
#define __itkDasTransformImageFilter_h
#include "itkVector.h"
#include "itkImage.h"
#include "itkImageToImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"

namespace itk {

/**
 * \class DasTransformImageFilter
 * \brief Transform an image using a vector field, where each vector represents the absolute sampling position,
 * not a vector displacement.
 * 
 * At the moment this class is quite basic, with internal linear interpolation.
 */

template <typename TScalarType, unsigned int NDimensions = 3>
class ITK_EXPORT DasTransformImageFilter:
    public ImageToImageFilter<
                               Image< TScalarType, NDimensions>,
                               Image< TScalarType, NDimensions>
                             >
{

public:  
  /** Standard class typedefs. */
  typedef DasTransformImageFilter                               Self;
  typedef ImageToImageFilter<
                              Image< TScalarType, NDimensions>,
                              Image< TScalarType, NDimensions>
                            >                                   Superclass;
  typedef SmartPointer<Self>                                    Pointer;
  typedef SmartPointer<const Self>                              ConstPointer;
  typedef Image< TScalarType, NDimensions >                     InputImageType;
  typedef typename InputImageType::IndexType                    InputImageIndexType;
  typedef typename InputImageType::RegionType                   InputImageRegionType;
  typedef Vector< TScalarType, NDimensions >                    VectorPixelType;
  typedef Image< VectorPixelType, NDimensions >                 VectorImageType;
  typedef TScalarType                                           OutputPixelType;
  typedef InputImageType                                        OutputImageType;
  typedef LinearInterpolateImageFunction< InputImageType, 
                                          TScalarType >         LinearInterpolatorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DasTransformImageFilter, ImageToImageFilter);

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, NDimensions);

  /** Set/Get the filname to which we dump the transformed moving image. */
  itkSetMacro(FileName, std::string);
  itkGetMacro(FileName, std::string);
  
  /** Set/Get a flag to determin if we will dump the transformed moving image. */
  itkSetMacro(WriteTransformedMovingImage, bool);
  itkGetMacro(WriteTransformedMovingImage, bool);
  
  /** Set/Get the default or 'padding' value. Defaults to 0. */
  itkSetMacro(DefaultValue, OutputPixelType);
  itkGetMacro(DefaultValue, OutputPixelType);
  
  /** Set the transformation we are transforming by. */
  void SetPhiTransformation(VectorImageType* image) { m_PhiTransformation = image; }
  
protected:
  DasTransformImageFilter();
  ~DasTransformImageFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  // Before we start multi-threaded section.
  virtual void BeforeThreadedGenerateData();

  // The main method to implement in derived classes, note, its threaded.
  virtual void ThreadedGenerateData( const InputImageRegionType &outputRegionForThread, int);

  // After the threaded bit.
  virtual void AfterThreadedGenerateData( void );
  
private:
  
  /**
   * Prohibited copy and assignment. 
   */
  DasTransformImageFilter(const Self&); 
  void operator=(const Self&); 

  /** The transformation phi. */
  typename VectorImageType::Pointer m_PhiTransformation;

  /** The interpolator. */
  typename LinearInterpolatorType::Pointer m_Interpolator;
  
  /** Filename to write transformed moving image to. */
  std::string m_FileName;
  
  /** Boolean to control if we write out the transformed moving image. */
  bool m_WriteTransformedMovingImage;
  
  /** The default or 'padding' value. */
  OutputPixelType m_DefaultValue;
  
}; // end class
    
} // end namespace
    
#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDasTransformImageFilter.txx"
#endif

#endif

