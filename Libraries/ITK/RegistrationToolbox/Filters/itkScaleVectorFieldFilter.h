/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkScaleVectorFieldFilter_h
#define itkScaleVectorFieldFilter_h

#include <itkVector.h>
#include <itkImage.h>
#include <itkImageToImageFilter.h>


namespace itk {
/** 
 * \class ScaleVectorFieldFilter.
 * \brief This class takes two inputs, the first is a vector field,
 * such as the output from a registration force generator. The second
 * is another vector field, such as the gradient of an image. The images
 * must have the same size, and same vector dimensionality (enforced
 * via template parameters). The output is the first field, scaled
 * by the second.  If ScaleByComponents is true, the vectors
 * are simply multiplied componentwise. If ScaleByComponents is false,
 * the vectors are simply multiplied by the individual magnitude of the
 * second vector field.
 */
template <
    class TScalarType = double,          // Data type for scalars
    unsigned int NDimensions = 3>        // Number of Dimensions i.e. 2D or 3D
class ITK_EXPORT ScaleVectorFieldFilter :
  public ImageToImageFilter< Image< Vector<TScalarType, NDimensions>,  NDimensions>, // Input image
                             Image< Vector<TScalarType, NDimensions>,  NDimensions>  // Output image
                           >
{
public:

  /** Standard "Self" typedef. */
  typedef ScaleVectorFieldFilter                                                        Self;
  typedef ImageToImageFilter< Image< Vector<TScalarType, NDimensions>,  NDimensions>,
                              Image< Vector<TScalarType, NDimensions>,  NDimensions>
                            >                                                           Superclass;
  typedef SmartPointer<Self>                                                            Pointer;
  typedef SmartPointer<const Self>                                                      ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ScaleVectorFieldFilter, ImageToImageFilter);

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, NDimensions);

  /** Standard typedefs. */
  typedef Vector< TScalarType, itkGetStaticConstMacro(Dimension) >                    OutputPixelType;
  typedef Image< OutputPixelType, itkGetStaticConstMacro(Dimension) >                 OutputImageType;
  typedef typename Superclass::InputImageType                                         InputImageType;
  typedef typename Superclass::InputImagePointer                                      InputImagePointer;
  typedef typename Superclass::InputImageRegionType                                   InputImageRegionType;
  typedef typename InputImageType::PixelType                                          InputImagePixelType;
  
  /** Set the image that gets scaled at position 0. */
  virtual void SetImageThatWillBeScaled(const InputImageType *image) { this->SetNthInput(0, image); }

  /** Set the image that determines the amount of scaling at position 1. */
  virtual void SetImageThatDeterminesTheAmountOfScaling(const InputImageType *image) { this->SetNthInput(1, image); }

  /** We set the input images by number. */
  virtual void SetNthInput(unsigned int idx, const InputImageType *);
  
  /** If true, we scale componentwise, if false, we scale first image by magnitude of second image. */
  itkSetMacro(ScaleByComponents, bool);
  itkGetMacro(ScaleByComponents, bool);

  /** Writes image to log file. */
  void WriteVectorImage(std::string filename);
  
protected:
  ScaleVectorFieldFilter();
  ~ScaleVectorFieldFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;
  
  // Check before we start.
  virtual void BeforeThreadedGenerateData();
  
  // The main method to implement in derived classes, note, its threaded.
  virtual void ThreadedGenerateData( const InputImageRegionType &outputRegionForThread, ThreadIdType threadId);
  
  /** Scale by components. Defaults to true. */
  bool m_ScaleByComponents;
  
private:
  
  /**
   * Prohibited copy and assingment. 
   */
  ScaleVectorFieldFilter(const Self&); 
  void operator=(const Self&); 

};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkScaleVectorFieldFilter.txx"
#endif

#endif
