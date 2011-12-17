/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3326 $
 Last modified by  : $Author: mjc $
 
 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkUCLLabelVotingImageFilter_h
#define __itkUCLLabelVotingImageFilter_h

#include "itkImage.h"
#include "itkImageToImageFilter.h"

namespace itk
{

/**
 * UCLLabelVotingImageFilter: just copied from LabelVotingImageFilter
 * with added option to pick a random label when votes are equal. 
 */
template <typename TInputImage, typename TOutputImage>
class ITK_EXPORT UCLLabelVotingImageFilter :
    public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef UCLLabelVotingImageFilter Self;
  typedef ImageToImageFilter< TInputImage, TOutputImage > Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods) */
  itkTypeMacro(UCLLabelVotingImageFilter, ImageToImageFilter);
  
  /** Extract some information from the image types.  Dimensionality
   * of the two images is assumed to be the same. */
  typedef typename TOutputImage::PixelType OutputPixelType;
  typedef typename TInputImage::PixelType  InputPixelType;
  
  /** Extract some information from the image types.  Dimensionality
   * of the two images is assumed to be the same. */
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      TInputImage::ImageDimension );
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TOutputImage::ImageDimension);
  
  /** Image typedef support */
  typedef TInputImage                           InputImageType;
  typedef TOutputImage                          OutputImageType;
  typedef typename InputImageType::ConstPointer InputImagePointer;
  typedef typename OutputImageType::Pointer     OutputImagePointer;
  
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
  
  /** Unset label value for undecided pixels and turn on automatic selection.
    */
  virtual void UnsetLabelForUndecidedPixels()
  {
    if ( this->m_HasLabelForUndecidedPixels )
    {
      this->m_HasLabelForUndecidedPixels = false;
      this->Modified();
    }
  }
    
protected:   
  /**
   * Constructor. 
   */
  UCLLabelVotingImageFilter() { this->m_HasLabelForUndecidedPixels = false; srand(time(NULL)); }
  /**
   * Destructor. 
   */
  virtual ~UCLLabelVotingImageFilter() {}  
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
  
private:
  UCLLabelVotingImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
}; 
    
  
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkUCLLabelVotingImageFilter.txx"
#endif


#endif



