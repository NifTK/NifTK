/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef ITKIMAGEUPDATEREGIONPROCESSOR_H
#define ITKIMAGEUPDATEREGIONPROCESSOR_H

#include "itkImageUpdateProcessor.h"
#include "itkExtractImageFilter.h"
#include "itkPasteImageFilter.h"

namespace itk
{

/**
 * \class ImageUpdateRegionProcessor
 * \brief Provides methods to do Undo/Redo within a specific Region.
 */
template <class TPixel, unsigned int VImageDimension>
class ITK_EXPORT ImageUpdateRegionProcessor : public ImageUpdateProcessor<TPixel, VImageDimension> {

public:

  /** Standard class typedefs */
  typedef ImageUpdateRegionProcessor                    Self;
  typedef ImageUpdateProcessor<TPixel, VImageDimension> Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageUpdateRegionProcessor, ImageUpdateProcessor);

  /** Additional typedefs */
  typedef TPixel PixelType;
  typedef Image<TPixel, VImageDimension>  ImageType;
  typedef typename ImageType::Pointer     ImagePointer;
  typedef typename ImageType::IndexType   IndexType;
  typedef typename ImageType::SizeType    SizeType;
  typedef typename ImageType::RegionType  RegionType;
  typedef itk::ExtractImageFilter<ImageType, ImageType> ExtractImageFilterType;
  typedef typename ExtractImageFilterType::Pointer      ExtractImageFilterPointer;
  typedef itk::PasteImageFilter<ImageType, ImageType>   PasteImageFilterType;
  typedef typename PasteImageFilterType::Pointer        PasteImageFilterPointer;

  /** Set the destination region of interest, which controls the region that is copied into m_BeforeImage and m_After image for Undo/Redo purposes. */
  itkSetMacro(DestinationRegionOfInterest, RegionType);
  itkGetMacro(DestinationRegionOfInterest, RegionType);

  /** Overloaded method to provide simple acess via a std::vector, where we assume the length is 6 corresponding to the first 3 numbers indicating the starting index, and the next 3 numbers indicating the region size. */
  void SetDestinationRegionOfInterest(std::vector<int> &region);

  /** This will copy the m_BeforeImage into the m_DestinationImage */
  virtual void Undo();

  /** This will copy the m_AfterImage into the m_DestinationImage. This method should also be called to execute the whole process first time round. */
  virtual void Redo();

protected:
  ImageUpdateRegionProcessor();
  void PrintSelf(std::ostream& os, Indent indent) const;
  virtual ~ImageUpdateRegionProcessor() {}

  /** Returns the after image, so derived classes can apply an update. */
  itkGetObjectMacro(AfterImage, ImageType);
  itkSetObjectMacro(AfterImage, ImageType);

  /** Derived classes calculate whatever update they like, but can only affect the m_AfterImage, which must be within the destination region of interest. */
  virtual void ApplyUpdateToAfterImage() = 0;

  virtual void ValidateInputs();

private:
  ImageUpdateRegionProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  void CopyImageRegionToDestination(ImagePointer sourceImage);

  bool         m_UpdateCalculated;
  RegionType   m_DestinationRegionOfInterest;
  ImagePointer m_BeforeImage;
  ImagePointer m_AfterImage;

}; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageUpdateRegionProcessor.txx"
#endif

#endif // ITKIMAGEUPDATEREGIONPROCESSOR_H
