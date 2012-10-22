/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 12:21:57 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7449 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef ITKMIDASIMAGEUPDATECOPYREGIONPROCESSOR_H
#define ITKMIDASIMAGEUPDATECOPYREGIONPROCESSOR_H

#include "itkMIDASImageUpdateRegionProcessor.h"
#include "itkPasteImageFilter.h"

namespace itk
{

/**
 * \class MIDASImageUpdateCopyRegionProcessor
 * \brief Class to support undo/redo of a copy operation, within a given region.
 */
template <class TPixel, unsigned int VImageDimension>
class ITK_EXPORT MIDASImageUpdateCopyRegionProcessor : public MIDASImageUpdateRegionProcessor<TPixel, VImageDimension> {

public:

  /** Standard class typedefs */
  typedef MIDASImageUpdateCopyRegionProcessor                      Self;
  typedef MIDASImageUpdateRegionProcessor<TPixel, VImageDimension> Superclass;
  typedef SmartPointer<Self>                                       Pointer;
  typedef SmartPointer<const Self>                                 ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MIDASImageUpdateCopyRegionProcessor, MIDASImageUpdateRegionProcessor);

  /** Additional typedefs */
  typedef TPixel PixelType;
  typedef Image<TPixel, VImageDimension>  ImageType;
  typedef typename ImageType::Pointer     ImagePointer;
  typedef typename ImageType::IndexType   IndexType;
  typedef typename ImageType::SizeType    SizeType;
  typedef typename ImageType::RegionType  RegionType;
  typedef PasteImageFilter<ImageType, ImageType> PasteImageFilterType;
  typedef typename PasteImageFilterType::Pointer PasteImagePointerType;

  /** Set the source image. Data is copied from here to destination image. */
  itkSetObjectMacro(SourceImage, ImageType);
  itkGetObjectMacro(SourceImage, ImageType);

  /** Set the source region of interest. */
  itkSetMacro(SourceRegionOfInterest, RegionType);
  itkGetMacro(SourceRegionOfInterest, RegionType);

  /** Overloaded method to provide simple acess via a std::vector, where we assume the length is 6 corresponding to the first 3 numbers indicating the starting index, and the next 3 numbers indicating the region size. */
  void SetSourceRegionOfInterest(std::vector<int> &region);

protected:
  MIDASImageUpdateCopyRegionProcessor();
  void PrintSelf(std::ostream& os, Indent indent) const;
  virtual ~MIDASImageUpdateCopyRegionProcessor() {}

  // This class
  virtual void ApplyUpdateToAfterImage();

private:
  MIDASImageUpdateCopyRegionProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  ImagePointer m_SourceImage;
  RegionType   m_SourceRegionOfInterest;
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASImageUpdateCopyRegionProcessor.txx"
#endif

#endif // ITKIMAGEUPDATEBYREGIONPROCESSOR_H
