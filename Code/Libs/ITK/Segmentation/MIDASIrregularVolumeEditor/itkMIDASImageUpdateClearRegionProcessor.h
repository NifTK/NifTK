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

#ifndef ITKMIDASIMAGEUPDATECLEARREGIONPROCESSOR_H
#define ITKMIDASIMAGEUPDATECLEARREGIONPROCESSOR_H

#include "itkMIDASImageUpdateRegionProcessor.h"

namespace itk
{

/**
 * \class MIDASImageUpdateClearRegionProcessor
 * \brief Class to support undo/redo of a clear operation (set value to zero), within a given region.
 */
template <class TPixel, unsigned int VImageDimension>
class ITK_EXPORT MIDASImageUpdateClearRegionProcessor : public MIDASImageUpdateRegionProcessor<TPixel, VImageDimension> {

public:

  /** Standard class typedefs */
  typedef MIDASImageUpdateClearRegionProcessor                     Self;
  typedef MIDASImageUpdateRegionProcessor<TPixel, VImageDimension> Superclass;
  typedef SmartPointer<Self>                                       Pointer;
  typedef SmartPointer<const Self>                                 ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MIDASImageUpdateClearRegionProcessor, MIDASImageUpdateRegionProcessor);

  /** Additional typedefs */
  typedef TPixel PixelType;
  typedef Image<TPixel, VImageDimension>  ImageType;
  typedef typename ImageType::Pointer     ImagePointer;
  typedef typename ImageType::IndexType   IndexType;
  typedef typename ImageType::SizeType    SizeType;
  typedef typename ImageType::RegionType  RegionType;

  /** Set the value that is used to wipe the image (normally zero). */
  itkSetMacro(WipeValue, PixelType);
  itkGetMacro(WipeValue, PixelType);

protected:
  MIDASImageUpdateClearRegionProcessor();
  void PrintSelf(std::ostream& os, Indent indent) const;
  virtual ~MIDASImageUpdateClearRegionProcessor() {}

  // This class, simply clears the image using value m_WipeValue.
  virtual void ApplyUpdateToAfterImage();

private:
  MIDASImageUpdateClearRegionProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  PixelType m_WipeValue;
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASImageUpdateClearRegionProcessor.txx"
#endif

#endif // ITKMIDASIMAGEUPDATECLEARREGIONPROCESSOR_H
