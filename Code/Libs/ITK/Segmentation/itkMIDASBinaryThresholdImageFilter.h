/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 10:55:39 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7447 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef itkMIDASBinaryThresholdImageFilter_h
#define itkMIDASBinaryThresholdImageFilter_h

#include "itkBinaryThresholdImageFilter.h"

namespace itk
{

  template <class TInputImage, class TOutputImage>
  class ITK_EXPORT MIDASBinaryThresholdImageFilter : public BinaryThresholdImageFilter<TInputImage, TOutputImage>
  {

    public:

      /** Standard class typedefs */
      typedef MIDASBinaryThresholdImageFilter                       Self;
      typedef BinaryThresholdImageFilter<TInputImage, TOutputImage> SuperClass;
      typedef SmartPointer<Self>                                    Pointer;
      typedef SmartPointer<const Self>                              ConstPointer;
      typedef typename TInputImage::IndexType                       IndexType;
      typedef typename TInputImage::SizeType                        SizeType;
      typedef typename TInputImage::RegionType                      RegionType;
      typedef typename TOutputImage::Pointer                        OutputImagePointer;

      /** Method for creation through the object factory */
      itkNewMacro(Self);

      /** Run-time type information (and related methods) */
      itkTypeMacro(MIDASBinaryThresholdImageFilter, BinaryThresholdImageFilter);

      /** Sets the axial cutoff region. */
      void SetAxialCutoffMaskedRegion(RegionType a) { m_AxialCutoffMaskedRegion = a; this->Modified(); }
      RegionType GetAxialCutoffMaskedRegion() const { return m_AxialCutoffMaskedRegion; }

    protected:

      MIDASBinaryThresholdImageFilter();
      virtual ~MIDASBinaryThresholdImageFilter() {};
      void PrintSelf(std::ostream& os, Indent indent) const;

      virtual void AfterThreadedGenerateData();

    private:
      MIDASBinaryThresholdImageFilter(const Self&); //purposely not implemented
      void operator=(const Self&); //purposely not implemented

      RegionType m_AxialCutoffMaskedRegion;

  }; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASBinaryThresholdImageFilter.txx"
#endif

#endif
