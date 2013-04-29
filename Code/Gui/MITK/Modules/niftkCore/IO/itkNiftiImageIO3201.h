/**************************************************************************
 * NifTK Note: This is take from ITK 3.20.1 (hence the name), which was
 * the version of ITK used within NifTK at the time of writing. The
 * version 3.20.1 actually signified a patched version of ITK, and
 * the patched tar file and patch can be seen in:
 *   NifTK/CMake/CMakeExternals/ITK.cmake and
 *   NifTK/CMake/CMakeExternals/PatchITK-3.20.cmake
 * However, none of this patching affects the Nifti reader, so the base
 * version of the Nifti reader used for this class is the same as in 3.20.0.
 *
 * The changes supplied here are to include the sform in the transformation.
 **************************************************************************/

/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkNiftiImageIO.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

/**
 *         The specification for this file format is taken from the
 *         web site http://www.mayo.edu/bir/PDF/ANALYZE75.pdf.
 * \author Hans J. Johnson
 *         The University of Iowa 2002
 */

#ifndef __itkNiftiImageIO3201_h
#define __itkNiftiImageIO3201_h

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include <fstream>
#include "itkImageIOBase.h"
#include <nifti1_io.h>
#include "niftkCoreExports.h"

namespace itk
{

/** \class NiftiImageIO3201
 *
 * \author Hans J. Johnson
 * \brief Class that defines how to read Nifti file format.
 * Nifti IMAGE FILE FORMAT - As much information as I can determine from sourceforge.net/projects/Niftilib
 *
 * \ingroup IOFilters
 */
  class NIFTKCORE_EXPORT ITK_EXPORT NiftiImageIO3201 : public ImageIOBase
{
public:
  /** Standard class typedefs. */
  typedef NiftiImageIO3201   Self;
  typedef ImageIOBase        Superclass;
  typedef SmartPointer<Self> Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(NiftiImageIO3201, Superclass);

  /*-------- This part of the interfaces deals with reading data. ----- */

  /** Determine if the file can be read with this ImageIO implementation.
   * \author Hans J Johnson
   * \param FileNameToRead The name of the file to test for reading.
   * \post Sets classes ImageIOBase::m_FileName variable to be FileNameToWrite
   * \return Returns true if this ImageIO can read the file specified.
   */
  virtual bool CanReadFile(const char* FileNameToRead);

  /** Set the spacing and dimension information for the set filename. */
  virtual void ReadImageInformation();

  /** Reads the data from disk into the memory buffer provided. */
  virtual void Read(void* buffer);

  /*-------- This part of the interfaces deals with writing data. ----- */

  /** Determine if the file can be written with this ImageIO implementation.
   * \param FileNameToWrite The name of the file to test for writing.
   * \author Hans J. Johnson
   * \post Sets classes ImageIOBase::m_FileName variable to be FileNameToWrite
   * \return Returns true if this ImageIO can write the file specified.
   */
  virtual bool CanWriteFile(const char * FileNameToWrite);

  /** Set the spacing and dimension information for the set filename. */
  virtual void WriteImageInformation();

  /** Writes the data to disk from the memory buffer provided. Make sure
   * that the IORegions has been set properly. */
  virtual void Write(const void* buffer);

  /** Calculate the region of the image that can be efficiently read 
   *  in response to a given requested region. */
  virtual ImageIORegion 
  GenerateStreamableReadRegionFromRequestedRegion( const ImageIORegion & requestedRegion ) const;

  /** A mode to allow the Nifti filter to read and write to the LegacyAnalyze75 format as interpreted by
    * the nifti library maintainers.  This format does not properly respect the file orientation fields.
    * The itkAnalyzeImageIO file reader/writer should be used to match the Analyze75 file definitions as
    * specified by the Mayo Clinic BIR laboratory.  By default this is set to false.
    */
  itkSetMacro(LegacyAnalyze75Mode,bool);
  itkGetConstMacro(LegacyAnalyze75Mode,bool);

protected:
  NiftiImageIO3201();
  ~NiftiImageIO3201();
  void PrintSelf(std::ostream& os, Indent indent) const;
  virtual bool GetUseLegacyModeForTwoFileWriting(void) const { return false; }
private:
  bool  MustRescale();
  void  DefineHeaderObjectDataType();
  void  SetNIfTIOrientationFromImageIO(unsigned short int origdims, unsigned short int dims);
  void  SetImageIOOrientationFromNIfTI(unsigned short int dims);
  void  SetImageIOMetadataFromNIfTI();

  nifti_image *     m_NiftiImage;
  double            m_RescaleSlope;
  double            m_RescaleIntercept;
  IOComponentType   m_OnDiskComponentType;
  bool              m_LegacyAnalyze75Mode;

  NiftiImageIO3201(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace itk

#endif // __itkNiftiImageIO3201_h
