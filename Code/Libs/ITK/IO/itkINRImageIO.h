/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-08-15 10:33:09 +0100 (Mon, 15 Aug 2011) $
 Revision          : $Revision: 7071 $
 Last modified by  : $Author: sj $

 Original author   : m.modat@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkINRImageIO_h
#define __itkINRImageIO_h

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include "NifTKConfigure.h"
#include "niftkITKWin32ExportHeader.h"

#include "itkImageIOBase.h"
#include "itkByteSwapper.h"

namespace itk
{

/**
 * \class INRImageIO
 * \brief ITK IO class to load INRIA image format.
 */
class NIFTKITK_WINEXPORT   ITK_EXPORT INRImageIO : public ImageIOBase
{
public:
  /** Standard class typedefs. */
  typedef INRImageIO            Self;
  typedef ImageIOBase  Superclass;
  typedef SmartPointer<Self>  Pointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(INRImageIO, ImageIOBase);

  /** Determine the file type. Returns true if this ImageIO can read the
   * file specified. */
  virtual bool CanReadFile(const char*);
  
  /** Set the spacing and diemention information for the set filename. */
  virtual void ReadImageInformation();

  /** Reads the data from disk into the memory buffer provided. */
  virtual void Read(void* buffer);

  /** Read header and setup data **/
  virtual bool ReadHeader(); 
  unsigned int m_pixelSize;

  /** Reads 3D data from multiple files assuming one slice per file. */
  virtual void ReadVolume(void* buffer);

  /** Compute the size (in bytes) of the components of a pixel. For
   * example, and RGB pixel of unsigned char would have a 
   * component size of 1 byte. */
  virtual unsigned int GetComponentSize() const;

  /** Determine the file type. Returns true if this ImageIO can read the
   * file specified. */
  virtual bool CanWriteFile(const char*);

  /** Writes the spacing and dimentions of the image.
   * Assumes SetFileName has been called with a valid file name. */
  virtual void WriteImageInformation();

  /** Writes the data to disk from the memory buffer provided. Make sure
   * that the IORegion has been set properly. */
  virtual void Write(const void* buffer);

protected:
  INRImageIO();
  virtual ~INRImageIO();

  /** Get an integer field from the header */
  virtual bool GetParamFromHeader( char* headerPtr, const char *variableToFind, const char *patternToFind, int *target );
  virtual bool GetParamFromHeader( char* headerPtr, const char *variableToFind, const char *patternToFind, char *target );
  virtual bool GetParamFromHeader( char* headerPtr, const char *variableToFind, const char *patternToFind, float *target );
  virtual bool GetParamFromHeader( char* headerPtr, const char *variableToFind, const char *patternToFind, int *target, int defaultValue );
  virtual bool GetParamFromHeader( char* headerPtr, const char *variableToFind, const char *patternToFind, unsigned int *target, unsigned int defaultValue );
  virtual bool GetParamFromHeader( char* headerPtr, const char *variableToFind, const char *patternToFind, float *target, float defaultValue );

  void PrintSelf(std::ostream& os, Indent indent) const;

  void WriteSlice(std::string& fileName, const void* buffer);

  void SwapBytesIfNecessary( void* buffer, unsigned long numberOfPixels );

private:
  INRImageIO(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  bool m_compression;

  bool m_LittleEndian;
};

} // end namespace itk

#endif // __itkINRImageIO_h

