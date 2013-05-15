/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkNIFTKTransformIO_h
#define __itkNIFTKTransformIO_h
#include <niftkITKWin32ExportHeader.h>
#include <itkTransformIOBase.h>

namespace itk
{
class  NIFTKITK_WINEXPORT ITK_EXPORT NIFTKTransformIO : public TransformIOBase
{
public:
  typedef NIFTKTransformIO                Self;
  typedef TransformIOBase               Superclass;
  typedef SmartPointer<Self>            Pointer;
  typedef TransformBase                 TransformType;
  typedef Superclass::TransformPointer  TransformPointer;
  typedef Superclass::TransformListType TransformListType;
  /** Run-time type information (and related methods). */
  itkTypeMacro(NIFTKTransformIO,TransformIOBase);
  itkNewMacro(Self);

  /** Determine the file type. Returns true if this ImageIO can read the
   * file specified. */
  virtual bool CanReadFile(const char*);
  /** Determine the file type. Returns true if this ImageIO can read the
   * file specified. */
  virtual bool CanWriteFile(const char*);
  /** Reads the data from disk into the memory buffer provided. */
  virtual void Read();
  /** Writes the data to disk from the memory buffer provided. Make sure
   * that the IORegions has been set properly. The buffer is cast to a
   * pointer to the beginning of the image data. */
  virtual void Write();

protected:
  NIFTKTransformIO();
  virtual ~NIFTKTransformIO();
private:
  /** trim spaces and newlines from start and end of a string */
  std::string trim(std::string const& source, char const* delims = " \t\r\n");
};

}
#endif // __itkNIFTKTransformIO_h
