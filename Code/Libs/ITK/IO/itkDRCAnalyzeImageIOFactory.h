/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkDRCAnalyzeImageIOFactory_h
#define itkDRCAnalyzeImageIOFactory_h

#include <NifTKConfigure.h>
#include <niftkITKWin32ExportHeader.h>

#include <itkObjectFactoryBase.h>
#include <itkImageIOBase.h>

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include <itkObjectFactoryBase.h>

namespace itk
{
/** \class DRCAnalyzeImageIOFactory
   * \brief Create instances of DRCAnalyzeImageIO objects using an object factory.
   */
class NIFTKITK_WINEXPORT ITK_EXPORT DRCAnalyzeImageIOFactory : public ObjectFactoryBase
{
public:
  /** Standard class typedefs. */
  typedef DRCAnalyzeImageIOFactory    Self;
  typedef ObjectFactoryBase           Superclass;
  typedef SmartPointer<Self>          Pointer;
  typedef SmartPointer<const Self>    ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion(void) const;

  virtual const char * GetDescription(void) const;

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DRCAnalyzeImageIOFactory, ObjectFactoryBase);

protected:
  DRCAnalyzeImageIOFactory();
  ~DRCAnalyzeImageIOFactory();

private:
  DRCAnalyzeImageIOFactory(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};


} // end namespace itk

#endif
