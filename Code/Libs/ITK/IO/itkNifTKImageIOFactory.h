/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkNifTKImageIOFactory_h
#define itkNifTKImageIOFactory_h

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include <NifTKConfigure.h>
#include <niftkITKWin32ExportHeader.h>

#include <itkObjectFactoryBase.h>
#include <itkImageIOBase.h>

namespace itk
{
/**
 * \class NifTKImageIOFactory
 * \brief Creates instances of NifTK specific image IO objects.
 */
class NIFTKITK_WINEXPORT ITK_EXPORT NifTKImageIOFactory : public ObjectFactoryBase
{
public:  
  /** Standard class typedefs. */
  typedef NifTKImageIOFactory   Self;
  typedef ObjectFactoryBase  Superclass;
  typedef SmartPointer<Self>  Pointer;
  typedef SmartPointer<const Self>  ConstPointer;
  
  /** Class methods used to interface with the registered factories. */
  virtual const char* GetITKSourceVersion(void) const;
  virtual const char* GetDescription(void) const;
    
  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self)

  /** Run-time type information (and related methods). */
  itkTypeMacro(NifTKImageIOFactory, ObjectFactoryBase)

protected:
  NifTKImageIOFactory();
  virtual ~NifTKImageIOFactory();

private:
  NifTKImageIOFactory(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};
  
  
} // end namespace itk

#endif
