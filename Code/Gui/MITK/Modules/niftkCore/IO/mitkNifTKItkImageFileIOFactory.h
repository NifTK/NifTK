/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef NifTKItkImageFileIOFactory_h
#define NifTKItkImageFileIOFactory_h

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include "niftkCoreExports.h"
#include "itkObjectFactoryBase.h"
#include "mitkBaseData.h"

namespace mitk
{

/**
 * \class NifTKItkImageFileIOFactory
 * \brief A NifTK specific MITK Object factory, to instantiate file readers based on ITK technology,
 * where the functionality will differ from the standard MITK Object factory, due to the
 * Analyze images being flipped/converted as per DRC "standards", and Nifti images
 * including sform if present.
 */
class NIFTKCORE_EXPORT NifTKItkImageFileIOFactory : public itk::ObjectFactoryBase
{
public:

  /** Standard class typedefs. */
  typedef NifTKItkImageFileIOFactory     Self;
  typedef itk::ObjectFactoryBase         Superclass;
  typedef itk::SmartPointer<Self>        Pointer;
  typedef itk::SmartPointer<const Self>  ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char* GetITKSourceVersion(void) const;
  virtual const char* GetDescription(void) const;

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);
  static NifTKItkImageFileIOFactory* FactoryNew() { return new NifTKItkImageFileIOFactory;}

  /** Run-time type information (and related methods). */
  itkTypeMacro(NifTKItkImageFileIOFactory, ObjectFactoryBase);

  /** Register one factory of this type  */
  static void RegisterOneFactory(void)
  {
    NifTKItkImageFileIOFactory::Pointer factory = NifTKItkImageFileIOFactory::New();
    ObjectFactoryBase::RegisterFactory(factory);
  }

protected:
  NifTKItkImageFileIOFactory();
  ~NifTKItkImageFileIOFactory();

private:
  NifTKItkImageFileIOFactory(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};


} // end namespace mitk

#endif
