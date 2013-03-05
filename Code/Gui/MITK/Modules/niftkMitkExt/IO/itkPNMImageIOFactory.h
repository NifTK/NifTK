/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkPNMImageIOFactory.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkPNMImageIOFactory_h
#define __itkPNMImageIOFactory_h

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include <itkObjectFactoryBase.h>
#include <itkImageIOBase.h>
#include "niftkMitkExtExports.h"

namespace itk
{
/** \class PNMImageIOFactory
 * \brief Create instances of PNMImageIO objects using an object factory.
 */
class NIFTKMITKEXT_EXPORT PNMImageIOFactory : public ObjectFactoryBase
{
public:  
  /** Standard class typedefs. */
  typedef PNMImageIOFactory        Self;
  typedef ObjectFactoryBase        Superclass;
  typedef SmartPointer<Self>       Pointer;
  typedef SmartPointer<const Self> ConstPointer;
  
  /** Class methods used to interface with the registered factories. */
  virtual const char* GetITKSourceVersion(void) const;
  virtual const char* GetDescription(void) const;
    
  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);
  static PNMImageIOFactory* FactoryNew() { return new PNMImageIOFactory;}
  /** Run-time type information (and related methods). */
  itkTypeMacro(PNMImageIOFactory, ObjectFactoryBase);

  /** Register one factory of this type  */
  static void RegisterOneFactory(void)
    {
    PNMImageIOFactory::Pointer PNMFactory = PNMImageIOFactory::New();
    ObjectFactoryBase::RegisterFactory(PNMFactory);
    }
  
protected:
  PNMImageIOFactory();
  ~PNMImageIOFactory();

private:
  PNMImageIOFactory(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};
  
  
} // end namespace itk

#endif
