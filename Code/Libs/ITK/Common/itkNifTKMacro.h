/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkNifTKMacro_h
#define itkNifTKMacro_h
#include <itkMutexLockHolder.h>
#include <itkObjectFactory.h>

/**
 * \brief Assumes the presence of an <code>itk::FastMutexLock::Pointer</code> variable called <code>m_Mutex</code>
 * and a member variable <code>m_"name"</code> and generates a <code>type Get"name"() const</code> method.
 */
#define itkThreadSafeGetConstMacro(name,type) \
  virtual type Get##name () const \
  { \
    itkDebugMacro("returning " << #name " of " << this->m_##name ); \
    return this->m_##name; \
  }

/**
 * \brief Assumes the presence of an <code>itk::FastMutexLock::Pointer</code> variable called <code>m_Mutex</code>
 * and a member variable <code>m_"name"</code> and generates a <code>type Get"name"() const</code> method.
 */
#define itkThreadSafeSetMacro(name,type) \
  virtual void Set##name (const type _arg) \
  { \
    itkDebugMacro("setting " #name " to " << _arg); \
    if (this->m_##name != _arg) \
      { \
      itk::MutexLockHolder<itk::FastMutexLock> lock(*this->m_Mutex); \
      this->m_##name = _arg; \
      this->Modified(); \
      } \
  }

#endif // itkNifTKMacro_h
