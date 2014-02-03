/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __mitkQtSignalCollector_h
#define __mitkQtSignalCollector_h

#include <QObject>

#include <mitkCommon.h>
#include <itkObject.h>
#include <itkObjectFactory.h>

#include <vector>

namespace mitk
{

class QtSignalCollector : public QObject, public itk::Object
{
  Q_OBJECT

public:
  mitkClassMacro(QtSignalCollector, QObject);
  itkNewMacro(QtSignalCollector);

  typedef std::pair<const QObject*, const char*> Signal;
  typedef std::vector<Signal> Signals;

  /// \brief Gets the signals collected by this object.
  const Signals& GetSignals() const;

  /// \brief Clears all the signals collected by now.
  virtual void Clear();

public slots:

//  void OnEventOccurred(const QObject* object, QEvent* event);

  void OnSignalEmitted(const QObject* object, const char* signal);

protected:

  /// \brief Constructs an QtSignalCollector object.
  QtSignalCollector();

  /// \brief Destructs an QtSignalCollector object.
  virtual ~QtSignalCollector();

  /// \brief Prints the collected signals to the given stream or to the standard output if no stream is given.
  virtual void PrintSelf(std::ostream & os, itk::Indent indent) const;

private:

  /// \brief The signals collected by this object.
  Signals m_Signals;
};

}

#endif
