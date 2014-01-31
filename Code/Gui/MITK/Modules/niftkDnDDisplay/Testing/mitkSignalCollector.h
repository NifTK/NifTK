/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __mitkSignalCollector_h
#define __mitkSignalCollector_h

#include <itkCommand.h>
#include <itkEventObject.h>

#include <vector>

namespace mitk
{

class SignalCollector : public itk::Command
{
public:
  mitkClassMacro(SignalCollector, itk::Command);
  itkNewMacro(SignalCollector);

  typedef std::pair<const itk::Object*, itk::EventObject*> Signal;
  typedef std::vector<Signal> Signals;

  /// \brief Called when the event happens to the caller.
  void Execute(itk::Object* caller, const itk::EventObject& event);

  /// \brief Called when the event happens to the caller.
  void Execute(const itk::Object* object, const itk::EventObject& event);

  /// \brief Gets the signals collected by this object.
  const Signals& GetSignals() const;

  /// \brief Clears all the signals collected by now.
  void Clear();

protected:

  /// \brief Constructs an SignalCollector object.
  SignalCollector();

  /// \brief Destructs an SignalCollector object.
  virtual ~SignalCollector();

  /// \brief Prints the collected signals to the given stream or to the standard output if no stream is given.
  virtual void PrintSelf(std::ostream & os, itk::Indent indent) const;

private:

  /// \brief The signals collected by this object.
  Signals m_Signals;
};

}

#endif
