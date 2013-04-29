/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkNIFTKTransformIO.h"
#include <itksys/SystemTools.hxx>
#include <vnl/vnl_matlab_read.h>
#include <vnl/vnl_matlab_write.h>
#include "itkUCLMacro.h"

namespace itk
{
NIFTKTransformIO::
NIFTKTransformIO()
{
}

NIFTKTransformIO::
~NIFTKTransformIO()
{
}

bool
NIFTKTransformIO::
CanReadFile(const char*  /* fileName*/)
{
  return true;
}

bool
NIFTKTransformIO::
CanWriteFile(const char*  /* fileName*/)
{
  return true;
}

std::string 
NIFTKTransformIO::
trim(std::string const& source, char const* delims)
{
  std::string result(source);
  std::string::size_type index = result.find_last_not_of(delims);
  if(index != std::string::npos)
    {
    result.erase(++index);
    }

  index = result.find_first_not_of(delims);
  if(index != std::string::npos)
    {
    result.erase(0, index);
    }
  else
    {
    result.erase();
    }
  return result;
}

void 
NIFTKTransformIO::
Read()
{  
  TransformPointer transform;
  std::ifstream in;
  in.open ( this->GetFileName(), std::ios::in | std::ios::binary );
  if( in.fail() )
    {
      in.close();
      niftkitkDebugMacro ("The file could not be opened for read access "
                        << std::endl << "Filename: \"" << this->GetFileName() << "\"" );
    }

  OStringStream InData;

  // in.get ( InData );
  std::filebuf *pbuf;
  pbuf=in.rdbuf();

  // get file size using buffer's members
  int size=pbuf->pubseekoff (0,std::ios::end,std::ios::in);
  pbuf->pubseekpos (0,std::ios::in);

  // allocate memory to contain file data
  char* buffer=new char[size+1];

  // get file data  
  pbuf->sgetn (buffer,size); 
  buffer[size]='\0';
  niftkitkDebugMacro ( "Read file transform Data" );
  InData << buffer;

  delete[] buffer;
  std::string data = InData.str();
  in.close();

  // Read line by line
  vnl_vector<double> VectorBuffer;
  std::string::size_type position = 0;
  
  Array<double> TmpParameterArray;
  Array<double> TmpFixedParameterArray;
  TmpParameterArray.clear();
  TmpFixedParameterArray.clear();
  bool haveFixedParameters = false;
  bool haveParameters = false;
  //
  // check for line end convention
  std::string line_end("\n");
  if(data.find('\n') == std::string::npos)
    {
    if(data.find('\r') == std::string::npos)
      {
        niftkitkErrorMacro (<< "No line ending character found, not a valid ITK Transform TXT file" );
      }
    line_end = "\r";
    }
  while ( position < data.size() )
    {
    // Find the next string
    std::string::size_type end = data.find ( line_end, position );
    std::string line = trim ( data.substr ( position, end - position ) );
    position = end+1;
    niftkitkDebugMacro ("Found line: \"" << line << "\"" );

    if ( line.length() == 0 )
      {
      continue;
      }
    if (line[0] == '#' || std::string::npos == line.find_first_not_of (" \t"))
      {
      // Skip lines beginning with #, or blank lines
      continue;
      }

    // Get the name
    end = line.find ( ":" );
    if ( end == std::string::npos )
      {
        // Throw an error
        niftkitkExceptionMacro ( "Tags must be delimited by :" );
      }
    std::string Name = trim ( line.substr ( 0, end ) );
    std::string Value = trim ( line.substr ( end + 1, line.length() ) );
    // Push back 
    niftkitkDebugMacro ( "Name: \"" << Name << "\"" );
    niftkitkDebugMacro ( "Value: \"" << Value << "\"" );
    itksys_ios::istringstream parse ( Value );
    VectorBuffer.clear();
    if ( Name == "Transform" )
      {
      this->CreateTransform(transform,Value);
      this->GetReadTransformList().push_back ( transform );
      }
    else if ( Name == "Parameters" || Name == "FixedParameters" )
      {
      VectorBuffer.clear();

      // Read them
      parse >> VectorBuffer;
      niftkitkDebugMacro ( "Parsed: " << VectorBuffer );
      if ( Name == "Parameters" )
        {
        TmpParameterArray = VectorBuffer;
        niftkitkDebugMacro ("Setting Parameters: " << TmpParameterArray );
        if ( haveFixedParameters )
          {
          transform->SetFixedParameters ( TmpFixedParameterArray );
          niftkitkDebugMacro ( "Set Transform Fixed Parameters" );
          transform->SetParametersByValue ( TmpParameterArray );
          niftkitkDebugMacro ( "Set Transform Parameters" );
          TmpParameterArray.clear();
          TmpFixedParameterArray.clear(); 
          haveFixedParameters = false;
          haveParameters = false;
          }
        else
          {
          haveParameters = true;
          }   
        }
      else if ( Name == "FixedParameters" )
        {
        TmpFixedParameterArray = VectorBuffer;
        niftkitkDebugMacro ( "Setting Fixed Parameters: " << TmpFixedParameterArray );
        if ( !transform )
          {
          itkExceptionMacro ("Please set the transform before parameters"
                             "or fixed parameters" );
          }
        if ( haveParameters )
          {
          transform->SetFixedParameters ( TmpFixedParameterArray );
          niftkitkDebugMacro ("Set Transform Fixed Parameters" );
          transform->SetParametersByValue ( TmpParameterArray );
          niftkitkDebugMacro ("Set Transform Parameters" );
          TmpParameterArray.clear();
          TmpFixedParameterArray.clear(); 
          haveFixedParameters = false;
          haveParameters = false;
          }
        else
          {
          haveFixedParameters = true;
          }
        }
      }
    }
}

void 
NIFTKTransformIO::
Write()
{
  ConstTransformListType::iterator it = this->GetWriteTransformList().begin();
  vnl_vector<double> TempArray;
  std::ofstream out;
  this->OpenStream(out,false);

  out << "#Insight Transform File V1.0" << std::endl;
  int count = 0;
  while(it != this->GetWriteTransformList().end())
    {
    out << "# Transform " << count << std::endl;
    out << "Transform: " << (*it)->GetTransformTypeAsString() << std::endl;

    TempArray = (*it)->GetParameters();
    out << "Parameters: " << TempArray << std::endl;
    TempArray = (*it)->GetFixedParameters();
    out << "FixedParameters: " << TempArray << std::endl;
    it++;
    count++;
    }
  out.close();
}

}
