/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkFilenameUtils_cxx
#define __itkFilenameUtils_cxx

#include "itkFilenameUtils.h"

#include <iostream>
#include <string>


namespace itk
{


/* ----------------------------------------------------------------------
   ModifyFilenameSuffix()
   ---------------------------------------------------------------------- */

std::string ModifyFilenameSuffix( std::string filename, std::string suffix ) 
{
  std::string tmpname, basename, extname;
  std::string compression;
  std::string::size_type idx;
  std::string::size_type idxZ = std::string::npos;
  std::string::size_type idxGZ = std::string::npos;
  std::string::size_type lenExt = std::string::npos;

  // Search for the last directory separator: '/'

#ifdef MS_VISUAL_Cpp
  std::string::size_type diridx = filename.find_last_of('\\');
#else
  std::string::size_type diridx = filename.find_last_of('/');
#endif

  if (diridx == std::string::npos)
  {  
    diridx = 0;
  }
  // Search for period in file name

  idx = filename.rfind('.');

  if ((idx == std::string::npos) || (idx < diridx)) 
  {
    // file name does not contain a period so add suffix
    tmpname = filename + suffix;
  }
  else 
  {
    // check that we haven't found a '.Z' or '.gz' suffix
    if (filename.compare(idx, 2, ".Z") == 0) 
    {
      idxZ = idx;
      idx = filename.rfind('.', idxZ-1);
      if ((idx == std::string::npos) || (idx < diridx)) 
      {
	      idx = idxZ;
	      lenExt = filename.length() - idx;
      }
      else 
      {
	      lenExt = idxZ - idx - 1;
	      compression = ".Z";
      }
    }
    else if (filename.compare(idx, 3, ".gz") == 0) 
    {
      idxGZ = idx;
      idx = filename.rfind('.', idxGZ-1);
      if ((idx == std::string::npos) || (idx < diridx)) 
      {
	      idx = idxGZ;
	      lenExt = filename.length() - idx;
      }
      else 
      {
	      lenExt = idxGZ - idx - 1;
	      compression = ".gz";
      }
    }
    else 
    {
      lenExt = filename.length() - idx;
    }

    // split file name into stem and extension
    basename = filename.substr(0, idx);
    extname = filename.substr(idx + 1, lenExt);

    if (extname.empty()) 
    {
      // contains period but no extension
      tmpname = filename + suffix + compression;
    }
    else 
    {
      // replace extension
      tmpname = filename;
      tmpname.replace(idx, lenExt+1, suffix);
    }
  }

  return tmpname;
}


/* ----------------------------------------------------------------------
   ExtractSuffix
   ---------------------------------------------------------------------- */

std::string ExtractSuffix( std::string filename ) 
{
  std::string tmpname, basename, extname;
  std::string::size_type idx;
  std::string::size_type idxZ = std::string::npos;
  std::string::size_type idxGZ = std::string::npos;

  // Search for the last directory separator: '/'

#ifdef MS_VISUAL_Cpp
  std::string::size_type diridx = filename.find_last_of('\\');
#else
  std::string::size_type diridx = filename.find_last_of('/');
#endif

  if (diridx == std::string::npos)
  {  
    diridx = 0;
  }
  // Search for period in file name

  idx = filename.rfind('.');

  // file name does not contain a period so return empty string
  if ((idx == std::string::npos) || (idx < diridx)) 
  {
    extname = "";
  }
  else 
  {
    // check that we haven't found a '.Z' or '.gz' suffix
    if (filename.compare(idx, 2, ".Z") == 0) 
    {
      idxZ = idx;
      idx = filename.rfind('.', idxZ-1);
      if ((idx == std::string::npos) || (idx < diridx)) 
      {
        extname = "";
      }
      else 
      {
	      // split file name into stem and extension
	      extname = filename.substr(idx + 1, idxZ - idx - 1);

        if (extname.empty()) 
        {
	        // contains period but no extension
          extname = "";
	      }
	    }
    }
    else if (filename.compare(idx, 3, ".gz") == 0) 
    {
      idxGZ = idx;
      idx = filename.rfind('.', idxGZ-1);
      if ((idx == std::string::npos) || (idx < diridx)) 
      {
        extname = "";
      }
      else 
      {
	      // split file name into stem and extension
	      extname = filename.substr(idx + 1, idxGZ - idx - 1);

	      if (extname.empty()) 
        {
	        // contains period but no extension
          extname = "";
	      }
      }
    }
    else 
    {
      // split file name into stem and extension
      extname = filename.substr(idx + 1);
      
      if (extname.empty()) 
      {
	      // contains period but no extension
        extname = "";
      }
    }
  }

  return extname;
}

}

#endif
