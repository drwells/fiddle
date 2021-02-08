#ifndef included_fiddle_exceptions_h
#define included_fiddle_exceptions_h

#include <deal.II/base/exceptions.h>

namespace fdl
{
  DeclExceptionMsg(ExcFDLInternalError,
                   "The program entered a state which was not anticipated "
                   "and will now abort. This an internal error.");

  DeclExceptionMsg(ExcFDLNotImplemented,
                   "The requested feature is not yet implemented.");
}

#endif
