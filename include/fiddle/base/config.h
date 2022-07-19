#ifndef included_fiddle_config_h
#define included_fiddle_config_h

#  define FDL_DISABLE_EXTRA_DIAGNOSTICS                              \
_Pragma("GCC diagnostic push")                                 /*!*/ \
_Pragma("GCC diagnostic ignored \"-Wunknown-pragmas\"")        /*!*/ \
_Pragma("GCC diagnostic ignored \"-Wpragmas\"")                /*!*/ \
_Pragma("GCC diagnostic ignored \"-Wunknown-warning-option\"") /*!*/ \
_Pragma("GCC diagnostic ignored \"-Wunknown-warning\"")        /*!*/ \
_Pragma("GCC diagnostic ignored \"-Wextra\"")                  /*!*/ \
_Pragma("GCC diagnostic ignored \"-Wclass-memaccess\"")              \
_Pragma("GCC diagnostic warning \"-Wpragmas\"")

#  define FDL_ENABLE_EXTRA_DIAGNOSTICS                               \
_Pragma("GCC diagnostic pop")

#endif
