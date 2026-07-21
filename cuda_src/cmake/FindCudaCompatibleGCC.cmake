#[=======================================================================[
FindCudaCompatibleGCC
----------------------

Scans the system for installed GCC/G++ versions and selects one that
falls within a specified supported range for use as the CUDA host
compiler (CMAKE_CUDA_HOST_COMPILER).

This must be include()'d BEFORE enable_language(CUDA) or
project(... CUDA), since CMAKE_CUDA_HOST_COMPILER is only read at that
point.

Usage
^^^^^

    set(CUDA_GCC_MIN 6)
    set(CUDA_GCC_MAX 15)      # e.g. CUDA 13.1 / 13.3 supported range
    include(FindCudaCompatibleGCC)

    project(myproject LANGUAGES CXX CUDA)

Result Variables
^^^^^^^^^^^^^^^^^

  CUDA_HOST_GXX_FOUND      - TRUE if a suitable g++ was found
  CUDA_HOST_GXX_PATH       - Full path to the selected g++
  CUDA_HOST_GXX_VERSION    - Major version number of the selected g++
  CMAKE_CUDA_HOST_COMPILER - Set automatically if a match is found

Cache Variables
^^^^^^^^^^^^^^^

  ALL_FOUND_GCC_VERSIONS   - List of all GCC major versions detected,
                              whether or not they were CUDA-compatible

Options
^^^^^^^

  CUDA_GCC_MIN   - Minimum acceptable GCC major version (default: 6)
  CUDA_GCC_MAX   - Maximum acceptable GCC major version (default: 15)
  CUDA_GCC_SEARCH_PATHS - Extra directories to search, beyond PATH
                          (default: common Fedora/RHEL compat locations)

#]=======================================================================]

# ---- Configurable defaults -------------------------------------------

if(NOT DEFINED CUDA_GCC_MIN)
    set(CUDA_GCC_MIN 6)
endif()

if(NOT DEFINED CUDA_GCC_MAX)
    set(CUDA_GCC_MAX 15)
endif()

if(NOT DEFINED CUDA_GCC_SEARCH_PATHS)
    set(CUDA_GCC_SEARCH_PATHS
        /usr/bin
        /usr/local/bin
        /opt/rh/gcc-toolset-${CUDA_GCC_MAX}/root/usr/bin   # RHEL/CentOS toolsets
    )
endif()

# ---- Step 1: Enumerate every gcc-N / g++-N on the system --------------
#
# We probe a generous version range rather than relying on a package
# manager query, since there's no portable CMake way to ask apt/dnf/etc.
# what's installed. This covers Debian/Ubuntu (gcc-13, g++-13, ...),
# Fedora compat packages (gcc13-c++ installs as /usr/bin/gcc-13), and
# RHEL devtoolsets/gcc-toolsets.

set(_GCC_PROBE_RANGE 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4)
set(ALL_FOUND_GCC_VERSIONS "" CACHE STRING "GCC major versions detected on this system" FORCE)

foreach(_ver ${_GCC_PROBE_RANGE})
    find_program(_GCC_${_ver}_C_PATH   gcc-${_ver} PATHS ${CUDA_GCC_SEARCH_PATHS})
    find_program(_GCC_${_ver}_CXX_PATH g++-${_ver} PATHS ${CUDA_GCC_SEARCH_PATHS})

    if(_GCC_${_ver}_CXX_PATH)
        list(APPEND ALL_FOUND_GCC_VERSIONS ${_ver})
        message(STATUS "FindCudaCompatibleGCC: found g++-${_ver} at ${_GCC_${_ver}_CXX_PATH}")
    endif()

    # Keep these out of the normal find_program cache clutter on rescans
    mark_as_advanced(_GCC_${_ver}_C_PATH _GCC_${_ver}_CXX_PATH)
endforeach()

# Also check the unversioned default `gcc`/`g++` on PATH, in case it's
# already within range (common on RHEL/Rocky where "gcc" IS gcc 11, etc.)
find_program(_GCC_DEFAULT_CXX_PATH g++)
if(_GCC_DEFAULT_CXX_PATH)
    execute_process(
        COMMAND ${_GCC_DEFAULT_CXX_PATH} -dumpversion
        OUTPUT_VARIABLE _GCC_DEFAULT_FULL_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "^[0-9]+" _GCC_DEFAULT_MAJOR "${_GCC_DEFAULT_FULL_VERSION}")
    if(_GCC_DEFAULT_MAJOR)
        list(APPEND ALL_FOUND_GCC_VERSIONS ${_GCC_DEFAULT_MAJOR})
        set(_GCC_${_GCC_DEFAULT_MAJOR}_CXX_PATH ${_GCC_DEFAULT_CXX_PATH})
        message(STATUS "FindCudaCompatibleGCC: default g++ is version ${_GCC_DEFAULT_MAJOR} at ${_GCC_DEFAULT_CXX_PATH}")
    endif()
endif()

if(ALL_FOUND_GCC_VERSIONS)
    list(REMOVE_DUPLICATES ALL_FOUND_GCC_VERSIONS)
    list(SORT ALL_FOUND_GCC_VERSIONS COMPARE NATURAL ORDER DESCENDING)
endif()
set(ALL_FOUND_GCC_VERSIONS ${ALL_FOUND_GCC_VERSIONS} CACHE STRING "GCC major versions detected on this system" FORCE)

if(NOT ALL_FOUND_GCC_VERSIONS)
    message(WARNING "FindCudaCompatibleGCC: no GCC installations found on PATH or in CUDA_GCC_SEARCH_PATHS")
endif()

# ---- Step 2: Pick the newest version within [CUDA_GCC_MIN, CUDA_GCC_MAX] ----

set(CUDA_HOST_GXX_FOUND FALSE)
set(CUDA_HOST_GXX_PATH "")
set(CUDA_HOST_GXX_VERSION "")

foreach(_ver ${ALL_FOUND_GCC_VERSIONS})
    if(_ver GREATER_EQUAL CUDA_GCC_MIN AND _ver LESS_EQUAL CUDA_GCC_MAX)
        set(CUDA_HOST_GXX_FOUND TRUE)
        set(CUDA_HOST_GXX_PATH "${_GCC_${_ver}_CXX_PATH}")
        set(CUDA_HOST_GXX_VERSION "${_ver}")
        break()
    endif()
endforeach()

# ---- Step 3: Report and configure -------------------------------------

if(CUDA_HOST_GXX_FOUND)
    message(STATUS "FindCudaCompatibleGCC: selected g++-${CUDA_HOST_GXX_VERSION} "
                    "(${CUDA_HOST_GXX_PATH}) as CUDA host compiler "
                    "[supported range: ${CUDA_GCC_MIN}-${CUDA_GCC_MAX}]")
    set(CMAKE_CUDA_HOST_COMPILER "${CUDA_HOST_GXX_PATH}" CACHE FILEPATH "CUDA host compiler" FORCE)
else()
    message(WARNING
        "FindCudaCompatibleGCC: no installed GCC falls within the supported "
        "range [${CUDA_GCC_MIN}-${CUDA_GCC_MAX}]. Detected versions: "
        "${ALL_FOUND_GCC_VERSIONS}. "
        "CMAKE_CUDA_HOST_COMPILER has NOT been set -- CUDA compilation may "
        "fail or require --allow-unsupported-compiler. "
        "On Debian/Ubuntu: apt install g++-${CUDA_GCC_MAX}. "
        "On Fedora: dnf install gcc${CUDA_GCC_MAX}-c++."
    )
endif()