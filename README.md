# fiddle: four-chambered heart + IBAMR + deal.II library

Some experiments in trying to implement something like `IBAMR::IBFEMethod` and
`IBTK::FEDataManager` within deal.II rather than libMesh.

# Building fiddle

1. Build IBAMR with CMake - see

https://github.com/IBAMR/IBAMR/blob/master/doc/cmake.md

2. Build deal.II (the current development version)

3. Run some shell commands like
```
mkdir build
cd build
cmake -DDEAL_II_ROOT=$HOME/Applications/deal.II \
      -DIBAMR_ROOT=$HOME/Applications/ibamr \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-g -DDEBUG -Wall -Wextra -Wpedantic -fopenmp"   \
      ../
```
for a debug build (uses deal.II's debug settings) or
```
mkdir build-release
cd build-release
cmake -DDEAL_II_ROOT=$HOME/Applications/deal.II \
      -DIBAMR_ROOT=$HOME/Applications/ibamr \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native -g"   \
      ../
```
for a release build. In both cases you need to signal to fiddle where IBAMR and
deal.II were installed via `-DDEAL_II_ROOT` and `-DIBAMR_ROOT`.

Fiddle doesn't do much checking - if you use different MPI versions in deal.II
and IBAMR, for example, fiddle won't catch it.

You can either use fiddle in-place or run `make install` to use it as a
dependency. As usual, you will need to specify `CMAKE_INSTALL_PREFIX` to install
to a non-default location (e.g., inside your home directory).

fiddle uses IBAMR's timer infrastructure. To achieve more accurate timings,
fiddle optionally (by default this is enabled) turns on MPI barriers between
sections to explicitly measure the amount of time spent waiting on something
else to finish. This is a compile-time option provided to CMake with
`-DFDL_ENABLE_TIMER_BARRIERS=ON` (default) or `-DFDL_ENABLE_TIMER_BARRIERS=OFF`.

# Project Goals

- Scalable implementations of all fundamental IFED algorithms.
- Simple to understand internal classes which can be composed in a variety of
  ways for different applications.
- `fdl::IFEDMethod` is a complete implementation of an `IBAMR::IBStrategy`
  object which performs either elemental or nodal coupling between a
  Lagrangian hyperelastic solid and Eulerian grid.
- lots of useful utilities, like meter meshes.
- (WIP) examples.

# Style Guide

## General Advice

1. "The lost art of structured programming": code is built recursively out of
   other code. Expose this structure as much as possible. A thousand-line
   function is basically impossible to understand or debug (code complexity
   scales nonlinearly with line count). Classes should be immutable for the most
   part aside from a reinit function (and maybe mutable position/velocity
   vectors). No staggered initialization: use RAII. Clear ownership semantics:
   e.g., users create `Part`s and `std::move()` them to signify that
   `fdl::IFEDMethod` now owns them.
2. We want to build ten classes inheriting from `IBAMR::IBStrategy`: Presently
   `fdl::IFEDMethod` is just an interface that adapts other pieces to work with
   IBAMR. It doesn't much past get the pieces talking to each-other.
3. No loops over elements in classes: these should always be in utility
   functions. This enforces a clean design and separation of concerns - e.g.,
   force spreading doesn't need anything besides the patch hierarchy, patch map,
   mapping, and FE vectors.
4. Avoid SAMRAI - there are a lot of bugs and poor design decisions in SAMRAI
   (some examples: only one visit data writer can be created at once, several
   classes must be registered for restarts or the program will crash, the string
   processing doesn't support binary data, lack of `const`, `PatchHierarchy`
   does way too many things (it shouldn't be possible for a function that needs
   to spread force to also be able to regrid the entire hierarchy),
   `RestartManager` is a pain (there is no way to write a single checkpoint file
   with both libMesh and SAMRAI since they are utterly inflexible in
   serialization)). Classes that store data should use `boost::archive` to
   serialize it to an output stream: yes, this is slower, but it also lets us
   write the data anywhere. Only classes that hook directly into IBAMR (like
   those inheriting from `IBStrategy`) should mess with `RestartManager`.
   Singletons and other global state make programs much more difficult to
   understand and impede interoperability.

## Naming Functions

1. When a function returns a currently available (i.e., no MPI communication or
   computation necessary) value: use `get_`.
2. When a function computes a subset of an existing thing: use `extract_`.
3. When a function combines a subset of an existing thing: use `combine_`.
4. When a function computes values: use `compute_`.
