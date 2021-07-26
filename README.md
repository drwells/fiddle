# fiddle: four-chambered heart + IBAMR + deal.II library

Some experiments in trying to implement something like `IBAMR::IBFEMethod` and
`IBTK::FEDataManager` within deal.II rather than libMesh.

# Project Goals

- (WIP) Scalable implementations of all fundamental IFED algorithms.
- Simple to understand internal classes which can be composed in a variety of
  ways for different applications.
- (WIP) `fdl::IFEDMethod` is a complete implementation of an `IBAMR::IBStrategy`
  object which performs either elemental or nodal coupling between a
  Lagrangian hyperelastic solid and Eulerian grid.

# Style Guide
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
4. Avoid SAMRAI - there are a lot of bugs and poor design decisions in samrai
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
