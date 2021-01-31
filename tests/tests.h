#include <Box.h>
#include <PatchHierarchy.h>

#include <tbox/SAMRAI_MPI.h>

#include <mpi.h>

#include <fstream>
#include <sstream>
#include <string>

// A utility function that prints @p part_str to @p out by sending each string to
// rank 0.
inline void
print_strings_on_0(const std::string& part_str, std::ofstream &out)
{
    using namespace SAMRAI::tbox;
    const int n_nodes = SAMRAI_MPI::getNodes();
    std::vector<unsigned long> string_sizes(n_nodes);

    const unsigned long size = part_str.size();
    int ierr = MPI_Gather(
        &size, 1, MPI_UNSIGNED_LONG, string_sizes.data(), 1, MPI_UNSIGNED_LONG, 0,
        SAMRAI_MPI::getCommunicator());
    TBOX_ASSERT(ierr == 0);

    // MPI_Gatherv would be more efficient, but this just a test so its
    // not too important
    if (SAMRAI_MPI::getRank() == 0)
    {
        out << part_str;
        for (int r = 1; r < n_nodes; ++r)
        {
            std::string input;
            input.resize(string_sizes[r]);
            ierr = MPI_Recv(&input[0], string_sizes[r], MPI_CHAR, r, 0, SAMRAI_MPI::getCommunicator(), MPI_STATUS_IGNORE);
            TBOX_ASSERT(ierr == 0);
            out << input;
        }
    }
    else
        MPI_Send(part_str.data(), size, MPI_CHAR, 0, 0, SAMRAI_MPI::getCommunicator());
}

/**
 * Print the parallel partitioning (i.e., the boxes) on all processes to @p out
 * on processor 0.
 */
template <int spacedim>
inline void
print_partitioning_on_0(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<spacedim> >& patch_hierarchy,
                        const int coarsest_ln,
                        const int finest_ln,
                        std::ofstream &out)
{
    using namespace SAMRAI;

    std::ostringstream part_steam;
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        tbox::Pointer<hier::PatchLevel<spacedim> > patch_level
            = patch_hierarchy->getPatchLevel(ln);
        part_steam << "rank: " << tbox::SAMRAI_MPI::getRank()
                   << " level: " << ln << " boxes:\n";
        for (typename hier::PatchLevel<spacedim>::Iterator p(patch_level); p; p++)
        {
            const hier::Box<spacedim> box = patch_level->getPatch(p())->getBox();
            part_steam << box << '\n';
        }
    }
    print_strings_on_0(part_steam.str(), out);
}
