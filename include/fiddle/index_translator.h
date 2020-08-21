#ifndef included_fiddle_index_translator_h
#define included_fiddle_index_translator_h

#include <deal.II/base/index_set.h>
#include <deal.II/base/types.h>

#include <algorithm>
#include <numerics>
#include <utility>
#include <vector>

namespace fdl
{
using namespace dealii;

// convenience class for converting between a local and global set of numbers.
class IndexTranslator
{
  public:
    IndexTranslator(const std::vector<types::global_dof_index> &consecutive_dofs,
                    const std::vector<types::global_dof_index> &nonconsecutive_dofs)
      : nonconsecutive_to_internal_index(nonconsecutive_dofs.size() == 0 ? 0  :
                                         *std::max_element(nonconsecutive_dofs.begin(),
                                                           nonconsecutive_dofs.end()))
    {
      Assert(consecutive_dofs.size() == nonconsecutive_dofs.size(),
             ExcMessage("should have same length"));

      // Set up the mapping from consecutive dofs to nonconsecutive dofs.
      std::vector<std::pair<types::global_dof_index, types::global_dof_index>> temp;
      for (std::size_t i = 0; i < consecutive_dofs.size(); ++i)
        temp.emplace_back(consecutive_dofs[i], nonconsecutive_dofs[i]);
      std::sort(temp.begin(), temp.end(),
                [&](const auto &a, const auto &b)
                {
                  return a.first < b.first;
                });
      for (const auto &pair : temp) consecutive_to_non.push_back(pair.second);

      // Set up the mapping from nonconsecutive to consecutive dofs.
      std::sort(temp.begin(), temp.end(),
                [](const auto &a, const auto &b)
                {
                  return a.second < b.second;
                });
      for (const auto &dof : nonconsecutive_dofs)
        nonconsecutive_to_internal_index.add_index(dof);

      nonconsecutive_to_internal_index.compress();
      for (const auto &index : nonconsecutive_to_internal_index)
      {
        // Find the consecutive dof corresponding to the current nonconsecutive
        // dof.
        const auto it = std::lower_bound(
          temp.begin(), temp.end(), index,
          [](const std::pair<types::global_dof_index, types::global_dof_index> &a,
             const types::global_dof_index &b)
        {
          return a.second < b;
        });
        // push back the entry for faster access in non_to_con:
        internal_to_consecutive.push_back(it->first);
      }
    }

    types::global_dof_index
    con_to_non(const types::global_dof_index a)
    {
      Assert(a < consecutive_to_non.size(), ExcMessage("a too big"));
      return consecutive_to_non[a];
    }


    types::global_dof_index
    non_to_con(const types::global_dof_index b)
    {
      return internal_to_consecutive[nonconsecutive_to_internal_index.index_within_set(b)];
    }


  protected:
    std::vector<types::global_dof_index> consecutive_to_non;

    IndexSet nonconsecutive_to_internal_index;
    std::vector<types::global_dof_index> internal_to_consecutive;
};
}

#endif // define included_fiddle_index_translator_h
