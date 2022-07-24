#include <fiddle/grid/grid_utilities.h>

#include <fstream>

int main()
{
  using namespace dealii;

  const std::string test_file = SOURCE_DIR "/two-nodesets.e";

  auto pair_42 = fdl::extract_nodeset<3>(test_file, 42);
  auto pair_100 = fdl::extract_nodeset<3>(test_file, 100);

  std::ofstream output("output");

  output << "nodeset 42 points =\n";
  for (const auto &point : pair_42.second)
    output << point << '\n';

  output << "nodeset 100 points =\n";
  for (const auto &point : pair_100.second)
    output << point << '\n';
}
