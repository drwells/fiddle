#include <fiddle/base/exceptions.h>
#include <fiddle/base/utilities.h>

#include <fstream>

void
test_printable(const std::string &input, std::ofstream &out)
{
  const std::string base64 =
    fdl::encode_base64(input.c_str(), input.c_str() + input.size());
  const std::string output =
    fdl::decode_base64(base64.c_str(), base64.c_str() + base64.size());
  // we messed up if they are not the same size
  Assert(input.size() == output.size(), fdl::ExcFDLInternalError());

  out << "start:  " << input << '\n';
  out << "base64: " << base64 << '\n';
  out << "output: " << output << '\n';
}

void
test_not_printable(const std::vector<int> &input, std::ofstream &out)
{
  const int *       begin = input.data();
  const int *       end   = input.data() + input.size();
  const std::string base64 =
    fdl::encode_base64(reinterpret_cast<const char *>(begin),
                       reinterpret_cast<const char *>(end));
  const std::string output =
    fdl::decode_base64(base64.c_str(), base64.c_str() + base64.size());
  // we messed up if they are not the same size
  Assert(input.size() == output.size() / sizeof(int),
         fdl::ExcFDLInternalError());
  // We have to be careful in converting back in case output is not aligned
  std::vector<int> output2;
  for (unsigned int i = 0; i < input.size(); ++i)
    {
      alignas(int) char buf[sizeof(int)];
      for (unsigned int j = 0; j < sizeof(int); ++j)
        buf[j] = output[sizeof(int) * i + j];
      output2.push_back(*reinterpret_cast<const int *>(buf));
    }

  out << "start:  ";
  for (unsigned int i = 0; i < input.size(); ++i)
    out << input[i] << ',';
  out << '\n';
  out << "base64: " << base64 << '\n';
  out << "output: ";
  for (unsigned int i = 0; i < input.size(); ++i)
    out << output2[i] << ',';
  out << '\n';
}

int
main()
{
  std::ofstream out("output");
  test_printable("", out);
  test_printable("l", out);
  test_printable("lo", out);
  test_printable("lor", out);
  test_printable("lore", out);
  test_printable("lorem", out);
  test_printable("lorem ", out);
  test_printable("lorem i", out);
  test_printable("lorem ip", out);
  test_printable("lorem ips", out);

  test_not_printable({}, out);
  test_not_printable({0}, out);
  test_not_printable({0, 1}, out);
  test_not_printable({0, 1, 2}, out);
  test_not_printable({0, 1, 2, 3}, out);
  test_not_printable({0, 1, 2, 3, 4}, out);
  test_not_printable({0, 1, 2, 3, 4, 5}, out);
  test_not_printable({0, 1, 2, 3, 4, 5, 6}, out);
}
