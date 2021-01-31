#include <fstream>

// Basic test verifying the test suite is running correctly

int
main()
{
  std::ofstream out("output");
  out << "hello, world\n";
}
