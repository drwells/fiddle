#include <fiddle/base/exceptions.h>
#include <fiddle/base/utilities.h>

#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/transform_width.hpp>

#include <array>

namespace fdl
{
  std::string
  encode_base64(const char *begin, const char *end)
  {
    using namespace boost::archive::iterators;
    using iterator = base64_from_binary<transform_width<const char *, 6, 8>>;
    std::string base64{iterator(begin), iterator(end)};
    // Add padding.
    std::array<std::string, 3> paddings{"", "==", "="};
    base64.append(paddings[(end - begin) % 3]);

    return base64;
  }

  std::string
  decode_base64(const char *begin, const char *end)
  {
    using namespace boost::archive::iterators;
    using iterator = transform_width<binary_from_base64<const char *>, 8, 6>;
    std::string binary{iterator(begin), iterator(end)};

    // We have three possibilities for padding, based on how boost decodes it:
    // input ends in "==": remove two NULs at the end
    // input ends in "=": remove one NUL at the end
    // otherwise: no padding, nothing to remove
    if (begin != end)
      {
        const char *input = end - 1;
        while (input >= begin && *input == '=')
          {
            binary.pop_back();
            --input;
          }
      }
    return binary;
  }
} // namespace fdl
