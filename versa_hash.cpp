// milter
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <cstring>
#include <stdexcept>

#include <boost/asio.hpp>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <secp256k1.h>
#include <secp256k1_schnorr.h>
#include <omp.h>

#include <cassert>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace dev
{
/**
 * A modifiable reference to an existing object or vector in memory.
 */
template <class _T>
class vector_ref
{
public:
    using value_type = _T;
    using element_type = _T;
    using mutable_value_type = typename std::conditional<std::is_const<_T>::value,
        typename std::remove_const<_T>::type, _T>::type;

    static_assert(std::is_pod<value_type>::value,
        "vector_ref can only be used with PODs due to its low-level treatment of data.");

    vector_ref() : m_data(nullptr), m_count(0) {}
    /// Creates a new vector_ref to point to @a _count elements starting at @a _data.
    vector_ref(_T* _data, size_t _count) : m_data(_data), m_count(_count) {}
    /// Creates a new vector_ref pointing to the data part of a string (given as pointer).
    vector_ref(
        typename std::conditional<std::is_const<_T>::value, std::string const*, std::string*>::type
            _data)
      : m_data(reinterpret_cast<_T*>(_data->data())), m_count(_data->size() / sizeof(_T))
    {}
    /// Creates a new vector_ref pointing to the data part of a vector (given as pointer).
    vector_ref(typename std::conditional<std::is_const<_T>::value,
        std::vector<typename std::remove_const<_T>::type> const*, std::vector<_T>*>::type _data)
      : m_data(_data->data()), m_count(_data->size())
    {}
    /// Creates a new vector_ref pointing to the data part of a string (given as reference).
    vector_ref(
        typename std::conditional<std::is_const<_T>::value, std::string const&, std::string&>::type
            _data)
      : m_data(reinterpret_cast<_T*>(_data.data())), m_count(_data.size() / sizeof(_T))
    {}
#if DEV_LDB
    vector_ref(ldb::Slice const& _s)
      : m_data(reinterpret_cast<_T*>(_s.data())), m_count(_s.size() / sizeof(_T))
    {}
#endif
    explicit operator bool() const { return m_data && m_count; }

    bool contentsEqual(std::vector<mutable_value_type> const& _c) const
    {
        if (!m_data || m_count == 0)
            return _c.empty();
        return _c.size() == m_count && !memcmp(_c.data(), m_data, m_count * sizeof(_T));
    }
    std::vector<mutable_value_type> toVector() const
    {
        return std::vector<mutable_value_type>(m_data, m_data + m_count);
    }
    std::vector<unsigned char> toBytes() const
    {
        return std::vector<unsigned char>(reinterpret_cast<unsigned char const*>(m_data),
            reinterpret_cast<unsigned char const*>(m_data) + m_count * sizeof(_T));
    }
    std::string toString() const
    {
        return std::string((char const*)m_data, ((char const*)m_data) + m_count * sizeof(_T));
    }

    template <class _T2>
    explicit operator vector_ref<_T2>() const
    {
        assert(m_count * sizeof(_T) / sizeof(_T2) * sizeof(_T2) / sizeof(_T) == m_count);
        return vector_ref<_T2>(reinterpret_cast<_T2*>(m_data), m_count * sizeof(_T) / sizeof(_T2));
    }
    operator vector_ref<_T const>() const { return vector_ref<_T const>(m_data, m_count); }

    _T* data() const { return m_data; }
    /// @returns the number of elements referenced (not necessarily number of bytes).
    size_t count() const { return m_count; }
    /// @returns the number of elements referenced (not necessarily number of bytes).
    size_t size() const { return m_count; }
    bool empty() const { return !m_count; }
    /// @returns a new vector_ref pointing at the next chunk of @a size() elements.
    vector_ref<_T> next() const
    {
        if (!m_data)
            return *this;
        return vector_ref<_T>(m_data + m_count, m_count);
    }
    /// @returns a new vector_ref which is a shifted and shortened view of the original data.
    /// If this goes out of bounds in any way, returns an empty vector_ref.
    /// If @a _count is ~size_t(0), extends the view to the end of the data.
    vector_ref<_T> cropped(size_t _begin, size_t _count) const
    {
        if (m_data && _begin <= m_count && _count <= m_count && _begin + _count <= m_count)
            return vector_ref<_T>(
                m_data + _begin, _count == ~size_t(0) ? m_count - _begin : _count);
        return {};
    }
    /// @returns a new vector_ref which is a shifted view of the original data (not going beyond
    /// it).
    vector_ref<_T> cropped(size_t _begin) const
    {
        if (m_data && _begin <= m_count)
            return vector_ref<_T>(m_data + _begin, m_count - _begin);
        return {};
    }
    void retarget(_T* _d, size_t _s)
    {
        m_data = _d;
        m_count = _s;
    }
    void retarget(std::vector<_T> const& _t)
    {
        m_data = _t.data();
        m_count = _t.size();
    }
    template <class T>
    bool overlapsWith(vector_ref<T> _t) const
    {
        void const* f1 = data();
        void const* t1 = data() + size();
        void const* f2 = _t.data();
        void const* t2 = _t.data() + _t.size();
        return f1 < t2 && t1 > f2;
    }
    /// Copies the contents of this vector_ref to the contents of @a _t, up to the max size of @a
    /// _t.
    void copyTo(vector_ref<typename std::remove_const<_T>::type> _t) const
    {
        if (overlapsWith(_t))
            memmove(_t.data(), m_data, std::min(_t.size(), m_count) * sizeof(_T));
        else
            memcpy(_t.data(), m_data, std::min(_t.size(), m_count) * sizeof(_T));
    }
    /// Copies the contents of this vector_ref to the contents of @a _t, and zeros further trailing
    /// elements in @a _t.
    void populate(vector_ref<typename std::remove_const<_T>::type> _t) const
    {
        copyTo(_t);
        memset(_t.data() + m_count, 0, std::max(_t.size(), m_count) - m_count);
    }
    /// Securely overwrite the memory.
    /// @note adapted from OpenSSL's implementation.
    void cleanse()
    {
        static unsigned char s_cleanseCounter = 0;
        auto* p = (uint8_t*)begin();
        size_t const len = (uint8_t*)end() - p;
        size_t loop = len;
        size_t count = s_cleanseCounter;
        while (loop--)
        {
            *(p++) = (uint8_t)count;
            count += (17 + ((size_t)p & 0xf));
        }
        p = (uint8_t*)memchr((uint8_t*)begin(), (uint8_t)count, len);
        if (p)
            count += (63 + (size_t)p);
        s_cleanseCounter = (uint8_t)count;
        memset((uint8_t*)begin(), 0, len);
    }

    _T* begin() { return m_data; }
    _T* end() { return m_data + m_count; }
    _T const* begin() const { return m_data; }
    _T const* end() const { return m_data + m_count; }

    _T& operator[](size_t _i)
    {
        assert(m_data);
        assert(_i < m_count);
        return m_data[_i];
    }
    _T const& operator[](size_t _i) const
    {
        assert(m_data);
        assert(_i < m_count);
        return m_data[_i];
    }

    bool operator==(vector_ref<_T> const& _cmp) const
    {
        return m_data == _cmp.m_data && m_count == _cmp.m_count;
    }
    bool operator!=(vector_ref<_T> const& _cmp) const { return !operator==(_cmp); }

    void reset()
    {
        m_data = nullptr;
        m_count = 0;
    }

private:
    _T* m_data;
    size_t m_count;
};

template <class _T>
vector_ref<_T const> ref(_T const& _t)
{
    return vector_ref<_T const>(&_t, 1);
}
template <class _T>
vector_ref<_T> ref(_T& _t)
{
    return vector_ref<_T>(&_t, 1);
}
template <class _T>
vector_ref<_T const> ref(std::vector<_T> const& _t)
{
    return vector_ref<_T const>(&_t);
}
template <class _T>
vector_ref<_T> ref(std::vector<_T>& _t)
{
    return vector_ref<_T>(&_t);
}

}  // namespace dev

#include <string>
#include <vector>

#include <boost/multiprecision/cpp_int.hpp>

using byte = uint8_t;

namespace dev
{
// Binary data types.
using bytes = std::vector<byte>;
using bytesRef = vector_ref<byte>;
using bytesConstRef = vector_ref<byte const>;

// Numeric types.
using bigint = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<>>;
using u64 = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<64, 64,
    boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using u128 = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<128, 128,
    boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using u256 = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<256, 256,
    boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using u160 = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<160, 160,
    boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using u512 = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<512, 512,
    boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;

// Null/Invalid values for convenience.
static const u256 Invalid256 = ~(u256)0;

/// Converts arbitrary value to string representation using std::stringstream.
template <class _T>
std::string toString(_T const& _t)
{
    std::ostringstream o;
    o << _t;
    return o.str();
}

}  // namespace dev

#include <algorithm>
#include <cstring>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <boost/algorithm/string.hpp>

// #include "Common.h"

namespace dev
{
// String conversion functions, mainly to/from hex/nibble/byte representations.

enum class WhenError
{
    DontThrow = 0,
    Throw = 1,
};

enum class HexPrefix
{
    DontAdd = 0,
    Add = 1,
};

enum class ScaleSuffix
{
    DontAdd = 0,
    Add = 1
};

/// Convert a series of bytes to the corresponding string of hex duplets.
/// @param _w specifies the width of the first of the elements. Defaults to two - enough to
/// represent a byte.
/// @example toHex("A\x69") == "4169"
template <class T>
std::string toHex(T const& _data, int _w = 2, HexPrefix _prefix = HexPrefix::DontAdd)
{
    std::ostringstream ret;
    unsigned ii = 0;
    for (auto i : _data)
        ret << std::hex << std::setfill('0') << std::setw(ii++ ? 2 : _w)
            << (int)(typename std::make_unsigned<decltype(i)>::type)i;
    return (_prefix == HexPrefix::Add) ? "0x" + ret.str() : ret.str();
}

/// Converts a (printable) ASCII hex character into the correspnding integer value.
/// @example fromHex('A') == 10 && fromHex('f') == 15 && fromHex('5') == 5
int fromHex(char _i, WhenError _throw);

/// Converts a (printable) ASCII hex string into the corresponding byte stream.
/// @example fromHex("41626261") == asBytes("Abba")
/// If _throw = ThrowType::DontThrow, it replaces bad hex characters with 0's, otherwise it will
/// throw an exception.
bytes fromHex(std::string const& _s, WhenError _throw = WhenError::DontThrow);

/// Converts byte array to a string containing the same (binary) data. Unless
/// the byte array happens to contain ASCII data, this won't be printable.
inline std::string asString(bytes const& _b)
{
    return std::string((char const*)_b.data(), (char const*)(_b.data() + _b.size()));
}

/// Converts a string to a byte array containing the string's (byte) data.
inline bytes asBytes(std::string const& _b)
{
    return bytes((byte const*)_b.data(), (byte const*)(_b.data() + _b.size()));
}


// Big-endian to/from host endian conversion functions.

/// Converts a templated integer value to the big-endian byte-stream represented on a templated
/// collection. The size of the collection object will be unchanged. If it is too small, it will not
/// represent the value properly, if too big then the additional elements will be zeroed out.
/// @a Out will typically be either std::string or bytes.
/// @a T will typically by unsigned, u160, u256 or bigint.
template <class T, class Out>
inline void toBigEndian(T _val, Out& o_out)
{
    static_assert(std::is_same<bigint, T>::value || !std::numeric_limits<T>::is_signed,
        "only unsigned types or bigint supported");  // bigint does not carry sign bit on shift
    for (auto i = o_out.size(); i != 0; _val >>= 8, i--)
    {
        T v = _val & (T)0xff;
        o_out[i - 1] = (typename Out::value_type)(uint8_t)v;
    }
}

/// Converts a big-endian byte-stream represented on a templated collection to a templated integer
/// value.
/// @a _In will typically be either std::string or bytes.
/// @a T will typically by unsigned, u160, u256 or bigint.
template <class T, class _In>
inline T fromBigEndian(_In const& _bytes)
{
    T ret = (T)0;
    for (auto i : _bytes)
        ret =
            (T)((ret << 8) | (byte)(typename std::make_unsigned<typename _In::value_type>::type)i);
    return ret;
}

/// Convenience functions for toBigEndian
inline bytes toBigEndian(u256 _val)
{
    bytes ret(32);
    toBigEndian(std::move(_val), ret);
    return ret;
}
inline bytes toBigEndian(u160 _val)
{
    bytes ret(20);
    toBigEndian(_val, ret);
    return ret;
}

/// Convenience function for toBigEndian.
/// @returns a byte array just big enough to represent @a _val.
template <class T>
inline bytes toCompactBigEndian(T _val, unsigned _min = 0)
{
    static_assert(std::is_same<bigint, T>::value || !std::numeric_limits<T>::is_signed,
        "only unsigned types or bigint supported");  // bigint does not carry sign bit on shift
    int i = 0;
    for (T v = _val; v; ++i, v >>= 8)
    {
    }
    bytes ret(std::max<unsigned>(_min, i), 0);
    toBigEndian(_val, ret);
    return ret;
}

/// Convenience function for conversion of a u256 to hex
inline std::string toHex(u256 val, HexPrefix prefix = HexPrefix::DontAdd)
{
    std::string str = toHex(toBigEndian(val));
    return (prefix == HexPrefix::Add) ? "0x" + str : str;
}

inline std::string toHex(uint64_t _n, HexPrefix _prefix = HexPrefix::DontAdd, int _bytes = 16)
{
    // sizeof returns the number of bytes (not the number of bits)
    // thus if CHAR_BIT != 8 sizeof(uint64_t) will return != 8
    // Use fixed constant multiplier of 16
    std::ostringstream ret;
    ret << std::hex << std::setfill('0') << std::setw(_bytes) << _n;
    return (_prefix == HexPrefix::Add) ? "0x" + ret.str() : ret.str();
}

inline std::string toHex(uint32_t _n, HexPrefix _prefix = HexPrefix::DontAdd, int _bytes = 8)
{
    // sizeof returns the number of bytes (not the number of bits)
    // thus if CHAR_BIT != 8 sizeof(uint64_t) will return != 4
    // Use fixed constant multiplier of 8
    std::ostringstream ret;
    ret << std::hex << std::setfill('0') << std::setw(_bytes) << _n;
    return (_prefix == HexPrefix::Add) ? "0x" + ret.str() : ret.str();
}

inline std::string toCompactHex(uint64_t _n, HexPrefix _prefix = HexPrefix::DontAdd)
{
    std::ostringstream ret;
    ret << std::hex << _n;
    return (_prefix == HexPrefix::Add) ? "0x" + ret.str() : ret.str();
}

inline std::string toCompactHex(uint32_t _n, HexPrefix _prefix = HexPrefix::DontAdd)
{
    std::ostringstream ret;
    ret << std::hex << _n;
    return (_prefix == HexPrefix::Add) ? "0x" + ret.str() : ret.str();
}



// Algorithms for string and string-like collections.

/// Escapes a string into the C-string representation.
/// @p _all if true will escape all characters, not just the unprintable ones.
std::string escaped(std::string const& _s, bool _all = true);

// General datatype convenience functions.

/// Determine bytes required to encode the given integer value. @returns 0 if @a _i is zero.
template <class T>
inline unsigned bytesRequired(T _i)
{
    static_assert(std::is_same<bigint, T>::value || !std::numeric_limits<T>::is_signed,
        "only unsigned types or bigint supported");  // bigint does not carry sign bit on shift
    unsigned i = 0;
    for (; _i != 0; ++i, _i >>= 8)
    {
    }
    return i;
}

/// Sets environment variable.
///
/// Portable wrapper for setenv / _putenv C library functions.
bool setenv(const char name[], const char value[], bool override = false);

/// Gets a target hash from given difficulty
std::string getTargetFromDiff(double diff, HexPrefix _prefix = HexPrefix::Add);

/// Gets the difficulty expressed in hashes to target
double getHashesToTarget(std::string _target);

/// Generic function to scale a value
std::string getScaledSize(double _value, double _divisor, int _precision, std::string _sizes[],
    size_t _numsizes, ScaleSuffix _suffix = ScaleSuffix::Add);

/// Formats hashrate
std::string getFormattedHashes(double _hr, ScaleSuffix _suffix = ScaleSuffix::Add, int _precision = 2);

/// Formats hashrate
std::string getFormattedMemory(
    double _mem, ScaleSuffix _suffix = ScaleSuffix::Add, int _precision = 2);

/// Adjust string to a fixed length filling chars to the Left
std::string padLeft(std::string _value, size_t _length, char _fillChar);

/// Adjust string to a fixed length filling chars to the Right
std::string padRight(std::string _value, size_t _length, char _fillChar);

}  // namespace dev


std::string address = "";
std::string workerid = "";
using boost::asio::ip::tcp;
using json = nlohmann::json;

std::string getTargetFromDiff(double diff, dev::HexPrefix _prefix = dev::HexPrefix::Add)
{
    using namespace boost::multiprecision;
    using BigInteger = boost::multiprecision::cpp_int;

    static BigInteger base("0x00000000ffff0000000000000000000000000000000000000000000000000000");
    BigInteger product;

    if (diff == 0)
    {
        product = BigInteger("0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    }
    else
    {
        diff = 1 / diff;

        BigInteger idiff(diff);
        product = base * idiff;

        std::string sdiff = boost::lexical_cast<std::string>(diff);
        size_t ldiff = sdiff.length();
        size_t offset = sdiff.find(".");

        if (offset != std::string::npos)
        {
            // Number of decimal places
            size_t precision = (ldiff - 1) - offset;

            // Effective sequence of decimal places
            std::string decimals = sdiff.substr(offset + 1);

            // Strip leading zeroes. If a string begins with
            // 0 or 0x boost parser considers it hex
            decimals = decimals.erase(0, decimals.find_first_not_of('0'));

            // Build up the divisor as string - just in case
            // parser does some implicit conversion with 10^precision
            std::string decimalDivisor = "1";
            decimalDivisor.resize(precision + 1, '0');

            // This is the multiplier for the decimal part
            BigInteger multiplier(decimals);

            // This is the divisor for the decimal part
            BigInteger divisor(decimalDivisor);

            BigInteger decimalproduct;
            decimalproduct = base * multiplier;
            decimalproduct /= divisor;

            // Add the computed decimal part
            // to product
            product += decimalproduct;
        }
    }

    // Normalize to 64 chars hex with "0x" prefix
    std::stringstream sss;
    sss << (_prefix == dev::HexPrefix::Add ? "" : "0x") << std::setw(64) << std::setfill('0') << std::hex
       << product;

    std::string target = sss.str();
    boost::algorithm::to_lower(target);
    return target;
}
// Convert hex string to bytes
std::vector<unsigned char> hex_to_bytes(const std::string& hex) {
    size_t len = hex.length();
    if (len % 2 != 0) throw std::invalid_argument("Invalid hex string length");
    std::vector<unsigned char> bytes;
    bytes.reserve(len / 2);
    for (size_t i = 0; i < len; i += 2) {
        unsigned char hi = static_cast<unsigned char>(std::tolower(hex[i]));
        unsigned char lo = static_cast<unsigned char>(std::tolower(hex[i + 1]));
        hi = (hi > '9') ? (hi - 'a' + 10) : (hi - '0');
        lo = (lo > '9') ? (lo - 'a' + 10) : (lo - '0');
        bytes.push_back((hi << 4) | lo);
    }
    return bytes;
}

// Custom versa_hash using pre-created context
inline void versa_hash(secp256k1_context* ctx, const unsigned char* data, uint32_t len, unsigned char* out) {
    unsigned char h1[32], h2[32], h3[32], sig[64], finalH[32];
    SHA256(data, len, h1);
    SHA256(h1, 32, h2);
    SHA256(h2, 32, h3);
    secp256k1_schnorr_sign(ctx, sig, h3, h2, NULL, NULL);
    SHA256(sig, 64, finalH);
    std::memcpy(out, finalH, 32);
}

class EthStratumClient {
public:
    EthStratumClient(const std::string& serverIp, uint16_t port)
        : m_ioContext(), m_socket(m_ioContext), m_serverIp(serverIp), m_port(port) {
        // Pre-create per-thread contexts
        m_ctxs.reserve(m_numThreads);
        for (int i = 0; i < m_numThreads; ++i) {
            m_ctxs.push_back(secp256k1_context_create(SECP256K1_CONTEXT_SIGN));
        }
        std::ios::sync_with_stdio(false);
    }

    ~EthStratumClient() {
        for (auto ctx : m_ctxs)
            secp256k1_context_destroy(ctx);
        disconnect();
    }

    bool connectToServer() {
        try {
            tcp::resolver resolver(m_ioContext);
            auto endpoints = resolver.resolve(m_serverIp, std::to_string(m_port));
            boost::asio::connect(m_socket, endpoints);
            std::cout << "Connected to " << m_serverIp << ":" << m_port << std::endl;
            subscribe();
        } catch (std::exception& e) {
            std::cerr << "Connection error: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    void run() {
        std::thread listener([this]() { listenForJobs(); });
        listener.detach();
        // Keep main thread alive
        while (true) std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

private:
    static constexpr int m_numThreads = 8;
    boost::asio::io_context m_ioContext;
    tcp::socket m_socket;
    std::string m_serverIp;
    uint16_t m_port;
    std::vector<secp256k1_context*> m_ctxs;

    std::atomic<bool> m_newJobFlag{false};
    std::atomic<int>  m_activeWorkers{0};

    std::string m_targetDifficulty;
    std::string m_extranonce;
    unsigned short m_idx{0};

    void disconnect() {
        if (m_socket.is_open()) {
            boost::system::error_code ec;
            m_socket.close(ec);
            if (ec) std::cerr << "Error closing socket: " << ec.message() << std::endl;
        }
    }

    bool sendMessage(const json& jmsg) {
        if (!m_socket.is_open()) return false;
        std::string msg = jmsg.dump() + "\n";
        boost::system::error_code ec;
        boost::asio::write(m_socket, boost::asio::buffer(msg), ec);
        if (ec) {
            std::cerr << "Send failed: " << ec.message() << std::endl;
            return false;
        }
        return true;
    }

    void subscribe() {
        json j = { {"id",1}, {"jsonrpc","2.0"}, {"method","mining.subscribe"}, {"params", json::array()} };
        sendMessage(j);
    }

    void authorize() {
        json j = { {"id",3}, {"jsonrpc","2.0"}, {"method","mining.authorize"},
                   {"params", json::array({address + "." + workerid,"X"})} };
        sendMessage(j);
    }

    void listenForJobs() {
        boost::asio::streambuf buf;
        while (true) {
            boost::system::error_code ec;
            size_t n = boost::asio::read_until(m_socket, buf, "\n", ec);
            if (ec) { std::cerr << "Receive error: " << ec.message() << std::endl; break; }
            std::istream is(&buf);
            std::string line;
            std::getline(is, line);
            if (line.empty()) continue;
            try {
                auto j = json::parse(line);
                if (j.contains("method") && j["method"] == "mining.notify") {
                    std::cout << "New Job: " << j["params"][0] << std::endl;

                    // Signal currently running workers to stop
                    m_newJobFlag.store(true, std::memory_order_release);

                    // Launch a detached manager thread to wait and start the next job
                    auto paramsCopy = j["params"];
                    std::thread([this, paramsCopy]() {
                        // Wait for workers to finish
                        while (m_activeWorkers.load(std::memory_order_acquire) > 0)
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        // Start new job
                        processJob(paramsCopy);
                    }).detach();
                }
                else if (j.contains("method") && j["method"] == "mining.set_difficulty") {
                    m_targetDifficulty = getTargetFromDiff(j["params"][0].get<double>());
                    std::cout << "Target Diffculty: " << m_targetDifficulty << std::endl;
                }
                else if (j.contains("id") && j["id"] == 1 && j.contains("result")) {
                    m_extranonce = j["result"][1].get<std::string>();
                    authorize();
                }
                else if (j.contains("id") && j["id"] == 3 && j.contains("result")) {
                    if(j["result"] == true){
                        std::cout << "Authorized worker XXX" << std::endl;
                    }
                    else{
                        std::cout << "Failed to Authorize worker XXX" << std::endl;
                    }
                }
                else if (j.contains("id") && j["id"] >= 40) {
                    if(j["result"] == true){
                        std::cout << "Share Accepted" << std::endl;
                    }
                    else{
                        std::cout << "Share Rejected" << std::endl;
                    }
                }
            } catch (...) { }
        }
    }

    void submitShare(const std::string& nonceHex, const std::string& jobid) {
        json j = { {"id", 40 + (++m_idx)}, {"jsonrpc","2.0"}, {"method","mining.submit"},
                   {"params", json::array({address, jobid, nonceHex})} };
        sendMessage(j);
    }

    void processJob(const json& params) {
        m_newJobFlag.store(false, std::memory_order_release);
        // Decode header and boundary once
        auto headerHex = params[1].get<std::string>();
        auto headerBytes = hex_to_bytes(headerHex);
        std::vector<unsigned char> boundaryBytes = hex_to_bytes(m_targetDifficulty);
        std::vector<unsigned char> extranonceBytes = hex_to_bytes(m_extranonce);
        uint64_t startNonce = 0; // or parse from params if provided

        std::atomic<bool> stopFlag{false};

        // Per-thread scratch buffers
        struct Scratch { unsigned char msg[49], hash[32]; };
        std::vector<Scratch> scratch(m_numThreads);

        // Initialize prefixes
        for (int t = 0; t < m_numThreads; ++t) {
            std::memcpy(scratch[t].msg, headerBytes.data(), 32);
            // extranonce suffix will be copied per-loop
        }

        #pragma omp parallel num_threads(m_numThreads) shared(stopFlag)
        {
            int tid = omp_get_thread_num();
            auto* ctx = m_ctxs[tid];
            auto& S = scratch[tid];
            uint64_t local = startNonce + tid;

            m_activeWorkers.fetch_add(1, std::memory_order_acq_rel);
            while (!m_newJobFlag.load(std::memory_order_acquire)) {
                // pack prefix byte
                S.msg[32] = 0x10;
                // pack nonce (8 bytes big-endian)
                for (int i = 0; i < 8; ++i)
                    S.msg[33 + i] = (local >> ((7 - i) * 8)) & 0xFF;
                // copy extranonce bytes
                std::memcpy(S.msg + 41, extranonceBytes.data(), extranonceBytes.size());

                // hash+sign+final
                versa_hash(ctx, S.msg, sizeof(S.msg), S.hash);
                // reverse
                std::reverse(S.hash, S.hash + 32);

                if (std::memcmp(S.hash, boundaryBytes.data(), 32) <= 0) {
                    // found
                    std::ostringstream oss;
                    oss << std::hex << std::setw(16) << std::setfill('0') << local;
                    std::string nonceHex = oss.str() + m_extranonce;
                    submitShare(nonceHex, params[0].get<std::string>());
                    stopFlag.store(true);
                }

                local += m_numThreads;
            }
            m_activeWorkers.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
};

int main(int argc, char **argv) {
    EthStratumClient client("67.220.70.51", 31588);
    if (!client.connectToServer()) return EXIT_FAILURE;
    client.run();
    return EXIT_SUCCESS;
}
