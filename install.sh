sudo apt update && sudo apt install libboost-dev libssl-dev libcrypto++-dev nlohmann-json3-dev
git clone https://github.com/Bitcoin-ABC/secp256k1.git
cd secp256k1
./autogen.sh
mkdir build
cd build
../configure
make -j4
sudo make install
cd ../..
g++ versa_hash.cpp -O3 -fopenmp -lssl -lcrypto -L/usr/local/lib -lsecp256k1 -static -o pub
./pub 0xF7B8CDa3831B03cC20d0208611ECf83E21E57edb worker005
