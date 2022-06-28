source /opt/intel/openvino_2022/setupvars.sh
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=release
cmake --build build
