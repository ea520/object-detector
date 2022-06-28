source /opt/intel/openvino_2022/setupvars.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=release
cmake --build build
