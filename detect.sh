args="$@" # store the args as the next line changes the value of $@
source /opt/intel/openvino_2022/setupvars.sh
echo "$args"
./build/detect $args
