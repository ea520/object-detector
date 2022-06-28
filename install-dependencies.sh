mlpack=$(ldconfig -p | grep libmlpack)
zbar=$(ldconfig -p | grep libzbar)
if [ -z "$mlpack" ]; then # if the string is empty, mlpack wasn't found
	echo "Not found mlpack"
	echo "Attempting install..."
	sudo apt-get install libmlpack-dev
else
	echo "Found mlpack!"
fi

if [ -z "$zbar" ]; then # if the string is empty, zbar wasn't found
	echo "Not found zbar"
	echo "Attempting install..."
	sudo apt-get install libzbar-dev
else
	echo "Found zbar!"
fi

install_openvino(){
	echo "Not found OpenVino"
	echo "Attempting install..."
	wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
	sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
	rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
	echo "deb https://apt.repos.intel.com/openvino/2022 bionic main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2022.list
	sudo apt update && sudo apt install openvino openvino-opencv
	pushd /opt/intel/openvino_2022/install_dependencies/
	sudo -E ./install_NEO_OCL_driver.sh -y
	popd
}
SETUPVARS="/opt/intel/openvino_2022/setupvars.sh"
if [ -f "$SETUPVARS" ]; then # assume openvino is installed if this file exists
	source "$SETUPVARS" >> /dev/null
	if [ -z "$OpenCV_DIR" ]; then 
		install_openvino
	else
		echo "Found OpenVino!"
	fi
else
	install_openvino
fi