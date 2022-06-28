# object-detector
## Linux install
### Install easier dependencies
```bash
sudo apt-get install libmlpack-dev libzbar-dev
```
#### Install intel's inference engine [instructions here](https://docs.openvino.ai/2022.1/openvino_docs_install_guides_installing_openvino_apt.html):
 ```bash
wget https://apt.repos.intel.com/intel-gpg-keys/
```
```bash
GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
```
```bash
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
```
```bash
echo "deb https://apt.repos.intel.com/openvino/2022 bionic main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2022.list
```
```bash
sudo apt update && sudo apt install openvino openvino-opencv 
```
```bash
cd /opt/intel/openvino_2022/install_dependencies/
sudo -E ./install_NEO_OCL_driver.sh -y # support for iGPU
```

#### Compile the code
```bash
git clone https://github.com/ea520/object-detector
cd object-detector
```
```bash
cmake -S . -B temp -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$(pwd) -DCMAKE_BUILD_TYPE=release
```
```bash
cmake --build temp
```
```bash
rm -r temp
```
