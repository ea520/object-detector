# object-detector
## Ububtu18+ install
Clone the repository
```bash
git clone https://github.com/ea520/object-detector
```
```bash
cd object-detector
```
Install the dependencies
```bash
# required packages: libmlpack-dev libzbar-dev openvino openvino-opencv
./install-dependencies.sh
```
Compile the code 
```bash
# sudo apt-get install g++-8
CXX=g++-8 # or any other compiler that supports c++17 (g++-7 probably won't work)
./compile.sh
```

Run the code 
```bash
./detect.sh # "./detect.sh --help" will show the command line options
```