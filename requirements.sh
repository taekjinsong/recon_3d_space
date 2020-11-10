sudo apt install -y python-pydot python-pydot-ng graphviz

conda install -y pytorch==1.2.0 torchvision==0.4.0
conda install -y pyqt=5
conda install -y ipython

pip install cython
pip install opencv-python opencv-contrib-python pillow pycocotools matplotlib
pip install numba numpy scikit-image
pip install vispy
pip install webcolors sklearn
pip install pycuda==2018.1.1
pip install graphviz

cd external/DCNv2
rm _ext.cpython-*
rm -r build
python setup.py build develop
cd ../..





