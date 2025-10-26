from setuptools import setup, find_packages


setup(name='',
      version='1.0.0',
      description='Adaptive Sparse Contrastive Learning for Unsupervised Object Re-identification',
      author='DINGYUAN ZHENG',
     
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu==1.6.4'],
      packages=find_packages()
      )
