import os
import shutil
from pathlib import Path
import subprocess
import sys

# Fix for OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def run_pip_uninstall(package):
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', package])
        print(f"Successfully uninstalled {package}")
    except subprocess.CalledProcessError:
        print(f"Package {package} was not installed")

def cleanup_deepface():
    # 1. Uninstall packages
    packages_to_remove = [
        'deepface',
        'tensorflow',
        'tf-keras',
        'keras',
        'h5py'
    ]
    
    print("Uninstalling DeepFace and related packages...")
    for package in packages_to_remove:
        run_pip_uninstall(package)

    # 2. Remove cached model directories
    directories_to_remove = [
        Path.home() / ".deepface",  # DeepFace models
        Path.home() / ".keras",     # TensorFlow/Keras models
    ]

    for directory in directories_to_remove:
        if directory.exists():
            try:
                shutil.rmtree(directory)
                print(f"Deleted directory: {directory}")
            except Exception as e:
                print(f"Error deleting {directory}: {e}")

    # 3. Remove TensorFlow analyzer file
    tf_analyzer = Path('models/tensorflow_analyzer.py')
    if tf_analyzer.exists():
        try:
            tf_analyzer.unlink()
            print(f"Deleted file: {tf_analyzer}")
        except Exception as e:
            print(f"Error deleting {tf_analyzer}: {e}")

    print("\nCleanup completed!")
    print("\nRemaining project structure:")
    for path in sorted(Path('.').rglob('*.py')):
        print(f"  {path}")

if __name__ == "__main__":
    cleanup_deepface() 