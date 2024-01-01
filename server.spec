# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_dynamic_libs, get_package_paths, collect_data_files
import os, sys
datas = []
datas += copy_metadata('tqdm')
datas += copy_metadata('regex')
datas += copy_metadata('requests')
datas += copy_metadata('packaging')
datas += copy_metadata('filelock')
datas += copy_metadata('numpy')
datas += copy_metadata('tokenizers')
datas += copy_metadata('huggingface-hub')
datas += copy_metadata('safetensors')
datas += copy_metadata('accelerate')
datas += copy_metadata('pyyaml')
datas += copy_metadata('xformers')
binaries = collect_dynamic_libs('bitsandbytes')

binaries += collect_dynamic_libs('torch')
datas += collect_data_files('torch')

package_path = get_package_paths('llama_cpp')[0]
datas += collect_data_files('llama_cpp')
if os.name == 'nt':  # Windows
    dll_path = os.path.join(package_path, 'llama_cpp', 'llama.dll')
    datas.append((dll_path, 'llama_cpp'))
elif sys.platform == 'darwin':  # Mac
    so_path = os.path.join(package_path, 'llama_cpp', 'llama.dylib')
    datas.append((so_path, 'llama_cpp'))
elif os.name == 'posix':  # Linux
    so_path = os.path.join(package_path, 'llama_cpp', 'libllama.so')
    datas.append((so_path, 'llama_cpp'))

datas += collect_data_files('xformers')

hiddenimports = []
hiddenimports.append('xformers')
hiddenimports.append("autogptq_cuda_256")
hiddenimports.append("autogptq_cuda_64")

a = Analysis(
    ['server.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    module_collection_mode={
        # requires source .py files for JIT
        'torch': 'pyz+py',
        'bitsandbytes': 'pyz+py',
        'transformers': 'pyz+py',
        'datasets': 'pyz+py',
        'optimum': 'pyz+py'
    }
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['favicon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='server',
)
