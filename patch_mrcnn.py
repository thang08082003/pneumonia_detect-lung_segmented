"""Patch mrcnn for Keras 2.10+ compatibility"""
import site
import os

# Find mrcnn installation path
mrcnn_path = None
for p in site.getsitepackages():
    potential_path = os.path.join(p, 'mrcnn', 'model.py')
    if os.path.exists(potential_path):
        mrcnn_path = potential_path
        break

if mrcnn_path:
    print(f'Found mrcnn at: {mrcnn_path}')
    
    # Read the file
    with open(mrcnn_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the problematic import
    old_import = 'import keras.engine as KE'
    new_import = '''import keras.engine as KE
# Patch for Keras 2.10+
try:
    KE.Layer
except AttributeError:
    from keras.layers import Layer as _Layer
    KE.Layer = _Layer'''
    
    if old_import in content and 'Patch for Keras 2.10' not in content:
        content = content.replace(old_import, new_import)
        with open(mrcnn_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print('Patched successfully!')
    else:
        print('Already patched or different format')
else:
    print('mrcnn not found')

