"""Full patch for mrcnn TensorFlow 2.x compatibility"""
import site
import os
import re

# Find mrcnn installation path
mrcnn_path = None
for p in site.getsitepackages():
    potential_path = os.path.join(p, 'mrcnn', 'model.py')
    if os.path.exists(potential_path):
        mrcnn_path = potential_path
        break

if mrcnn_path:
    print(f'Found mrcnn at: {mrcnn_path}')
    
    with open(mrcnn_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # List of replacements for TF2 compatibility
    replacements = [
        # Fix tf.log -> tf.math.log
        (r'\btf\.log\b', 'tf.math.log'),
        
        # Fix tf.sets.set_intersection -> tf.sets.intersection
        (r'tf\.sets\.set_intersection', 'tf.sets.intersection'),
        
        # Fix tf.to_float -> tf.cast(..., tf.float32)
        # This is tricky, need manual fix
        
        # Fix resize_images
        (r'tf\.image\.resize_images', 'tf.image.resize'),
        
        # Fix is -> == for string comparison
        (r"if os\.name is 'nt':", "if os.name == 'nt':"),
    ]
    
    modified = False
    for pattern, replacement in replacements:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            modified = True
            print(f'Replaced: {pattern} -> {replacement}')
    
    if modified:
        with open(mrcnn_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print('Patching complete!')
    else:
        print('No changes needed or already patched')
else:
    print('mrcnn not found')

