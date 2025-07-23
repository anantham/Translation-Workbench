#!/usr/bin/env python3
"""
Production EPUB Test - Simulates User Workflow
Tests the exact same workflow that users experience through Experimentation Lab
"""

import os
import sys
import zipfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import epub_builder directly without going through utils package
import importlib.util
import sys

epub_spec = importlib.util.spec_from_file_location("epub_builder", "utils/epub_builder.py")
epub_module = importlib.util.module_from_spec(epub_spec)
epub_spec.loader.exec_module(epub_module)
build_epub = epub_module.build_epub

def test_production_epub_creation():
    """Test production EPUB creation with user workflow simulation"""
    
    print('🧪 Testing Production EPUB Creation (User Workflow Simulation)')
    print('=' * 60)
    
    # Use a translation directory that simulates what users would actually create
    source_path = 'data/custom_translations/master_20250716_1928_oai_gpt4o'
    output_path = 'user_test_with_images.epub'
    
    # Remove any existing test file
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f'Removed existing {output_path}')
    
    print(f'📂 Source: {source_path}')
    print(f'📖 Output: {output_path}')
    print(f'🎯 Novel Slug: 永生_kanunu')
    print()
    
    # Simulate the exact call that the fixed Experimentation Lab now makes
    success, message = build_epub(
        chapter_dir=source_path,  # Correct parameter name
        output_path=output_path,
        title='User Test - Eternal Life',
        author='Meng Shenji',
        translator='Claude AI (Test)',
        novel_slug='永生_kanunu',
        include_images=True  # This is the critical fix we implemented
    )
    
    print(f'✅ Success: {success}')
    print(f'📝 Message: {message}')
    
    if success and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024*1024)
        print(f'📦 Generated EPUB: {size_mb:.2f} MB')
        
        # Quick validation - extract and check for images
        try:
            with zipfile.ZipFile(output_path, 'r') as epub_zip:
                files = epub_zip.namelist()
                image_files = [f for f in files if '/images/' in f and any(ext in f.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif'])]
                print(f'🖼️  Images in EPUB: {len(image_files)} files')
                
                if image_files:
                    print('📋 Sample image files:')
                    for img in image_files[:5]:  # Show first 5
                        print(f'   • {img}')
                    if len(image_files) > 5:
                        print(f'   ... and {len(image_files) - 5} more')
                
                # Check for gallery HTML
                gallery_files = [f for f in files if 'gallery' in f.lower()]
                print(f'🎨 Gallery files: {len(gallery_files)}')
                
                # Check content structure
                content_files = [f for f in files if f.startswith('content/')]
                print(f'📄 Content files: {len(content_files)}')
                
                # Success validation
                if image_files and gallery_files:
                    print()
                    print('🎉 SUCCESS: Production EPUB contains images and galleries!')
                    print('✅ The Experimentation Lab fix is working correctly')
                    return True
                else:
                    print()
                    print('❌ FAILED: EPUB missing images or galleries')
                    return False
                    
        except Exception as e:
            print(f'❌ Error reading EPUB: {e}')
            return False
            
    else:
        print('❌ EPUB creation failed or file not found')
        return False

if __name__ == '__main__':
    success = test_production_epub_creation()
    sys.exit(0 if success else 1)