#!/usr/bin/env python3
"""
Test Template Fix - Verify that all template placeholders are resolved
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
epub_spec = importlib.util.spec_from_file_location("epub_builder", "utils/epub_builder.py")
epub_module = importlib.util.module_from_spec(epub_spec)
epub_spec.loader.exec_module(epub_module)
build_epub = epub_module.build_epub

def test_template_fix():
    """Test that template placeholders are properly resolved"""
    
    print('🧪 Testing Template Placeholder Fix')
    print('=' * 50)
    
    # Use a translation directory with job metadata
    source_path = 'data/custom_translations/master_20250716_1928_oai_gpt4o'
    output_path = 'template_test.epub'
    
    # Remove any existing test file
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f'Removed existing {output_path}')
    
    print(f'📂 Source: {source_path}')
    print(f'📖 Output: {output_path}')
    print('🎯 Novel Slug: 永生_kanunu')
    print()
    
    # Create EPUB with the fixed template
    success, message = build_epub(
        chapter_dir=source_path,
        output_path=output_path,
        title='Template Test - Eternal Life',
        author='Meng Shenji',
        translator='Claude AI (Template Test)',
        novel_slug='永生_kanunu',
        include_images=True
    )
    
    print(f'✅ Success: {success}')
    print(f'📝 Message: {message}')
    
    if success and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024*1024)
        print(f'📦 Generated EPUB: {size_mb:.2f} MB')
        
        # Extract and check the metadata report
        try:
            with zipfile.ZipFile(output_path, 'r') as epub_zip:
                files = epub_zip.namelist()
                metadata_files = [f for f in files if 'metadata_report' in f]
                
                if metadata_files:
                    metadata_file = metadata_files[0]
                    print(f'📋 Found metadata file: {metadata_file}')
                    
                    # Read metadata content
                    metadata_content = epub_zip.read(metadata_file).decode('utf-8')
                    
                    # Check for unresolved placeholders
                    unresolved = []
                    lines = metadata_content.split('\n')
                    for i, line in enumerate(lines):
                        if '{' in line and '}' in line:
                            # Extract potential placeholders
                            start = line.find('{')
                            end = line.find('}', start)
                            if start != -1 and end != -1:
                                placeholder = line[start:end+1]
                                unresolved.append(f"Line {i+1}: {placeholder}")
                    
                    if unresolved:
                        print(f'❌ Found {len(unresolved)} unresolved placeholders:')
                        for placeholder in unresolved[:5]:  # Show first 5
                            print(f'   • {placeholder}')
                        if len(unresolved) > 5:
                            print(f'   ... and {len(unresolved) - 5} more')
                        return False
                    else:
                        print('✅ All template placeholders resolved!')
                        
                        # Show sample of actual values
                        print()
                        print('📊 Sample Analytics (first 10 lines):')
                        for i, line in enumerate(lines[:10]):
                            if line.strip() and not line.startswith('#'):
                                print(f'   {line.strip()}')
                        
                        return True
                else:
                    print('❌ No metadata report found in EPUB')
                    return False
                    
        except Exception as e:
            print(f'❌ Error reading EPUB: {e}')
            return False
            
    else:
        print('❌ EPUB creation failed or file not found')
        return False

if __name__ == '__main__':
    success = test_template_fix()
    print()
    print('🎉 Template fix test PASSED!' if success else '❌ Template fix test FAILED!')
    sys.exit(0 if success else 1)