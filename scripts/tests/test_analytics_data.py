#!/usr/bin/env python3
"""
Test Analytics Data - Verify that real metrics are showing up in EPUBs
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

def test_analytics_data():
    """Test that real analytics data is showing up in EPUBs"""
    
    print('🧪 Testing Real Analytics Data in EPUB')
    print('=' * 50)
    
    # Use a translation directory with job metadata
    source_path = 'data/custom_translations/master_20250716_1928_oai_gpt4o'
    output_path = 'analytics_test.epub'
    
    # Remove any existing test file
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f'Removed existing {output_path}')
    
    # Create EPUB
    success, message = build_epub(
        chapter_dir=source_path,
        output_path=output_path,
        title='Analytics Test - Eternal Life',
        author='Meng Shenji',
        translator='Claude AI (Analytics Test)',
        novel_slug='永生_kanunu',
        include_images=True
    )
    
    if success and os.path.exists(output_path):
        # Extract and check the metadata report for real data
        try:
            with zipfile.ZipFile(output_path, 'r') as epub_zip:
                files = epub_zip.namelist()
                metadata_files = [f for f in files if 'metadata_report' in f]
                
                if metadata_files:
                    metadata_file = metadata_files[0]
                    metadata_content = epub_zip.read(metadata_file).decode('utf-8')
                    
                    # Look for specific analytics lines
                    lines = metadata_content.split('\n')
                    analytics_found = {}
                    
                    for line in lines:
                        if 'Total Time' in line and '3063.6s' in line:
                            analytics_found['total_time'] = line.strip()
                        elif 'API Cost' in line and '$6.0831' in line:
                            analytics_found['cost'] = line.strip()
                        elif 'Processing Speed' in line and 'chapters/minute' in line:
                            analytics_found['speed'] = line.strip()
                        elif 'Input Tokens' in line and '2,087,056' in line:
                            analytics_found['input_tokens'] = line.strip()
                        elif 'Output Tokens' in line and '86,544' in line:
                            analytics_found['output_tokens'] = line.strip()
                        elif 'gpt-4o' in line:
                            analytics_found['model'] = line.strip()
                    
                    print('🔍 Analytics Data Found:')
                    for key, value in analytics_found.items():
                        print(f'   ✅ {key}: {value}')
                    
                    # Check that we found the key metrics
                    required_metrics = ['total_time', 'cost', 'input_tokens', 'output_tokens']
                    missing = [m for m in required_metrics if m not in analytics_found]
                    
                    if missing:
                        print(f'❌ Missing metrics: {missing}')
                        return False
                    else:
                        print(f'✅ All required analytics data found!')
                        return True
                        
                else:
                    print('❌ No metadata report found')
                    return False
                    
        except Exception as e:
            print(f'❌ Error reading EPUB: {e}')
            return False
            
    else:
        print('❌ EPUB creation failed')
        return False

if __name__ == '__main__':
    success = test_analytics_data()
    print()
    print('🎉 Analytics test PASSED!' if success else '❌ Analytics test FAILED!')
    sys.exit(0 if success else 1)