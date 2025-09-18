#!/usr/bin/env python3
"""
Convert the detailed review document to Word format
"""

import subprocess
import os

def convert_to_word():
    """Convert the markdown document to Word format"""
    
    input_file = "CLIENT_DETAILED_REVIEW_DOCUMENT.md"
    output_file = "CLIENT_DETAILED_REVIEW_DOCUMENT.docx"
    
    print("🔄 Converting markdown to Word document...")
    
    try:
        # Try using pandoc if available
        result = subprocess.run([
            "pandoc", 
            input_file, 
            "-o", 
            output_file,
            "--reference-doc=template.docx" if os.path.exists("template.docx") else ""
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully converted to: {output_file}")
            return True
        else:
            print(f"❌ Pandoc error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("⚠️ Pandoc not found. Creating alternative format...")
        return create_alternative_format()

def create_alternative_format():
    """Create an alternative format if pandoc is not available"""
    
    print("📝 Creating HTML format for easy conversion to Word...")
    
    # Read the markdown file
    with open("CLIENT_DETAILED_REVIEW_DOCUMENT.md", "r") as f:
        content = f.read()
    
    # Convert to HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Client Detailed Review Document</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: monospace;
        }}
        .verification {{
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .formula {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-family: monospace;
        }}
        .example {{
            background-color: #e7f3ff;
            border: 1px solid #b3d9ff;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        ul, ol {{
            margin: 10px 0;
            padding-left: 30px;
        }}
        li {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
{content}
</body>
</html>
"""
    
    # Save HTML file
    with open("CLIENT_DETAILED_REVIEW_DOCUMENT.html", "w") as f:
        f.write(html_content)
    
    print("✅ Created HTML format: CLIENT_DETAILED_REVIEW_DOCUMENT.html")
    print("📝 To convert to Word:")
    print("   1. Open the HTML file in a web browser")
    print("   2. Print to PDF or copy to Word")
    print("   3. Or use online HTML to Word converters")
    
    return True

def main():
    """Main function"""
    
    print("📄 CLIENT DETAILED REVIEW DOCUMENT CONVERTER")
    print("=" * 50)
    
    if not os.path.exists("CLIENT_DETAILED_REVIEW_DOCUMENT.md"):
        print("❌ Source file not found: CLIENT_DETAILED_REVIEW_DOCUMENT.md")
        return
    
    # Try to convert to Word
    if convert_to_word():
        print("\n✅ Conversion completed successfully!")
        print("\n📁 Files created:")
        print("   - CLIENT_DETAILED_REVIEW_DOCUMENT.md (source)")
        if os.path.exists("CLIENT_DETAILED_REVIEW_DOCUMENT.docx"):
            print("   - CLIENT_DETAILED_REVIEW_DOCUMENT.docx (Word format)")
        if os.path.exists("CLIENT_DETAILED_REVIEW_DOCUMENT.html"):
            print("   - CLIENT_DETAILED_REVIEW_DOCUMENT.html (HTML format)")
        
        print("\n🎯 Next steps:")
        print("   1. Send the Word/HTML file to your client")
        print("   2. Client can review step-by-step")
        print("   3. Client can verify calculations using the formulas")
        print("   4. Client can use the verification checklist")
        
    else:
        print("❌ Conversion failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
