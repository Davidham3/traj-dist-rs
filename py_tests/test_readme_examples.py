"""
Test Python code examples in README.md

This script extracts all Python code blocks from README.md and executes them,
ensuring that code examples in the documentation can run normally.
"""

import os
import re
import sys
import tempfile
from pathlib import Path


def extract_python_code_from_readme():
    """Extract all Python code blocks from README.md"""
    readme_path = Path(__file__).parent.parent / "README.md"

    if not readme_path.exists():
        raise FileNotFoundError(f"README.md not found at {readme_path}")

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract code blocks between ```python and ```
    python_blocks = re.findall(r"```python\n(.*?)```", content, re.DOTALL)

    return python_blocks


def execute_python_code(code, index):
    """Execute Python code and return results"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        # Add necessary imports
        enhanced_code = code
        f.write(enhanced_code)
        temp_file = f.name

    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(Path(__file__).parent.parent),  # Execute in project root directory
        )

        success = result.returncode == 0
        return success, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Execution timeout"
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_readme_python_examples():
    """Test all Python code examples in README"""
    print("=" * 60)
    print("Testing README.md Python Code Examples")
    print("=" * 60)

    try:
        python_blocks = extract_python_code_from_readme()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        assert False, f"README.md not found: {e}"

    if not python_blocks:
        print("⚠️  Warning: No Python code blocks found in README.md")
        return

    print(f"\nFound {len(python_blocks)} Python code block(s) in README.md\n")

    all_passed = True
    failed_examples = []

    for i, code in enumerate(python_blocks, 1):
        print(f"Testing Example {i}...")
        print("-" * 60)

        # Display code snippet (first 200 characters)
        code_preview = code.strip().split("\n")[0][:200]
        print(f"Code preview: {code_preview}...")

        # Execute code
        passed, stdout, stderr = execute_python_code(code, i)

        if passed:
            print(f"✅ Example {i} passed")
            if stdout:
                print(f"   Output: {stdout.strip()[:200]}")
        else:
            print(f"❌ Example {i} failed")
            all_passed = False
            failed_examples.append(i)

            if stderr:
                print(f"   Error: {stderr[:500]}")

        print()

    print("=" * 60)
    if all_passed:
        print("✅ All README Python examples passed!")
    else:
        print(f"❌ {len(failed_examples)} example(s) failed: {failed_examples}")
    print("=" * 60)

    assert all_passed, f"{len(failed_examples)} README example(s) failed"


if __name__ == "__main__":
    success = test_readme_python_examples()
    assert success, "Some README examples failed"
    sys.exit(0)
