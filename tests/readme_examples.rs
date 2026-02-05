/// 测试 README.md 中的 Rust 代码示例
///
/// 此模块从 README.md 中提取所有 Rust 代码块并编译它们，
/// 确保文档中的代码示例能够正常编译。
use std::fs;
use std::path::PathBuf;

/// 从 README.md 中提取所有 Rust 代码块
fn extract_rust_code_from_readme() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut readme_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    readme_path.push("README.md");

    if !readme_path.exists() {
        return Err(format!("README.md not found at {:?}", readme_path).into());
    }

    let content = fs::read_to_string(&readme_path)?;

    // 提取 ```rust 和 ``` 之间的代码块
    let mut rust_blocks = Vec::new();
    let mut in_rust_block = false;
    let mut current_block = String::new();

    for line in content.lines() {
        if line.trim() == "```rust" {
            in_rust_block = true;
            current_block.clear();
        } else if line.trim() == "```" && in_rust_block {
            in_rust_block = false;
            if !current_block.trim().is_empty() {
                rust_blocks.push(current_block.clone());
            }
        } else if in_rust_block {
            current_block.push_str(line);
            current_block.push('\n');
        }
    }

    Ok(rust_blocks)
}

/// 测试单个 Rust 代码块是否能够编译

fn test_rust_code_compilation(code: &str, _index: usize) -> Result<(), String> {
    use std::fs;

    use std::path::PathBuf;

    // 创建临时测试文件

    let temp_dir = tempfile::tempdir().map_err(|e| format!("Failed to create temp dir: {}", e))?;

    // 构建 Cargo 项目结构

    let cargo_dir = temp_dir.path().join("readme_test");

    let src_dir = cargo_dir.join("src");

    fs::create_dir_all(&src_dir).map_err(|e| format!("Failed to create directories: {}", e))?;

    // 获取项目根目录的绝对路径

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // 创建 Cargo.toml

    let cargo_toml = format!(
        r#"

[package]

name = "readme_test"

version = "0.1.0"

edition = "2021"



[dependencies]

traj-dist-rs = {{ path = "{}" }}

"#,
        project_root.display()
    );

    fs::write(cargo_dir.join("Cargo.toml"), cargo_toml)
        .map_err(|e| format!("Failed to write Cargo.toml: {}", e))?;

    // 解析代码，提取 use 语句和主代码

    let use_statements: Vec<&str> = code
        .lines()
        .filter(|line| line.trim().starts_with("use "))
        .collect();

    let main_code: Vec<&str> = code
        .lines()
        .filter(|line| !line.trim().starts_with("use ") && !line.trim().starts_with("fn main()"))
        .collect();

    // 构建完整的 main.rs

    let full_code = if code.contains("fn main()") {
        // 如果代码已经有 main 函数，直接使用

        code.trim().to_string()
    } else {
        // 否则，包裹在 main 函数中

        format!(
            r#"

{}



fn main() {{

{}

}}

"#,
            use_statements.join("\n"),
            main_code.join("\n        ")
        )
    };

    fs::write(src_dir.join("main.rs"), full_code)
        .map_err(|e| format!("Failed to write main.rs: {}", e))?;

    // 尝试编译代码

    let output = std::process::Command::new("cargo")
        .arg("check")
        .current_dir(&cargo_dir)
        .env("CARGO_TARGET_DIR", temp_dir.path().join("target"))
        .output()
        .map_err(|e| format!("Failed to execute cargo: {}", e))?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);

        let stdout = String::from_utf8_lossy(&output.stdout);

        Err(format!(
            "Compilation failed:\nstdout:\n{}\n\nstderr:\n{}",
            stdout, stderr
        ))
    }
}

#[test]
fn test_readme_rust_examples() {
    println!("Testing README.md Rust Code Examples");
    println!("====================================");

    match extract_rust_code_from_readme() {
        Ok(rust_blocks) => {
            if rust_blocks.is_empty() {
                println!("Warning: No Rust code blocks found in README.md");
                return;
            }

            println!(
                "\nFound {} Rust code block(s) in README.md\n",
                rust_blocks.len()
            );

            let mut all_passed = true;
            let mut failed_examples = Vec::new();

            for (i, code) in rust_blocks.iter().enumerate() {
                println!("Testing Example {}...", i + 1);
                println!("-----------------------------------");

                // 显示代码片段（前 200 字符）
                let code_preview = code.lines().next().unwrap_or("").trim();
                println!("Code preview: {}...", code_preview);

                // 测试编译
                match test_rust_code_compilation(code, i + 1) {
                    Ok(_) => {
                        println!("✅ Example {} passed", i + 1);
                    }
                    Err(e) => {
                        println!("❌ Example {} failed", i + 1);
                        println!("   Error: {}", e);
                        all_passed = false;
                        failed_examples.push(i + 1);
                    }
                }

                println!();
            }

            println!("====================================");
            if all_passed {
                println!("✅ All README Rust examples compiled successfully!");
            } else {
                panic!(
                    "❌ {} example(s) failed to compile: {:?}",
                    failed_examples.len(),
                    failed_examples
                );
            }
        }
        Err(e) => {
            panic!("Error: {}", e);
        }
    }
}
