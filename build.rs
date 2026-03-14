use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Rebuild when C++ sources change.
    for path in [
        "src/cpp/cpu.h",
        "src/cpp/cpu.cpp",
        "src/cpp/cpu_gen.h",
        "src/cpp/cpu_gen.cpp",
        "src/cpp/cpu_jit.h",
        "src/cpp/cpu_jit.cpp",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
    // Rebuild when the compiler or LLVM installation changes.
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=LLVM_CONFIG");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"));
    let llvm_config = llvm_config_path();
    let llvm_cxxflags = llvm_config_flags(&llvm_config, &["--cxxflags"]);
    let llvm_ldflags = llvm_config_flags(&llvm_config, &["--ldflags"]);
    let llvm_libs = llvm_config_flags(
        &llvm_config,
        &[
            "--system-libs",
            "--libs",
            "core",
            "orcjit",
            "native",
            "passes",
        ],
    );

    let mut objects = Vec::new();
    for source in [
        "src/cpp/cpu.cpp",
        "src/cpp/cpu_gen.cpp",
        "src/cpp/cpu_jit.cpp",
    ] {
        let object = out_dir.join(
            Path::new(source)
                .file_name()
                .expect("source file should have a file name")
                .to_string_lossy()
                .replace(".cpp", ".o"),
        );
        let mut command = cxx_compiler();
        command.arg("-std=c++17").arg("-fPIC").arg("-fexceptions");
        for flag in &llvm_cxxflags {
            if flag == "-std=c++17" || flag == "-fno-exceptions" {
                continue;
            }
            command.arg(flag);
        }
        command.arg("-c").arg(source).arg("-o").arg(&object);
        run(&mut command);
        objects.push(object);
    }

    let lib = out_dir.join("libcast_cpu_ffi.a");
    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&lib);
    for object in &objects {
        ar.arg(object);
    }
    run(&mut ar);

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=cast_cpu_ffi");
    for flag in llvm_ldflags.iter().chain(llvm_libs.iter()) {
        if let Some(path) = flag.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={path}");
        } else if let Some(lib) = flag.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
    println!("cargo:rustc-link-lib={}", cxx_stdlib_name());
}

/// Returns a `Command` for the C++ compiler, respecting the `CXX` environment variable.
fn cxx_compiler() -> Command {
    if let Some(cxx) = env::var_os("CXX") {
        return Command::new(cxx);
    }
    Command::new("c++")
}

/// Returns the system C++ standard library name for the current target.
fn cxx_stdlib_name() -> &'static str {
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("apple") || target.contains("freebsd") || target.contains("openbsd") {
        "c++"
    } else {
        "stdc++"
    }
}

/// Locates `llvm-config`, preferring `LLVM_CONFIG` env var then Homebrew then `PATH`.
fn llvm_config_path() -> PathBuf {
    if let Some(path) = env::var_os("LLVM_CONFIG") {
        return PathBuf::from(path);
    }
    let homebrew = PathBuf::from("/opt/homebrew/opt/llvm/bin/llvm-config");
    if homebrew.exists() {
        return homebrew;
    }
    PathBuf::from("llvm-config")
}

/// Runs `llvm-config` with `args` and returns the whitespace-split output tokens.
fn llvm_config_flags(llvm_config: &Path, args: &[&str]) -> Vec<String> {
    let output = Command::new(llvm_config)
        .args(args)
        .output()
        .unwrap_or_else(|err| {
            panic!(
                "failed to run `{}` with args {:?}: {}",
                llvm_config.display(),
                args,
                err
            )
        });
    if !output.status.success() {
        panic!(
            "`{}` with args {:?} failed with {}",
            llvm_config.display(),
            args,
            output.status
        );
    }
    String::from_utf8(output.stdout)
        .expect("llvm-config output should be utf-8")
        .split_whitespace()
        .map(ToOwned::to_owned)
        .collect()
}

/// Runs `command`, panicking with a descriptive message on failure.
fn run(command: &mut Command) {
    let program = command.get_program().to_string_lossy().into_owned();
    let args = command
        .get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let status = command
        .status()
        .unwrap_or_else(|err| panic!("failed to run `{}` with args {:?}: {}", program, args, err));
    if !status.success() {
        panic!(
            "command `{}` with args {:?} failed with {}",
            program, args, status
        );
    }
}
